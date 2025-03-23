import os
import time
import pandas as pd
import argparse
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from models import PICPModel
from data_loader import load_dataset

PATH_TRAIN = "../Data/Processed/Test.nc"
PATH_VAL = "../Data/Processed/Val.nc"

PATH_WEIGHTS = "../Models/Weights"
PATH_METRICS = "../Models/Metrics"

def get_unused_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return str(s.getsockname()[1])

def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def load_checkpoint(path, model, optimizer):
    try:
        checkpoint = torch.load(path, map_location="cuda")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        return checkpoint["epoch"] + 1, checkpoint["metrics"]
    except FileNotFoundError:
        return 0, {"train_loss": [], "val_loss": [], "epoch": []}

def save_checkpoint(model, optimizer, epoch, metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model_state_dict = model.state_dict()
    checkpoint = {
        "epoch": epoch,
        "model_state": model_state_dict,
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics
    }
    if dist.get_rank() == 0:
        torch.save(checkpoint, path)
        print(f"Checkpoint saved at {path}")

def validate(rank, model, val_loader, loss_function):
    model.eval()
    total_loss = torch.tensor(0.0, device=rank)

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(rank), targets.to(rank)
            predictions = model(inputs)
            loss = loss_function(predictions, targets)
            total_loss += loss.detach()

    # Average loss across all ranks
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    total_loss /= dist.get_world_size()

    return total_loss.item()

def train_epoch(rank, model, train_loader, loss_function, optimizer):
    total_loss = torch.tensor(0.0, device=rank)

    progress_bar = tqdm(train_loader, desc="Training", leave=False) if rank == 0 else train_loader
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(rank), targets.to(rank)
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = loss_function(predictions, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach()

        if rank == 0:
            progress_bar.set_postfix(loss=loss.item())

    # Average loss across all ranks
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    total_loss /= dist.get_world_size()

    return total_loss.item()

def train_experiment(rank, world_size, port, model, loss_function, optimizer, train_ds, val_ds, batch_size, epochs=5):
    setup(rank, world_size, port)
    
    model.to(rank)
    model = FSDP(model)
    
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, drop_last=True)

    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, sampler=val_sampler)

    start_time = time.perf_counter()
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        model.train()
        avg_train_loss = train_epoch(rank, model, train_loader, loss_function, optimizer)
    end_time = time.perf_counter()
    avg_val_loss = validate(rank, model, val_loader, loss_function)

    cleanup()
    if rank == 0: print(f"Validation Loss: {avg_val_loss}")

    return avg_val_loss, end_time - start_time

def train(rank, world_size, port, model, name, loss_function, optimizer, train_ds, val_ds, batch_size, epochs=5):
    setup(rank, world_size, port)
    
    model.to(rank)
    model = FSDP(model)
    start_epoch, metrics = load_checkpoint(f"{PATH_WEIGHTS}/{name}.ckpt", model, optimizer)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, drop_last=True)

    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, sampler=val_sampler)

    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        model.train()
        avg_train_loss = train_epoch(rank, model, train_loader, loss_function, optimizer)
        avg_val_loss = validate(rank, model, val_loader, loss_function)

        if rank == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss} - Val Loss: {avg_val_loss}")

            metrics["train_loss"].append(avg_train_loss)
            metrics["val_loss"].append(avg_val_loss)
            metrics["epoch"].append(epoch)

            save_checkpoint(model, optimizer, epoch, metrics, f"{PATH_WEIGHTS}/{name}.ckpt")

    if rank == 0:
        pd.DataFrame(metrics).to_csv(f"{PATH_METRICS}/{name}.csv", index=False)
    
    cleanup()

def main(args):
    model_type = args["model"]
    epochs = args["epochs"]

    match model_type:
        case "PINN":
            model_class = PICPModel
        case _:
            raise ValueError(f"Unknown model type")

    params = model_class.load_params()
    train_ds, image_size = load_dataset(path=PATH_TRAIN, input_days=params["input_days"], target_days=params["target_days"])
    val_ds, _ = load_dataset(path=PATH_VAL, input_days=params["input_days"], target_days=params["target_days"])

    model_kwargs = model_class.initialize_model(image_size, params)
    name = model_kwargs["name"]
    model = model_kwargs["model"]
    loss_function = model_kwargs["loss_function"]
    optimizer = model_kwargs["optimizer"]

    port = get_unused_port()
    world_size = torch.cuda.device_count()
    
    if world_size > 1:
        mp.spawn(train, args=(world_size, port, model, name, loss_function, optimizer, train_ds, val_ds, params["batch_size"], epochs), nprocs=world_size, join=True)
    else:
        train(0, world_size, port, model, name, loss_function, optimizer, train_ds, val_ds, params["batch_size"], epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with specific parameters.")
    parser.add_argument("--model", type=str, required=True, help="Type of model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    args = parser.parse_args()
    
    main(vars(args))