import os
import time
import argparse
import pandas as pd

import torch
import torch.distributed as dist

from tqdm import tqdm
from mpi4py import MPI
from axonn import axonn as ax
from torch.nn.parallel import DistributedDataParallel as DDP
from models import initialize_model
from data_loader import load_data, get_image_size

PATH_TRAIN = "../Data/Processed/Train.nc"
PATH_VAL = "../Data/Processed/Val.nc"

PATH_WEIGHTS = "../Models/Weights"
PATH_METRICS = "../Models/Metrics"

def save_checkpoint(path, model, optimizer, epoch, metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if hasattr(model, "module"):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    checkpoint = {
        "epoch": epoch,
        "model_state": model_state,
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
    }

    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer, experiment):
    defaults = (0, {"train_loss": [], "val_loss": [], "epoch": [], "time": []})
    if experiment:
        return defaults

    try:
        checkpoint = torch.load(path, map_location="cuda")

        # Handle DDP wrapping if present
        model_state = checkpoint["model_state"]
        if hasattr(model, "module"):
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)

        optimizer.load_state_dict(checkpoint["optimizer_state"])

        return checkpoint["epoch"] + 1, checkpoint["metrics"]

    except FileNotFoundError:
        return defaults

def validate(rank, val_loader, model, loss_function, warmup=False):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            #inputs, targets = inputs.to(rank), targets.to(rank)
            predictions = model(inputs)
            loss = loss_function(predictions, targets)
            total_loss += loss.item()
            if warmup: return
    
    return total_loss / len(val_loader)

def train_epoch(rank, train_loader, model, scaler, loss_function, optimizer, show_progress_bar=True, warmup=False):
    model.train()
    final_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Training", leave=False) if show_progress_bar else train_loader
    
    for inputs, targets in progress_bar:
        #inputs, targets = inputs.to(rank), targets.to(rank)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(inputs)
            loss = loss_function(output, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad(set_to_none=True)
        final_loss = loss.item()

        if warmup: return
        if show_progress_bar:
            progress_bar.set_postfix(loss=loss.item())

    return final_loss

def train(model_type, epochs, path_train, path_val, downsampling_scale=2, experiment=False, world_size=None, show_progress_bar=True, hyperparameters=None):
    # Initialize multiprocessing environment
    rank = MPI.COMM_WORLD.Get_rank()
    world_size = world_size if world_size is not None else MPI.COMM_WORLD.Get_size()
    torch.backends.cudnn.benchmark = True
    if rank == 0: print("Initializing AxoNN...")
    # Initialize model
    image_size = get_image_size(path_train, downsampling_scale)
    model, optimizer, loss_function, batch_size = initialize_model(image_size, model_type, hyperparameters)
    start_epoch, metrics = load_checkpoint(f"{PATH_WEIGHTS}/{model.name}.ckpt", model, optimizer, experiment)

    scaler = torch.amp.GradScaler("cuda")
    model.to(rank)
    name = model.name
    # Initialize AxoNN
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, init_method="env://")
    ax.init(world_size, 1, 1, 1, 1)
    # Load data
    if rank == 0: 
        print("Loading Data...")
        val_loader = load_data(0, 1, path_val, batch_size)
    train_loader = load_data(rank, world_size, path_train, batch_size)
    # Warmup pass
    if rank == 0: 
        print("Warming Up...\n")
        validate(rank, val_loader, model, loss_function, True)
    train_epoch(rank, train_loader, model, scaler, loss_function, optimizer, False, True)
    # Training loop
    if rank == 0: print("Training...")
    model = DDP(model, device_ids=[rank])
    total_time = 0.0
    val_loss = float("inf")

    try:
        for epoch in range(start_epoch, epochs):
            if rank == 0:
                # Training process
                start_time = time.perf_counter()
                train_loss = train_epoch(rank, train_loader, model, scaler, loss_function, optimizer, show_progress_bar)
                end_time = time.perf_counter()

                # Validate
                val_loss = validate(rank, val_loader, model, loss_function)
                time_taken = end_time - start_time
                total_time += time_taken

                # Update metrics
                metrics["train_loss"].append(train_loss)
                metrics["val_loss"].append(val_loss)
                metrics["epoch"].append(epoch)
                metrics["time"].append(time_taken)

                # Progress update
                print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Time: {time_taken:.4f} seconds")
            else: train_epoch(rank, train_loader, model, scaler, loss_function, optimizer, False)

        # Save checkpoint
        if rank == 0:
            print("Training Complete\n")
            print(f"Final Val Loss: {val_loss:.4f} - Training Time: {total_time:.1f} seconds")
            pd.DataFrame(metrics).to_csv(f"{PATH_METRICS}/{name}_{world_size}.csv", index=False)
            # if not experiment: save_checkpoint(f"{PATH_WEIGHTS}/{name}.ckpt", model, optimizer, epochs, metrics)

            return val_loss, total_time
        return 0, 0

    except Exception as e:
        print(e)
        return float("inf"), float("inf")

    finally:
        # Cleanup
        #if rank == 0: val_loader._iterator._shutdown_workers()
        #train_loader._iterator._shutdown_workers()
        dist.destroy_process_group()
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="Train a model with specific parameters.")

    parser.add_argument("--model", type=str, required=True, help="Type of model")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--downsampling", type=int, default=2, help="Downsampling reduction scale")

    args = parser.parse_args()

    model_type = args.model
    epochs = args.epochs
    downsampling_scale = args.downsampling

    val_loss, time_taken = train(
        model_type=model_type, epochs=epochs, 
        path_train=PATH_TRAIN, path_val=PATH_VAL, downsampling_scale=downsampling_scale, 
        experiment=False, show_progress_bar=True
    )

if __name__ == "__main__":
    main()