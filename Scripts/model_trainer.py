import os
import time
import argparse
import pandas as pd
import contextlib
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from models import load_and_initialize
from data_loader import load_data

PATH_TRAIN = "../Data/Processed/Train.nc"
PATH_VAL = "../Data/Processed/Val.nc"

PATH_WEIGHTS = "../Models/Weights"
PATH_METRICS = "../Models/Metrics"

def save_checkpoint(path, model, optimizer, epoch, metrics, best_val_loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
        "best_val_loss": best_val_loss
    }
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer, experiment):
    defaults = (0, float("inf"), {"train_loss": [], "val_loss": [], "epoch": [], "time": []})
    if experiment: return defaults

    try:
        checkpoint = torch.load(path, map_location="cuda")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        return checkpoint["epoch"] + 1, checkpoint["best_val_loss"], checkpoint["metrics"]
    except FileNotFoundError:
        print("No checkpoint found")
        return defaults

def validate(rank, val_loader, model, loss_function, show_progress_bar=True):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validating", leave=False) if show_progress_bar else val_loader
        
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(rank), targets.to(rank)
            predictions = model(inputs)
            loss = loss_function(predictions, targets)
            total_loss += loss.item()
            
            if show_progress_bar:
                progress_bar.set_postfix(loss=loss.item())
    
    return total_loss / len(val_loader)

def train_chunk(rank, train_loader, model, loss_function, optimizer, show_progress_bar=True):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Training", leave=False) if show_progress_bar else train_loader
    
    i = 0
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(rank), targets.to(rank)
        optimizer.zero_grad()
        with model.no_sync() if i < len(train_loader) - 1 else contextlib.nullcontext():
            predictions = model(inputs)
            loss = loss_function(predictions, targets)
            loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if show_progress_bar:
            progress_bar.set_postfix(loss=loss.item())
        i += 1

    return total_loss / len(train_loader)

def train_epoch(rank, chunks, val_loader, model, loss_function, optimizer, params, show_progress_bar=True):   
    chunk_count = len(chunks)
    for chunk, i in zip(chunks, range(chunk_count)):
        # Load data for the assigned split
        train_loader = load_data(chunk, params["batch_size"], params["input_days"], params["target_days"])
        # Train and validate
        show_progress_bar = rank == 0 and show_progress_bar
        train_loss = train_chunk(rank, train_loader, model, loss_function, optimizer, show_progress_bar)
        val_loss = validate(rank, val_loader, model, loss_function, show_progress_bar)
        # Log results
        if rank == 0:
            print(f"Chunk {i+1}/{chunk_count} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    # Wait for all processes and return
    dist.barrier()
    return train_loss, val_loss

def train_process(rank, world_size, name, model, optimizer, loss_function, chunks, val_set, epochs, start_epoch, best_val_loss, metrics, params, show_progress_bar):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    # Initialize DDP and train
    if rank == 0: print("Loading data...\n")
    model = DDP(model.to(rank), device_ids=[rank])
    val_loader = load_data(val_set, params["batch_size"], params["input_days"], params["target_days"])
    for epoch in range(start_epoch, epochs):
        if rank == 0: print(f"Starting Epoch {epoch+1}/{epochs}...")
        start_time = time.perf_counter()
        train_loss, val_loss = train_epoch(rank, chunks, val_loader, model, loss_function, optimizer, params, show_progress_bar)
        end_time = time.perf_counter()
        # Update metrics on rank 0
        if rank == 0:
            time_taken = end_time - start_time
            metrics["train_loss"].append(train_loss)
            metrics["val_loss"].append(val_loss)
            metrics["epoch"].append(epoch)
            metrics["time"].append(time_taken)
            print(f"Completed Epoch {epoch+1} in {time_taken:.4f} seconds\n")
            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(f"{PATH_WEIGHTS}/{name}.ckpt", model, optimizer, epoch, metrics, best_val_loss)

        dist.barrier()

    # Cleanup
    dist.destroy_process_group()
    if rank == 0: pd.DataFrame(metrics).to_csv(f"{PATH_METRICS}/{name}.csv", index=False)

def train(model_type, epochs=10, path_train=None, path_val=None, downsampling_scale=2, splits=1, experiment=False, show_progress_bar=True, hyperparameters=None):
    # Initialization
    print("Initializing Environment...")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    mp.set_start_method("spawn", force=True)
    print("Initializing Model...")
    world_size = torch.cuda.device_count()
    model_dict = load_and_initialize(model_type, path_train, path_val, downsampling_scale, splits, hyperparameters)
    # Unpack dictionaries
    model_kwargs = model_dict["model_kwargs"]
    params = model_kwargs["hyperparameters"]
    # Unpack model
    name = model_kwargs["name"]
    model = model_kwargs["model"]
    optimizer = model_kwargs["optimizer"]
    loss_function = model_kwargs["loss_function"]
    # Unpack datasets
    print("Initializing Data...")
    train_set = model_dict["datasets"][0]
    val_set = model_dict["datasets"][1]
    # Load checkpoint
    print("Loading Checkpoint...")
    start_epoch, best_val_loss, metrics = load_checkpoint(f"{PATH_WEIGHTS}/{name}-Best.ckpt", model, optimizer, experiment)
    processes = []
    # Initialize processes
    for rank in range(world_size):
        # Split data across GPUs
        print(f"Starting Process {rank+1}...")
        chunk_size = len(train_set) // world_size
        # Slice training data
        start_idx = rank * chunk_size
        end_idx = (rank + 1) * chunk_size
        chunks = train_set[start_idx:end_idx]
        # Start processes
        p = mp.Process(target=train_process, args=(
            rank, world_size, name, model, optimizer, loss_function, chunks, val_set, epochs, start_epoch, best_val_loss, metrics, params, show_progress_bar
        ))
        p.start()
        processes.append(p)
    # Collect processes
    print("All process started")
    for p in processes:
        p.join()
    print("All process finished")
    # Return training results
    metrics = pd.read_csv(f"{PATH_METRICS}/{name}.csv")
    return metrics["val_loss"].iloc[-1], sum(metrics["time"])

def main():
    parser = argparse.ArgumentParser(description="Train a model with specific parameters.")

    parser.add_argument("--model", type=str, required=True, help="Type of model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--downsampling", type=int, default=2, help="Downsampling reduction scale")
    parser.add_argument("--splits", type=int, default=1, help="Number of splits")

    args = parser.parse_args()

    model_type = args.model
    epochs = args.epochs
    downsampling_scale = args.downsampling
    splits = args.splits

    val_loss, time_taken = train(
        model_type=model_type, epochs=epochs, 
        path_train=PATH_TRAIN, path_val=PATH_VAL, downsampling_scale=downsampling_scale, splits=splits, 
        experiment=False, show_progress_bar=True
    )
    time_taken /= 60
    print(f"Val Loss: {val_loss:.4f} - Training Time: {time_taken:.4f} minutes")

if __name__ == "__main__":
    main()