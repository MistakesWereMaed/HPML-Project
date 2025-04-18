import os
import time
import argparse
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from models import initialize_model
from data_loader import load_data, get_dataset, get_image_size

PATH_TRAIN = "../Data/Processed/Train.nc"
PATH_VAL = "../Data/Processed/Val.nc"

PATH_WEIGHTS = "../Models/Weights"
PATH_METRICS = "../Models/Metrics"

def save_checkpoint(path, model, optimizer, epoch, metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
    }
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer, experiment):
    defaults = (0, {"train_loss": [], "val_loss": [], "epoch": [], "time": []})
    if experiment: return defaults

    try:
        checkpoint = torch.load(path, map_location="cuda")
        state_dict = checkpoint["model_state"]
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(
            state_dict, "module."
        )
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        return checkpoint["epoch"] + 1, checkpoint["metrics"]
    except FileNotFoundError: return defaults

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

def train_chunk(rank, train_loader, model, scaler, loss_function, optimizer, show_progress_bar=True):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Training", leave=False) if show_progress_bar else train_loader
    
    i = 0
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(rank), targets.to(rank)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(inputs)
            loss = loss_function(output, targets)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        
        if show_progress_bar:
            progress_bar.set_postfix(loss=loss.item())
        i += 1

    return total_loss / len(train_loader)

def train_epoch(rank, chunks, batch_size, model, scaler, loss_function, optimizer, show_progress_bar=True):   
    chunk_count = len(chunks)
    for chunk, i in zip(chunks, range(chunk_count)):
        # Load data for the assigned split
        train_loader = load_data(chunk, batch_size)
        # Train and validate
        show_progress_bar = rank == 0 and show_progress_bar
        train_loss = train_chunk(rank, train_loader, model, scaler, loss_function, optimizer, show_progress_bar)
        # Log results
        if rank == 0:
            print(f"Chunk {i+1}/{chunk_count} - Train Loss: {train_loss:.4f}")

    return train_loss

def train_process(rank, world_size, name, model, optimizer, loss_function, chunks, val_set, batch_size, epochs, start_epoch, metrics, experiment, show_progress_bar):
    # Initialize DDP
    if rank == 0: print("Initializing DDP...\n")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    model = DDP(model.to(rank), device_ids=[rank])
    # Training loop
    val_loader = load_data(val_set, batch_size)
    scaler = torch.amp.GradScaler("cuda")
    try:
        for epoch in range(start_epoch, epochs):
            # Skip logs and metrics unless rank 0
            if rank != 0: 
                train_epoch(rank, chunks, batch_size, model, loss_function, optimizer, False)
            else:
                print(f"Starting Epoch {epoch+1}/{epochs}...")
                # Train
                start_time = time.perf_counter()
                train_loss = train_epoch(rank, chunks, batch_size, model, scaler, loss_function, optimizer, show_progress_bar)
                end_time = time.perf_counter()
                # Validate
                val_loss = validate(rank, val_loader, model, loss_function, show_progress_bar)
                time_taken = end_time - start_time
                # Update metrics
                metrics["train_loss"].append(train_loss)
                metrics["val_loss"].append(val_loss)
                metrics["epoch"].append(epoch)
                metrics["time"].append(time_taken)
                # Progress update
                print(f"Epoch {epoch+1:3d} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Time: {time_taken:.4f} seconds\n")
        # Cleanup
        dist.destroy_process_group()
        if rank == 0: 
            pd.DataFrame(metrics).to_csv(f"{PATH_METRICS}/{name}.csv", index=False)
            if not experiment: save_checkpoint(f"{PATH_WEIGHTS}/{name}.ckpt", model, optimizer, epochs, metrics)

    except Exception as e:
        print(e)
        dist.destroy_process_group()      

def train(model_type, epochs=10, path_train=None, path_val=None, downsampling_scale=2, splits=1, experiment=False, world_size=None, show_progress_bar=True, hyperparameters=None):
    # Initialize multiprocessing environment
    mp.set_start_method("spawn", force=True)
    world_size = world_size if world_size is not None else torch.cuda.device_count()
    # Initialize model
    print("Initializing Model...")
    image_size = get_image_size(path_train, downsampling_scale)
    model, optimizer, loss_function, batch_size = initialize_model(image_size, model_type, hyperparameters)
    # Unpack datasets
    train_set = get_dataset(path=path_train, downsampling_scale=downsampling_scale, splits=splits)
    val_set = get_dataset(path=path_val, downsampling_scale=downsampling_scale, splits=1)
    # Load checkpoint
    print("Loading Checkpoint...")
    start_epoch, metrics = load_checkpoint(f"{PATH_WEIGHTS}/{model.name}.ckpt", model, optimizer, experiment)
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
            rank, world_size, model.name, model, optimizer, loss_function, chunks, val_set, batch_size, epochs, start_epoch, metrics, experiment, show_progress_bar
        ))
        p.start()
        processes.append(p)
    # Collect processes
    for p in processes:
        p.join()
    print("All process finished")
    # Return training results
    if os.path.exists(f"{PATH_METRICS}/{model.name}.csv"):
        metrics = pd.read_csv(f"{PATH_METRICS}/{model.name}.csv")
        return metrics["val_loss"].iloc[-1], sum(metrics["time"])
    return float('inf'), float('inf')

def main():
    parser = argparse.ArgumentParser(description="Train a model with specific parameters.")

    parser.add_argument("--model", type=str, required=True, help="Type of model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--downsampling", type=int, default=2, help="Downsampling reduction scale")
    parser.add_argument("--splits", type=int, default=12, help="Number of splits")

    args = parser.parse_args()

    model_type = args.model
    epochs = args.epochs
    downsampling_scale = args.downsampling
    splits = args.splits

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    val_loss, time_taken = train(
        model_type=model_type, epochs=epochs, 
        path_train=PATH_TRAIN, path_val=PATH_VAL, downsampling_scale=downsampling_scale, splits=splits, 
        experiment=False, show_progress_bar=True
    )
    time_taken /= 60
    print(f"Val Loss: {val_loss:.4f} - Training Time: {time_taken:.1f} minutes")

if __name__ == "__main__":
    main()