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
from data_loader import load_data, get_image_size

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

def validate(val_loader, model, loss_function, show_progress_bar=True, warmup=False):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validating", leave=False) if show_progress_bar else val_loader
        
        for inputs, targets in progress_bar:
            predictions = model(inputs)
            loss = loss_function(predictions, targets)
            total_loss += loss.item()
            
            if warmup: return
            if show_progress_bar:
                progress_bar.set_postfix(loss=loss.item())
    
    return total_loss / len(val_loader)

def train_epoch(train_loader, model, scaler, loss_function, optimizer, show_progress_bar=True):
    model.train()
    final_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Training", leave=False) if show_progress_bar else train_loader
    
    for inputs, targets in progress_bar:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(inputs)
            loss = loss_function(output, targets)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad(set_to_none=True)
        final_loss = loss.item()

        if show_progress_bar:
            progress_bar.set_postfix(loss=loss.item())

    return final_loss

def train_process(rank, world_size, name, model, optimizer, loss_function, path_train, path_val, batch_size, epochs, start_epoch, metrics, experiment, show_progress_bar):
    # Initialize DDP
    if rank == 0: print("\nInitializing DDP...")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    model = DDP(model.to(rank), device_ids=[rank])
    scaler = torch.amp.GradScaler("cuda")
    # Load data
    if rank == 0: 
        print("Pre-loading data...")
        val_loader = load_data(rank, 1, path_val, batch_size)
    train_loader = load_data(rank, world_size, path_train, batch_size)
    # Warm-up step
    if rank == 0: 
        print("Starting Warm-up...\n")
        validate(val_loader, model, loss_function, False, True)
    validate(train_loader, model, loss_function, False, True)
    # Training loop
    if rank == 0: print("Training...")
    try:
        for epoch in range(start_epoch, epochs):
            # Skip logs and metrics unless rank 0
            if rank != 0: train_epoch(train_loader, model, scaler, loss_function, optimizer, False)
            else:
                start_time = time.perf_counter()
                train_loss = train_epoch(train_loader, model, scaler, loss_function, optimizer, show_progress_bar)
                end_time = time.perf_counter()
                # Validate
                val_loss = validate(val_loader, model, loss_function, show_progress_bar)
                time_taken = end_time - start_time
                # Update metrics
                metrics["train_loss"].append(train_loss)
                metrics["val_loss"].append(val_loss)
                metrics["epoch"].append(epoch)
                metrics["time"].append(time_taken)
                # Progress update
                print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Time: {time_taken:.4f} seconds")
        # Save checkpoint
        if rank == 0: 
            print("Training Complete\n")
            pd.DataFrame(metrics).to_csv(f"{PATH_METRICS}/{name}.csv", index=False)
            if not experiment: save_checkpoint(f"{PATH_WEIGHTS}/{name}.ckpt", model, optimizer, epochs, metrics)

    except Exception as e: print(e)
    # Cleanup
    finally:
        train_loader._iterator._shutdown_workers()
        if rank == 0: val_loader._iterator._shutdown_workers()
        dist.destroy_process_group()
        torch.cuda.empty_cache()

def train(model_type, epochs, path_train, path_val, downsampling_scale=2, experiment=False, world_size=None, show_progress_bar=True, hyperparameters=None):
    # Initialize multiprocessing environment
    torch.backends.cudnn.benchmark = True
    mp.set_start_method("spawn", force=True)
    world_size = world_size if world_size is not None else torch.cuda.device_count()
    # Initialize model
    print("\nInitializing Model...")
    image_size = get_image_size(path_train, downsampling_scale)
    model, optimizer, loss_function, batch_size = initialize_model(image_size, model_type, hyperparameters)
    # Load checkpoint
    print("Loading Checkpoint...\n")
    start_epoch, metrics = load_checkpoint(f"{PATH_WEIGHTS}/{model.name}.ckpt", model, optimizer, experiment)
    processes = []
    # Initialize processes
    for rank in range(world_size):
        print(f"Starting Process {rank+1}...")
        # Start processes
        p = mp.Process(target=train_process, args=(
            rank, world_size, model.name, model, optimizer, loss_function, path_train, path_val, batch_size, epochs, start_epoch, metrics, experiment, show_progress_bar
        ))
        p.start()
        processes.append(p)
    for p in processes: p.join()
    # Return training results
    if os.path.exists(f"{PATH_METRICS}/{model.name}.csv"):
        metrics = pd.read_csv(f"{PATH_METRICS}/{model.name}.csv")
        return metrics["val_loss"].iloc[-1], sum(metrics["time"])
    return float('inf'), float('inf')

def main():
    parser = argparse.ArgumentParser(description="Train a model with specific parameters.")

    parser.add_argument("--model", type=str, required=True, help="Type of model")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--downsampling", type=int, default=2, help="Downsampling reduction scale")

    args = parser.parse_args()

    model_type = args.model
    epochs = args.epochs
    downsampling_scale = args.downsampling

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    val_loss, time_taken = train(
        model_type=model_type, epochs=epochs, 
        path_train=PATH_TRAIN, path_val=PATH_VAL, downsampling_scale=downsampling_scale, 
        experiment=False, show_progress_bar=True
    )
    print(f"Final Val Loss: {val_loss:.4f} - Training Time: {time_taken:.1f} seconds")

if __name__ == "__main__":
    main()