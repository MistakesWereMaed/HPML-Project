import os
import pandas as pd
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from models import PICPModel

PATH_WEIGHTS = "../Models/Weights"
PATH_METRICS = "../Models/Metrics"

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def load_checkpoint(path, model, optimizer):
    try:
        checkpoint = torch.load(path, map_location="cuda")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        return checkpoint["epoch"] + 1, checkpoint["best_val_loss"], checkpoint["metrics"]
    except FileNotFoundError:
        return 0, float("inf"), {"train_loss": [], "val_loss": [], "epoch_time": []}

def save_checkpoint(model, optimizer, epoch, val_loss, metrics, path):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_loss": val_loss,
        "metrics": metrics
    }, path)

def train_epoch(rank, model, train_loader, loss_function, optimizer, scaler, use_progressbar = True):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Rank {rank} - Training", leave=True) if use_progressbar else train_loader
    # Training Loop
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(f"cuda:{rank}"), targets.to(f"cuda:{rank}")
        optimizer.zero_grad()
        # Mixed precision training
        predictions = model(inputs)
        loss = loss_function(predictions, targets)
        # Backward pass
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        # Progress update
        total_loss += loss.item()
        if use_progressbar:
            progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(train_loader)

def validate(rank, model, val_loader, loss_function):
    model.eval()
    total_loss = 0.0
    # Validation Loop
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(rank), targets.to(rank)
            predictions = model(inputs)
            # Progress update
            loss = loss_function(predictions, targets)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def train_experiment(rank, epochs=5, **model_kwargs):
    # Unpack kwargs
    model = model_kwargs["model"]
    loss_function = model_kwargs["loss_function"]
    optimizer = model_kwargs["optimizer"]
    train_ds = model_kwargs["data"][0]
    val_ds = model_kwargs["data"][1]
    model.to(f"cuda:{rank}")
    # Training Loop
    scaler = torch.GradScaler()
    for epoch in range(epochs):
        train_ds.sampler.set_epoch(epoch)
        avg_train_loss = train_epoch(rank, model, train_ds, loss_function, optimizer, scaler, use_progressbar=False)
        print(f"GPU {rank} - Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss}")
    # Compute Validation Loss
    avg_val_loss = validate(rank, model, val_ds, loss_function)
    print(f"GPU {rank} - Validation Loss: {avg_val_loss}")
    return avg_val_loss
    
# Main Training Function
def train(rank, world_size, model_class, epochs=5, patience=10):
    model_kwargs = model_class.initialize_model()
    # Unpack kwargs
    name = model_kwargs["name"]
    model = model_kwargs["model"]
    loss_function = model_kwargs["loss_function"]
    optimizer = model_kwargs["optimizer"]
    train_ds = model_kwargs["data"][0]
    val_ds = model_kwargs["data"][1]
    # Initialize DDP
    setup(rank, world_size)
    model.to(f"cuda:{rank}")
    model = DDP(model, device_ids=[rank])
    # Load Checkpoint if Available
    start_epoch, best_val_loss, metrics = load_checkpoint(f"{PATH_WEIGHTS}/{name}-current.ckpt", model, optimizer)
    patience_counter = 0
    # Training Loop
    scaler = torch.GradScaler()
    for epoch in range(start_epoch, epochs):
        train_ds.sampler.set_epoch(epoch)
        # Compute Training and Validation Loss
        avg_train_loss = train_epoch(rank, model, train_ds, loss_function, optimizer, scaler)
        avg_val_loss = validate(rank, model, val_ds, loss_function)
        # Use all reduce to get metrics
        metrics_tensor = torch.tensor([avg_train_loss, avg_val_loss], dtype=torch.float, device=f"cuda:{rank}")
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        # Average the values by dividing by the number of GPUs
        avg_train_loss = metrics_tensor[0].item() / world_size
        avg_val_loss = metrics_tensor[1].item() / world_size
        # Update on only GPU 0 
        if rank == 0:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss} - Val Loss: {avg_val_loss}")
            metrics["train_loss"].append(avg_train_loss)
            metrics["val_loss"].append(avg_val_loss)
            # Checkpointing
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0

                save_checkpoint(model, optimizer, epoch, best_val_loss, metrics, f"{PATH_WEIGHTS}/{name}-best.ckpt")
                print(f"New best model saved!")

            patience_counter += 1
            save_checkpoint(model, optimizer, epoch, avg_val_loss, metrics, f"{PATH_WEIGHTS}/{name}-current.ckpt")
            # Early Stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs. Best val loss: {best_val_loss}")
                break

    if rank == 0:
        df = pd.DataFrame(metrics)
        df.to_csv(f"{PATH_METRICS}/{name}-metrics.csv", index=False)
    cleanup()

def main(args):
    model_type = args["model"]
    epochs = args["epochs"]
    patience = args["patience"]
    # Select model
    match model_type:
        case "PINN":
            model_class = PICPModel
        #case "GNN": 
        #case "FNO":
        case _:
            raise ValueError(f"Unknown model type")
    # Launch training on multiple GPUs
    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(train, args=(world_size, model_class, epochs, patience), nprocs=world_size)
    else:
        train(0, world_size, model_class, epochs, patience)
    
# Launch Training on Multiple GPUs
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a model with specific parameters.")
    parser.add_argument("--model", type=str, required=True, help="Type of model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    args = parser.parse_args()
    
    main(vars(args))