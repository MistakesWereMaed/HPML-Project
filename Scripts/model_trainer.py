import os
import time
import argparse
import pandas as pd
import torch
import torch.distributed as dist
import deepspeed

from tqdm import tqdm
from models import initialize_model
from data_loader import load_data, get_image_size

PATH_TRAIN = "../Data/Processed/Train.nc"
PATH_VAL = "../Data/Processed/Val.nc"

PATH_WEIGHTS = "../Models/Weights"
PATH_METRICS = "../Models/Metrics"

def save_checkpoint(path, model, epoch, metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if hasattr(model, "module"):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    checkpoint = {
        "epoch": epoch,
        "model_state": model_state,
        "metrics": metrics,
    }

    torch.save(checkpoint, path)

def load_checkpoint(path, model, experiment):
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

        return checkpoint["epoch"] + 1, checkpoint["metrics"]

    except FileNotFoundError:
        return defaults
    
def get_config(lr):
    return {
                "train_batch_size": 8,
                "gradient_accumulation_steps": 1,
                "optimizer": {
                    "type": "Adam",
                    "params": {
                        "lr": lr
                    }
                },
                "amp_enabled": True,
                "zero_optimization": True
            }

@torch.no_grad()
def validate(val_loader, model_engine, loss_function, warmup=False):
    model_engine.eval()
    val_loss = 0.0
    total_batches = len(val_loader)

    for inputs, targets in val_loader:
        outputs = model_engine(inputs)
        loss = loss_function(outputs, targets)

        val_loss += loss.item()

        if warmup: return

    return val_loss / total_batches

def train_epoch(train_loader, model_engine, loss_function, show_progress_bar=True, warmup=False):
    model_engine.train()
    train_loss = 0.0

    iterator = tqdm(train_loader, desc="Training", leave=False) if show_progress_bar else train_loader
    for inputs, targets in iterator:

        outputs = model_engine(inputs)
        loss = loss_function(outputs, targets)

        model_engine.backward(loss)
        model_engine.step()

        train_loss = loss.item()
        if warmup: return
        if show_progress_bar: iterator.set_postfix(loss=loss.item())

    return train_loss

def train(model_type, epochs, args, path_train, path_val, downsampling_scale=2, experiment=False, world_size=None, show_progress_bar=True, hyperparameters=None):
    # Initialize multiprocessing environment
    deepspeed.init_distributed()
    torch.backends.cudnn.benchmark = True

    rank = int(os.environ.get("LOCAL_RANK"))
    world_size = world_size if world_size is not None else int(os.environ.get("WORLD_SIZE"))

    # Initialize model
    image_size = get_image_size(path_train, downsampling_scale)
    model, optimizer, loss_function, batch_size, lr = initialize_model(image_size, model_type, hyperparameters)

    # Load checkpoint if necessary
    start_epoch, metrics = load_checkpoint(f"{PATH_WEIGHTS}/{model.name}.ckpt", model, experiment)

    # Load data
    if rank == 0:
        print("Loading Data...")
        val_loader = load_data(0, 1, path_val, batch_size)
    train_loader = load_data(rank, world_size, path_train, batch_size)

    # Initialize DeepSpeed Engine
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_engine, optimizer, _, _ = deepspeed.initialize(args=args, config=get_config(lr), model=model, optimizer=optimizer, model_parameters=parameters, training_data=None)

    # Warmup pass
    if rank == 0:
        print("Warming Up...\n")
        validate(val_loader, model_engine, loss_function, True)
    train_epoch(train_loader, model_engine, loss_function, False, True)

    # Training loop
    if rank == 0: print("Training...")
    total_time = 0.0
    val_loss = float("inf")

    for epoch in range(start_epoch, epochs):
        if rank == 0:
            # Training process
            start_time = time.perf_counter()
            loss = train_epoch(train_loader, model_engine, loss_function, show_progress_bar)
            end_time = time.perf_counter()

            # Validate
            val_loss = validate(val_loader, model_engine, loss_function)
            time_taken = end_time - start_time
            total_time += time_taken

            # Update metrics
            metrics["train_loss"].append(loss)
            metrics["val_loss"].append(val_loss)
            metrics["epoch"].append(epoch)
            metrics["time"].append(time_taken)

            # Progress update
            print(f"Epoch {epoch+1} - Train Loss: {loss:.4f} - Val Loss: {val_loss:.4f} - Time: {time_taken:.4f} seconds")
        else: train_epoch(train_loader, model_engine, loss_function, False)

    # Save checkpoint
    dist.destroy_process_group()
    if rank == 0:
        print("Training Complete\n")
        print(f"Final Val Loss: {val_loss:.4f} - Training Time: {total_time:.1f} seconds")
        pd.DataFrame(metrics).to_csv(f"{PATH_METRICS}/{model.name}_{world_size}.csv", index=False)
        # if not experiment: save_checkpoint(f"{PATH_WEIGHTS}/{model.name}.ckpt", model, epochs, metrics)
        return val_loss, total_time
    return 0, 0

        

def main():
    parser = argparse.ArgumentParser(description="Train a model with specific parameters.")

    parser.add_argument("--model", type=str, required=True, help="Type of model")
    parser.add_argument("--downsampling", type=int, default=2, help="Downsampling reduction scale")

    parser.add_argument('--with_cuda', default=False, action='store_true',
                         help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema', default=False, action='store_true',
                        help='whether use exponential moving average')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e', '--epochs', default=1, type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
    
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    model_type = args.model
    epochs = args.epochs
    downsampling_scale = args.downsampling

    val_loss, time_taken = train(
        model_type=model_type, epochs=epochs, args=args,
        path_train=PATH_TRAIN, path_val=PATH_VAL, downsampling_scale=downsampling_scale, 
        experiment=False, show_progress_bar=True
    )

if __name__ == "__main__":
    main()