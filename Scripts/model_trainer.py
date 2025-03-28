import os
import time
import argparse
import pandas as pd
import torch

from tqdm import tqdm
from models import load_and_initialize
from data_loader import load_data

PATH_TRAIN = "../Data/Processed/Train.nc"
PATH_VAL = "../Data/Processed/Val.nc"

PATH_WEIGHTS = "../Models/Weights"
PATH_METRICS = "../Models/Metrics"

def load_checkpoint(path, model, optimizer):
    try:
        checkpoint = torch.load(path, map_location="cuda")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        return checkpoint["epoch"] + 1, checkpoint["best_val_loss"], checkpoint["metrics"]
    except FileNotFoundError:
        return 0, float("inf"), {"train_loss": [], "val_loss": [], "epoch": []}

def save_checkpoint(path, model, optimizer, epoch, metrics, best_val_loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model_state_dict = model.state_dict()
    checkpoint = {
        "epoch": epoch,
        "model_state": model_state_dict,
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
        "best_val_loss": best_val_loss
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at {path}")

def validate(val_loader, model, loss_function, show_progress_bar=True, **kwargs):
    model.cuda()
    model.eval()

    with torch.no_grad():
        total_loss = 0.0
        progress_bar = tqdm(val_loader, desc="Validating", leave=False) if show_progress_bar else val_loader
        
        for inputs, targets in progress_bar:
            inputs, targets = inputs.cuda(), targets.cuda()
            predictions = model(inputs)

            loss = loss_function(predictions, targets)
            total_loss += loss.item()

            if show_progress_bar:
                progress_bar.set_postfix(loss=loss.item())
    return total_loss / len(val_loader)

def train_batch(train_loader, scheduler, show_progress_bar, model, loss_function, optimizer, **kwargs):
    model.cuda()
    model.train()

    total_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Training", leave=False) if show_progress_bar else train_loader

    for inputs, targets in progress_bar:
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()

        predictions = model(inputs)
        loss = loss_function(predictions, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if show_progress_bar:
            progress_bar.set_postfix(loss=loss.item())

    scheduler.step()
    return total_loss / len(train_loader)

def train(model_type, epochs=10,
          path_train=None, path_val=None, downsampling_scale=2, splits=1,
          experiment=False, show_progress_bar=True, hyperparameters=None):
    # Initialize model
    model_dict = load_and_initialize(model_type=model_type, path1=path_train, path2=path_val, downsampling_scale=downsampling_scale, splits=splits, hyperparameters=hyperparameters)
    # Unpack sub-dictionaries
    model_kwargs = model_dict["model_kwargs"]
    params = model_kwargs["hyperparameters"]
    #Unpack model
    name = model_kwargs["name"]
    optimizer = model_kwargs["optimizer"]
    model = model_kwargs["model"]
    # Unpack datasets
    train_set = model_dict["datasets"][0]
    val_set = model_dict["datasets"][1]
    # Skip checkpoint for experiments
    if experiment:
        start_epoch, best_val_loss, metrics = (0, float("inf"), {"train_loss": [], "val_loss": [], "epoch": []})
    else:
        start_epoch, best_val_loss, metrics = load_checkpoint(f"{PATH_WEIGHTS}/{name}-Best.ckpt", model, optimizer)
    # Training loop
    total_time = 0.0
    total_val_loss = 0.0
    val_loader = load_data(val_set, batch_size=params["batch_size"], input_days=params["input_days"], target_days=params["target_days"])
    for epoch in range(start_epoch, epochs):
        start_time = time.perf_counter()
        for dataset, i in zip(train_set, range(splits)):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
            train_loader = load_data(dataset, batch_size=params["batch_size"], input_days=params["input_days"], target_days=params["target_days"])

            train_loss  = train_batch(train_loader=train_loader, scheduler=scheduler, show_progress_bar=show_progress_bar, **model_kwargs)
            val_loss = validate(val_loader=val_loader, show_progress_bar=show_progress_bar, **model_kwargs)

            print(f"Epoch {epoch+1}: Split {i+1}/{splits} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        end_time = time.perf_counter()
        time_taken = end_time - start_time
        print(f"Time for Epoch {epoch+1}: {time_taken:.4f} Seconds")
        # Update time and val loss for experiments
        if experiment:
            total_time += time_taken
            total_val_loss += val_loss
        # Update metrics otherwise
        else:
            metrics["train_loss"].append(train_loss)
            metrics["val_loss"].append(val_loss)
            metrics["epoch"].append(epoch)
            # Save metrics in case of interruption
            pd.DataFrame(metrics).to_csv(f"{PATH_METRICS}/{name}.csv", index=False)
            # Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(f"{PATH_WEIGHTS}/{name}-Best.ckpt", model, optimizer, epoch, metrics, best_val_loss)
            else:
                save_checkpoint(f"{PATH_WEIGHTS}/{name}-Current.ckpt", model, optimizer, epoch, metrics, best_val_loss)
    # Only experiments need a return value
    if experiment:
        return total_val_loss / epochs, total_time / epochs

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

    train(
        model_type=model_type, epochs=epochs, 
        path_train=PATH_TRAIN, path_val=PATH_VAL, downsampling_scale=downsampling_scale, splits=splits, 
        experiment=False, show_progress_bar=True
    )

if __name__ == "__main__":
    main()

    # Asynch train file load during validation