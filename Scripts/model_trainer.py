import os
import time
import argparse
import pandas as pd
import torch

from tqdm import tqdm
from models import load_and_initialize

PATH_TRAIN = "../Data/Processed/Test.nc"
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

def validate(val_loader, model, loss_function, **kwargs):
    model.cuda()
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            predictions = model(inputs)
            loss = loss_function(predictions, targets)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def train_epoch(train_loader, show_progress_bar, model, loss_function, optimizer, **kwargs):
    model.cuda()
    model.train()

    total_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Training", leave=True) if show_progress_bar else train_loader

    start_time = time.perf_counter()
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

    end_time = time.perf_counter()
        
    return total_loss / len(train_loader), end_time - start_time

def train(model_type, epochs,
          path_train, path_val, downsampling_scale, 
          experiment, show_progress_bar, hyperparameters=None):
    # Initialize model
    model_dict = load_and_initialize(model_type=model_type, path1=path_train, path2=path_val, downsampling_scale=downsampling_scale, hyperparameters=hyperparameters)
    # Unpack model parameters
    model_kwargs = model_dict["model_kwargs"]
    name = model_kwargs["name"]
    optimizer = model_kwargs["optimizer"]
    model = model_kwargs["model"]
    # Unpack loaders
    train_loader = model_dict["loaders"][0]
    val_loader = model_dict["loaders"][1]
    # Skip checkpoint for experiments
    if experiment:
        start_epoch, best_val_loss, metrics = (0, float("inf"), {"train_loss": [], "val_loss": [], "epoch": []})
    else:
        start_epoch, best_val_loss, metrics = load_checkpoint(f"{PATH_WEIGHTS}/{name}-Current.ckpt", model, optimizer)
    
    average_time = 0.0
    average_val_loss = 0.0
    for epoch in range(start_epoch, epochs):
        train_loss, time = train_epoch(train_loader=train_loader, show_progress_bar=show_progress_bar, **model_kwargs)
        val_loss = validate(val_loader=val_loader, **model_kwargs)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss}")
        # Update time and val loss for experiments
        if experiment:
            average_time += time
            average_val_loss += val_loss
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
        return average_val_loss, average_time

def main():
    parser = argparse.ArgumentParser(description="Train a model with specific parameters.")
    parser.add_argument("--model", type=str, required=True, help="Type of model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--downsampling", type=int, default=2, help="Downsampling reduction scale")

    args = parser.parse_args()
    model_type = args.model
    epochs = args.epochs
    downsampling_scale = args.downsampling

    train(model_type, epochs, PATH_TRAIN, PATH_VAL, downsampling_scale, experiment=False, show_progress_bar=True)

if __name__ == "__main__":
    main()