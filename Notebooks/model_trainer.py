import os
import time
import pandas as pd
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from models import PICPModel
from data_loader import load_dataset

PATH_TRAIN = "../Data/Processed/Train.nc"
PATH_VAL = "../Data/Processed/Val.nc"

PATH_WEIGHTS = "../Models/Weights"
PATH_METRICS = "../Models/Metrics"

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
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at {path}")

def validate(model, val_loader, loss_function):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            predictions = model(inputs)
            loss = loss_function(predictions, targets)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def train_epoch(model, train_loader, loss_function, optimizer):
    total_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for inputs, targets in progress_bar:
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = loss_function(predictions, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        
    return total_loss / len(train_loader)

def train_experiment(model, loss_function, optimizer, train_ds, val_ds, batch_size, epochs=5):
    model.cuda()
    train_loader = DataLoader(train_ds, batch_size=batch_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    start_time = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        avg_train_loss = train_epoch(model, train_loader, loss_function, optimizer)
    end_time = time.perf_counter()
    avg_val_loss = validate(model, val_loader, loss_function)

    print(f"Validation Loss: {avg_val_loss}")

    return avg_val_loss, end_time - start_time

def train(model, name, loss_function, optimizer, train_ds, val_ds, batch_size, epochs=5):
    model.cuda()
    start_epoch, metrics = load_checkpoint(f"{PATH_WEIGHTS}/{name}.ckpt", model, optimizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    for epoch in range(start_epoch, epochs):
        model.train()
        avg_train_loss = train_epoch(model, train_loader, loss_function, optimizer)
        avg_val_loss = validate(model, val_loader, loss_function)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss} - Val Loss: {avg_val_loss}")
        metrics["train_loss"].append(avg_train_loss)
        metrics["val_loss"].append(avg_val_loss)
        metrics["epoch"].append(epoch)
        save_checkpoint(model, optimizer, epoch, metrics, f"{PATH_WEIGHTS}/{name}.ckpt")
        pd.DataFrame(metrics).to_csv(f"{PATH_METRICS}/{name}.csv", index=False)

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
    
    train(model, name, loss_function, optimizer, train_ds, val_ds, params["batch_size"], epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with specific parameters.")
    parser.add_argument("--model", type=str, required=True, help="Type of model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    args = parser.parse_args()
    main(vars(args))