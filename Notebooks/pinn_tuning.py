import torch
import torch.nn as nn
import torch.optim as optim

import json
import argparse

from linformer import LinformerSelfAttention
from hyperopt import hp
from hyperopt import fmin, tpe, Trials

from data_loader import load_dataset
from pinn import PICPModel

PATH_VAL = "../Data/Processed/Val.nc"
PATH_TEST = "../Data/Processed/Test.nc"
PATH_TRIALS = "../Models/PINN/Tuning"

def train_eval(model, loss_function, optimizer, train, val, epochs):
    def validate(model, val, loss_function, device="cuda"):
        model.eval()
        val_loss = 0.0
        # Validation loop
        with torch.no_grad():
            for inputs, targets in val:
                inputs, targets = inputs.to(device), targets.to(device)
                # Forward pass
                predictions = model(inputs)
                loss = loss_function(predictions, targets)
                val_loss += loss.item()
        # Validation summary
        avg_val_loss = val_loss / len(val)
        return avg_val_loss
    # Use CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # Gradient scaler for mixed precision
    scaler = torch.GradScaler(device)
    # Epoch training loop
    for epoch in range(epochs):
        # Epoch initialization
        total_loss = 0.0
        torch.cuda.empty_cache()
        model.train()
        # Inner training loop
        for inputs, targets in train:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            # Forward pass
            with torch.autocast(device):
                predictions = model(inputs)
                loss = loss_function(predictions, targets)
            # Backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train):.4f}")

    # Validation   
    return validate(model, val, loss_function, device)
    
def objective(params):
    print("Beginning Trial with params:")
    print(params)

    model_params = {
        "kernel_size": (5, 10),
        "linformer_k": params["linformer_k"],
        "num_heads": 1,
        "embed_dim": 128,
        "mlp_hidden_dim": 128,
        "dropout_p": 0.1,
    }

    data_params = {
        "batch_size": params["batch_size"],
        "input_days": 1,
        "target_days": 1,
        "downsampling_scale": 2
    }

    train_ds = load_dataset(PATH_VAL, **data_params)
    val_ds = load_dataset(PATH_TEST, **data_params)
    model = PICPModel(**model_params)

    training_params = {
        "model": model,
        "loss_function": nn.L1Loss(reduction='mean'),
        "optimizer": optim.Adam(model.parameters(), lr=params["learning_rate"]),
        "epochs": 5,
        "train": train_ds,
        "val": val_ds
    }

    val_loss = train_eval(**training_params)
    print(f"Validation Loss: {val_loss:.4f}\n")
    
    return {'loss': val_loss, 'status': 'ok'}


def main(args):
    print("Tuning hyperparameters...")

    num_trials = args.trials
    space = {
        "linformer_k": hp.choice("linformer_k", [64, 128, 256, 512, 768]),
        "batch_size": hp.choice("batch_size", [2, 4, 8]),
        "learning_rate": hp.uniform("learning_rate", 1e-5, 1e-3)
    }

    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=num_trials, trials=trials)

    with open(f"{PATH_TRIALS}/best_hyperparams.json", "w") as f:
        json.dump(best, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with specific parameters.")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials")
    args = parser.parse_args()

    main(args)