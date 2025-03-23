import json
import argparse
import numpy as np
import torch
import torch.multiprocessing as mp

from hyperopt import fmin, tpe, Trials
from functools import partial
from models import PICPModel
from model_trainer import train_experiment, get_unused_port
from data_loader import load_dataset

PATH_VAL = "../Data/Processed/Val.nc"
PATH_TEST = "../Data/Processed/Test.nc"
PATH_PARAMS = "../Models/Params"

def convert_value(v):
    # If v is a tensor, check if it's scalar or not
    if isinstance(v, torch.Tensor):
        if v.ndimension() == 0:  # Scalar tensor
            return v.item()
        else:  # Non-scalar tensor
            return v.tolist()
    # If it's a numpy.int64, convert it to a native Python int
    elif isinstance(v, np.int64):
        return int(v)
    return v  # Return as-is for other types

def train_wrapper(rank, world_size, port, model_class, params, epochs, return_dict):
    train_ds, image_size = load_dataset(path=PATH_VAL, input_days=params["input_days"], target_days=params["target_days"])
    val_ds, _ = load_dataset(path=PATH_TEST, input_days=params["input_days"], target_days=params["target_days"])
    # Initialize model
    model_kwargs = model_class.initialize_model(image_size, params)
    model = model_kwargs["model"]
    loss_function = model_kwargs["loss_function"]
    optimizer = model_kwargs["optimizer"]
    # Train and get validation loss
    val_loss, _ = train_experiment(rank, world_size, port, model, loss_function, optimizer, train_ds, val_ds, batch_size=params["batch_size"], epochs=epochs)
    if rank == 0:
        return_dict["loss"] = val_loss
    # Cleanup
    del model, loss_function, optimizer
    torch.cuda.empty_cache()

def objective(params, model_class, epochs):
    print(params)
    port = get_unused_port()
    world_size = torch.cuda.device_count()

    with mp.Manager() as manager:
        return_dict = manager.dict()
        # Launch training across multiple GPUs
        if world_size > 1:
            mp.spawn(train_wrapper, args=(world_size, port, model_class, params, epochs, return_dict), nprocs=world_size, join=True)
        else:
            train_wrapper(0, world_size, port, model_class, params, epochs, return_dict)
        val_loss = return_dict["loss"]

    return {'loss': val_loss, 'status': 'ok', 'params': params}

def main(args):
    # Select model
    model_type = args["model"]
    match model_type:
        case "PINN":
            model_class = PICPModel
        #case "GNN":
        #case "FNO":
        case _:
            raise ValueError(f"Unknown model type")
    # Define hyperparameter space
    space = model_class.get_hyperparam_space()
    objective_with_args = partial(objective, model_class=model_class, epochs=args["epochs"])
    # Perform hyperparameter tuning
    try:
        trials = Trials()
        best = fmin(fn=objective_with_args, space=space, algo=tpe.suggest, max_evals=args["trials"], trials=trials)
        best_loss = trials.best_trial['result']['loss']
        best_params = trials.best_trial['result']['params']
        # Save best hyperparameters
        with open(f"{PATH_PARAMS}/{model_type}.json", "w") as f:
            json.dump({k: v for k, v in best_params.items()}, f)
        print(f"Best hyperparameters saved with loss {best_loss:.4f}")
    # Handle exceptions
    except Exception as e:
        print(f"Hyperparameter tuning failed: {e}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a model with specific parameters.")
    parser.add_argument("--model", type=str, required=True, help="Type of model")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    args = parser.parse_args()
    main(vars(args))