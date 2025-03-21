import json
import torch
import numpy as np
import torch
import sys
import argparse

from hyperopt import fmin, tpe, Trials
from hyperopt.exceptions import AllTrialsFailed
from functools import partial
from models import PICPModel
from model_trainer import train_experiment

PATH_VAL = "../Data/Processed/Val.nc"
PATH_TEST = "../Data/Processed/Test.nc"
PATH_PARAMS = "../Models/Params"

def objective(params, model_class):
    # Initialize model
    print(params)
    model_kwargs = model_class.initialize_model(path_train=PATH_VAL, path_val=PATH_TEST, **params)
    # Train and get validation loss
    val_loss = train_experiment(0, **model_kwargs)
    return {'loss': val_loss, 'status': 'ok'}

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
    objective_with_args = partial(objective, model_class=model_class)
    
    # Perform hyperparameter tuning
    try:
        trials = Trials()
        best = fmin(fn=objective_with_args, space=space, algo=tpe.suggest, max_evals=args["trials"], trials=trials)
    except AllTrialsFailed:
        print("All trials failed.")
        sys.exit()
    
    best_loss = trials.best_trial['result']['loss']
    best_params = best
    
    # Save best hyperparameters
    with open(f"{PATH_PARAMS}/{model_type}.json", "w") as f:
        json.dump({
            'loss': float(best_loss),
            'params': {k: int(v) if isinstance(v, (np.integer, torch.Tensor)) else v for k, v in best_params.items()}
        }, f)
    print(f"Best hyperparameters saved with loss {best_loss:.4f}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a model with specific parameters.")
    parser.add_argument("--model", type=str, required=True, help="Type of model")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials")
    args = parser.parse_args()
    main(vars(args))