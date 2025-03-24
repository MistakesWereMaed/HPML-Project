import json
import pickle
import os
import argparse
import torch

from hyperopt import fmin, tpe, Trials
from functools import partial
from models import PICPModel
from model_trainer import train
from data_loader import load_dataset

PATH_VAL = "../Data/Processed/Val.nc"
PATH_TEST = "../Data/Processed/Test.nc"
PATH_PARAMS = "../Models/Params"

def train_wrapper(model_class, params, epochs):
    train_ds, image_size = load_dataset(path=PATH_VAL, input_days=params["input_days"], target_days=params["target_days"])
    val_ds, _ = load_dataset(path=PATH_TEST, input_days=params["input_days"], target_days=params["target_days"])
    
    model_kwargs = model_class.initialize_model(image_size, params)
    print("Training...")
    val_loss, _ = train(train_ds=train_ds, val_ds=val_ds, batch_size=params["batch_size"], 
                        epochs=epochs, experiment=True, show_progress_bar=False, **model_kwargs)
    
    del model_kwargs
    torch.cuda.empty_cache()
    
    return val_loss

def objective(params, model_class, epochs):
    print(params)
    try:
        val_loss = train_wrapper(model_class, params, epochs)
    except Exception as e:
        print(f"Training failed with params {params}: {e}")
        return {'loss': float('inf'), 'status': 'fail', 'params': params}
    
    return {'loss': val_loss, 'status': 'ok', 'params': params}

def main(args):
    model_type = args["model"]
    match model_type:
        case "PINN":
            model_class = PICPModel
        case _:
            raise ValueError(f"Unknown model type")
    
    space = model_class.get_hyperparam_space()
    objective_with_args = partial(objective, model_class=model_class, epochs=args["epochs"])
    
    trials_file = f"{PATH_PARAMS}/{model_type}_trials.pkl"
    
    # Load existing trials if available
    if os.path.exists(trials_file):
        with open(trials_file, "rb") as f:
            trials = pickle.load(f)
        print(f"Loaded existing trials with {len(trials.trials)} evaluations.")
    else:
        trials = Trials()

    try:
        # Continue tuning from previous progress
        max_evals = len(trials.trials) + args["trials"]
        best = fmin(fn=objective_with_args, space=space, algo=tpe.suggest, 
                    max_evals=max_evals, trials=trials)

        # Save progress
        with open(trials_file, "wb") as f:
            pickle.dump(trials, f)

        best_loss = trials.best_trial['result']['loss']
        best_params = trials.best_trial['result']['params']
        
        # Save best parameters
        with open(f"{PATH_PARAMS}/{model_type}.json", "w") as f:
            json.dump({k: v for k, v in best_params.items()}, f)

        print(f"Best hyperparameters saved with loss {best_loss:.4f}")
    
    except Exception as e:
        print(f"Hyperparameter tuning failed: {e}")
        # Save trials even if an error occurs
        with open(trials_file, "wb") as f:
            pickle.dump(trials, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with specific parameters.")
    parser.add_argument("--model", type=str, required=True, help="Type of model")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    args = parser.parse_args()
    main(vars(args))