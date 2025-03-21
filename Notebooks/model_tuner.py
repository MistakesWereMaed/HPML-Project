import json
import sys
import argparse
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from hyperopt import fmin, tpe, Trials
from hyperopt.exceptions import AllTrialsFailed
from functools import partial
from models import PICPModel
from model_trainer import train_experiment, setup, cleanup

PATH_VAL = "../Data/Processed/Val.nc"
PATH_TEST = "../Data/Processed/Test.nc"

PATH_PARAMS = "../Models/Params"

def objective(params, model_class, rank):
    # Initialize model
    model_kwargs = model_class.initialize_model(path_train=PATH_VAL, path_val=PATH_TEST, **params)
    # Train and get validation loss
    val_loss = train_experiment(rank, **model_kwargs)
    print(f"GPU {rank}: Validation Loss: {val_loss}\n")
    return {'loss': val_loss, 'status': 'ok'}

def main(rank, world_size, args):
    # Setup distributed processing
    setup(rank, world_size)
    # Select model
    model = args["model"]
    match model:
        case "PINN":
            model_class = PICPModel
        #case "GNN": 
        #case "FNO":
        case _:
            raise ValueError(f"Unknown model type")
    # Define hyperparameter space
    space = model_class.get_hyperparam_space()
    objective_with_args = partial(objective, model_class=model_class, rank=rank)
    # Perform hyperparameter tuning
    try:
        trials = Trials()
        best = fmin(fn=objective_with_args, space=space, algo=tpe.suggest, max_evals=args["trials"], trials=trials)
    except AllTrialsFailed:
        print("All trials failed.")
        cleanup()
        sys.exit()
    best_loss = trials.best_trial['result']['loss']
    best_loss = torch.nan_to_num(best_loss, 999999, 999999, 999999)
    best_params = best
    # Convert loss to tensor for all_reduce
    best_loss_tensor = torch.tensor(best_loss, dtype=torch.float, device=f"cuda:{rank}")
    dist.all_reduce(best_loss_tensor, op=dist.ReduceOp.MIN)
    # Check if this GPU has the best loss
    if best_loss == best_loss_tensor.item():
        # Save best hyperparameters
        with open(f"{PATH_PARAMS}/{model}.json", "w") as f:
            json.dump({'loss': best_loss, 'params': best_params}, f)
        print(f"GPU {rank}: Best hyperparameters saved with loss {best_loss_tensor.item():.4f}")

    cleanup()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a model with specific parameters.")
    parser.add_argument("--model", type=str, required=True, help="Type of model")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials")
    args = parser.parse_args()
    # Launch tuning on multiple GPUs
    world_size = torch.cuda.device_count()
    print(f"Tuning on {world_size} GPU(s)...")
    if world_size > 1:
        mp.spawn(main, args=(world_size, vars(args)), nprocs=world_size, join=True)
    else:
        main(0, world_size, vars(args))