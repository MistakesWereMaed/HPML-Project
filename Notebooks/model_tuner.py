import json
import argparse
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from hyperopt import fmin, tpe, Trials
from functools import partial
from models import PICPModel
from model_trainer import train_experiment, setup, cleanup

PATH_PARAMS = "../Models/Params"

def objective(params, model_class, rank):
    print(f"Beginning Trial on GPU {rank} with params:")
    print(params)
    # Initialize model
    model_kwargs = model_class.initialize_model(tuning=True, **params)
    # Train and get validation loss
    val_loss = train_experiment(rank, **model_kwargs)
    print(f"GPU {rank}: Validation Loss: {val_loss:.4f}\n")
    return {'loss': val_loss, 'status': 'ok'}

def main(rank, world_size, args):
    # Setup distributed processing
    setup(rank, world_size)
    print(f"GPU {rank}: Tuning hyperparameters...")
    # Select model
    model = args["model"]
    match model:
        case "PINN":
            model_class = PICPModel
        #case "GNN": 
        #case "FNO":
        case _:
            raise ValueError(f"Unknown model type: {args.model}")
    # Define hyperparameter space
    space = model_class.get_hyperparam_space()
    objective_with_args = partial(objective, model_class=model_class, rank=rank)
    # Perform hyperparameter tuning
    trials = Trials()
    best = fmin(fn=objective_with_args, space=space, algo=tpe.suggest, max_evals=args["trials"], trials=trials)
    best_loss = trials.best_trial['result']['loss']
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
    if world_size > 1:
        mp.spawn(main, args=(world_size, vars(args)), nprocs=world_size, join=True)
    else:
        main(0, world_size, vars(args))