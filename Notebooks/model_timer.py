import time
import csv
import argparse
import torch
import torch.multiprocessing as mp

from model_trainer import train_experiment, setup, cleanup
from models import PICPModel

PATH_VAL = "../Data/Processed/Val.nc"
PATH_TEST = "../Data/Processed/Test.nc"
PATH_TIMINGS = "../Models/Timings"

GPU_CONFIGS = [1, 2, 4]

def train_timing(rank, world_size, model_type, epochs, return_dict):
    # Initialize model
    model_kwargs = model_type.initialize_model(path_train=PATH_VAL, path_val=PATH_TEST)
    model = model_kwargs["model"]
    optimizer = model_kwargs["optimizer"]
    loss_function = model_kwargs["loss_function"]
    train_ds = model_kwargs["data"][0]
    val_ds = model_kwargs["data"][1]
    # Set up GPU
    setup(rank, world_size)
    device = f"cuda:{rank}"
    model.to(device)
    # Time tracking
    start_time = time.perf_counter()
    final_loss = train_experiment(rank, model, train_ds, val_ds, optimizer, loss_function, epochs)
    end_time = time.perf_counter()
    train_time = end_time - start_time
    # Save results
    if rank == 0:
        return_dict["time"] = train_time
        return_dict["loss"] = final_loss
    print(f"GPU {rank}: Training completed in {train_time:.2f} seconds with final validation loss {final_loss:.4f}")
    cleanup()

def main(args):
    model_type = args["model"]
    epochs = args["epochs"]
    trials = args["trials"]
    # Select model class
    match model_type:
        case "PINN":
            model_class = PICPModel
        case _:
            raise ValueError(f"Unknown model type")
    # Train with different GPU configurations
    results = []
    for world_size in GPU_CONFIGS:
        if world_size > torch.cuda.device_count():
            print(f"Skipping {world_size} GPUs (only {torch.cuda.device_count()} available)")
            continue
        print(f"\nTraining with {world_size} GPU(s)...")
        # Average multiple trials
        total_time = 0
        total_loss = 0
        for trial in range(trials):
            return_dict = {"time": None, "loss": None}
            # Launch training on multiple GPUs
            if world_size > 1:
                mp.spawn(train_timing, args=(world_size, model_class, epochs, return_dict), nprocs=world_size, join=True)
            else:
                train_timing(0, world_size, model_class, epochs, return_dict)
            # Aggregate results
            total_time += return_dict["time"]
            total_loss += return_dict["loss"]
        # Compute average
        avg_time = total_time / trials
        avg_loss = total_loss / trials
        results.append([world_size, avg_time, avg_loss])
    # Save results to CSV
    with open(f"{PATH_TIMINGS}/{model_type}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["GPUs", "Avg Training Time (s)", "Avg Validation Loss"])
        writer.writerows(results)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a model with specific parameters.")
    parser.add_argument("--model", type=str, required=True, help="Type of model")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials")
    args = parser.parse_args()

    main(vars(args))