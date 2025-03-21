import time
import csv
import torch
import torch.multiprocessing as mp

from model_trainer import train_experiment, setup, cleanup
from models import PICPModel

PATH_VAL = "../Data/Processed/Val.nc"
PATH_TEST = "../Data/Processed/Test.nc"
PATH_TIMINGS = "../Models/Timings"

def train_on_gpus(rank, world_size, model_type, epochs, return_dict):
    setup(rank, world_size)
    
    # Select model class
    match model_type:
        case "PINN":
            model_class = PICPModel
        case _:
            raise ValueError(f"Unknown model type")

    # Initialize model
    model_kwargs = model_class.initialize_model(path_train=PATH_VAL, path_val=PATH_TEST)
    model = model_kwargs["model"]
    optimizer = model_kwargs["optimizer"]
    loss_function = model_kwargs["loss_function"]
    train_ds = model_kwargs["data"][0]
    val_ds = model_kwargs["data"][1]

    device = f"cuda:{rank}"
    model.to(device)

    # Time tracking
    start_time = time.perf_counter()
    
    # Train model
    final_loss = train_experiment(rank, model, train_ds, val_ds, optimizer, loss_function, epochs)

    # End timing
    end_time = time.perf_counter()
    train_time = end_time - start_time

    print(f"GPU {rank}: Training completed in {train_time:.2f} seconds with final validation loss {final_loss:.4f}")

    cleanup()

    if rank == 0:
        return_dict["time"] = train_time
        return_dict["loss"] = final_loss

def benchmark_training(model_type="PINN", epochs=10, trials=3):
    world_sizes = [1, 2, 4]  # Number of GPUs to test
    results = []

    for world_size in world_sizes:
        if world_size > torch.cuda.device_count():
            print(f"Skipping {world_size} GPUs (only {torch.cuda.device_count()} available)")
            continue
        
        print(f"\nTraining with {world_size} GPU(s)...")

        total_time = 0
        total_loss = 0

        for trial in range(trials):
            return_dict = {"time": None, "loss": None}

            if world_size > 1:
                # Multi-GPU training using torch.multiprocessing
                mp.spawn(train_on_gpus, args=(world_size, model_type, epochs, return_dict), nprocs=world_size, join=True)
            else:
                # Single GPU training
                train_on_gpus(0, world_size, model_type, epochs, return_dict)

            total_time += return_dict["time"]
            total_loss += return_dict["loss"]

        avg_time = total_time / trials
        avg_loss = total_loss / trials
        results.append([world_size, avg_time, avg_loss])

    # Save results to CSV
    with open(f"{PATH_TIMINGS}/{model_type}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["GPUs", "Avg Training Time (s)", "Avg Validation Loss"])
        writer.writerows(results)

if __name__ == "__main__":
    benchmark_training(model_type="PINN", epochs=10, trials=3)
