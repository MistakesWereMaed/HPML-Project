import csv
import argparse
import torch
import torch.multiprocessing as mp

from model_trainer import train_experiment, get_unused_port
from models import PICPModel
from data_loader import load_dataset

PATH_TRAIN = "../Data/Processed/Test.nc"
PATH_VAL = "../Data/Processed/Val.nc"
PATH_TIMINGS = "../Models/Timings"

GPU_CONFIGS = [1, 2, 3, 4]

def train_timing(rank, world_size, port, model, loss_function, optimizer, train_ds, val_ds, batch_size, epochs, results_dict):
    val_loss, train_time = train_experiment(rank, world_size, port, model, loss_function, optimizer, train_ds, val_ds, batch_size, epochs)
    results_dict[rank] = (train_time / epochs, val_loss)
    # Cleanup
    del model, loss_function, optimizer
    torch.cuda.empty_cache()

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
    # Load Data
    params = model_class.load_params()
    train_ds, image_size = load_dataset(path=PATH_TRAIN, input_days=params["input_days"], target_days=params["target_days"])
    val_ds, _ = load_dataset(path=PATH_VAL, input_days=params["input_days"], target_days=params["target_days"])
    # Train with different GPU configurations
    results = []
    for world_size in GPU_CONFIGS:
        if world_size > torch.cuda.device_count():
            print(f"Skipping {world_size} GPUs (only {torch.cuda.device_count()} available)")
            continue
        print(f"\nTraining with {world_size} GPU(s)...")
        # Average multiple trials
        total_time = 0
        total_val_loss = 0
        for trial in range(trials):
            # Initialize model
            model_kwargs = model_class.initialize_model(image_size, params)
            model = model_kwargs["model"]
            loss_function = model_kwargs["loss_function"]
            optimizer = model_kwargs["optimizer"]
            # Shared dictionary for results
            with mp.Manager() as manager:
                results_dict = manager.dict()
                # Launch training on multiple GPUs
                port = get_unused_port()
                if world_size > 1:
                    mp.spawn(train_timing, args=(world_size, port, model, loss_function, optimizer, train_ds, val_ds, params["batch_size"], epochs, results_dict), nprocs=world_size, join=True)
                else:
                    train_timing(0, world_size, port, model, loss_function, optimizer, train_ds, val_ds, params["batch_size"], epochs, results_dict)
                # Aggregate results
                total_time += sum(t[0] for t in results_dict.values())
                total_val_loss += sum(t[1] for t in results_dict.values())
        # Compute averages
        avg_time = total_time / trials / world_size
        avg_val_loss = total_val_loss / trials / world_size
        results.append([world_size, avg_time, avg_val_loss])

        print(f"Average time per epoch for {world_size} GPU(s): {avg_time:.2f} seconds")
        print(f"Average validation loss for {world_size} GPU(s): {avg_val_loss:.4f}")
    # Save results to CSV
    with open(f"{PATH_TIMINGS}/{model_type}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gpu_count", "train_time", "avg_val_loss"])
        writer.writerows(results)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a model with specific parameters.")
    parser.add_argument("--model", type=str, required=True, help="Type of model")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    args = parser.parse_args()

    main(vars(args))