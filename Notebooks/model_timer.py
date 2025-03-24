import os
import csv
import argparse
import torch
import torch.multiprocessing as mp
from multiprocessing import Manager

from model_trainer import train

PATH_TRAIN = "../Data/Processed/Test.nc"
PATH_VAL = "../Data/Processed/Val.nc"
PATH_TIMINGS = "../Models/Timings"


def train_wrapper(rank, world_size, model_type, epochs, path_train, path_val, downsampling_scale, experiment, show_progress_bar, return_dict):
    """ Trains the model and stores timing results in a shared dictionary. """
    val_loss, train_time = train(rank, world_size, model_type, epochs, path_train, path_val, downsampling_scale, experiment, show_progress_bar)

    # Only store results from rank 0 to avoid duplicate entries
    if rank == 0:
        return_dict["train_time"] = train_time
        return_dict["val_loss"] = val_loss


def main():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    parser = argparse.ArgumentParser(description="Train a model with specific parameters.")
    parser.add_argument("--model", type=str, required=True, help="Type of model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--trials", type=int, default=2, help="Number of trials")

    args = parser.parse_args()
    model_type = args.model
    epochs = args.epochs
    trials = args.trials

    world_size = torch.cuda.device_count()

    # Ensure output directory exists
    os.makedirs(PATH_TIMINGS, exist_ok=True)

    results = []

    # Loop through different GPU configurations (1 to world_size GPUs)
    for num_gpus in range(1, world_size + 1):
        total_time = 0.0
        total_val_loss = 0.0

        for trial in range(trials):
            print(f"Starting training on {num_gpus} GPUs (Trial {trial+1}/{trials})...")

            # Use a shared dictionary to store results
            with Manager() as manager:
                return_dict = manager.dict()

                # Spawn processes for training
                mp.spawn(
                    train_wrapper,
                    args=(num_gpus, model_type, epochs, PATH_TRAIN, PATH_VAL, 2, True, True, return_dict),
                    nprocs=num_gpus,
                    join=True
                )

                # Retrieve stored results from shared dict (only rank 0 has valid results)
                train_time = return_dict.get("train_time", float("inf"))
                val_loss = return_dict.get("val_loss", float("inf"))

            total_time += train_time
            total_val_loss += val_loss

            # Free up GPU memory
            torch.cuda.empty_cache()

        # Compute averages
        avg_time = total_time / trials
        avg_val_loss = total_val_loss / trials

        print(f"GPU Count: {num_gpus}")
        print(f"  -> Average training time per epoch: {avg_time:.2f} seconds")
        print(f"  -> Average validation loss: {avg_val_loss:.4f}")

        results.append([num_gpus, avg_time, avg_val_loss])

    # Save results to CSV
    csv_path = f"{PATH_TIMINGS}/{model_type}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gpu_count", "train_time", "avg_val_loss"])
        writer.writerows(results)

    print(f"Timing results saved to {csv_path}")


if __name__ == "__main__":
    main()
