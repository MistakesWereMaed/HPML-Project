import csv
import argparse
import torch
import os

from model_trainer import train

PATH_TRAIN = "../Data/Processed/Train.nc"
PATH_VAL = "../Data/Processed/Val.nc"
PATH_TIMINGS = "../Models/Timings"

import argparse
import os
import csv
import torch

def main():
    parser = argparse.ArgumentParser(description="Train a model with specific parameters.")
    parser.add_argument("--model", type=str, required=True, help="Type of model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--splits", type=int, default=12, help="Number of splits")
    parser.add_argument("--trials", type=int, default=2, help="Number of trials")
    parser.add_argument("--downsampling", type=int, default=2, help="Downsampling reduction scale")

    args = parser.parse_args()
    model_type = args.model
    epochs = args.epochs
    splits = args.splits
    trials = args.trials
    downsampling_scale = args.downsampling

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs.")

    os.makedirs(PATH_TIMINGS, exist_ok=True)
    csv_path = f"{PATH_TIMINGS}/{model_type}.csv"

    # Write CSV header
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gpu_count", "avg_train_time", "avg_val_loss"])

    # Loop through each GPU count
    for gpu_count in range(1, num_gpus + 1):
        print(f"\nTraining with {gpu_count} GPU(s)...")
        total_time = 0
        total_val_loss = 0

        for trial in range(trials):
            val_loss, train_time = train(
                model_type=model_type, epochs=epochs, 
                path_train=PATH_TRAIN, path_val=PATH_VAL, downsampling_scale=downsampling_scale, splits=splits, 
                experiment=True, world_size=gpu_count, show_progress_bar=True
            )

            total_time += train_time
            total_val_loss += val_loss
            torch.cuda.empty_cache()

        avg_time = total_time / trials / epochs
        avg_val_loss = total_val_loss / trials

        print(f"[{gpu_count} GPU(s)] Average time per epoch: {avg_time:.2f} seconds")
        print(f"[{gpu_count} GPU(s)] Average validation loss: {avg_val_loss:.4f}")

        # Append results to CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([gpu_count, avg_time, avg_val_loss])

if __name__ == "__main__":
    main()
