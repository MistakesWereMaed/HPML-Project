import csv
import argparse
import torch
import os

PATH_METRICS = "../Models/Metrics"
PATH_TIMINGS = "../Models/Timings"

def read_metrics_from_csv(model_type, gpu_count):
    csv_path = os.path.join(PATH_METRICS, f"{model_type}_{gpu_count}.csv")
    val_losses = []
    train_times = []

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Expected metrics CSV not found: {csv_path}")

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "val_loss" in row and "time" in row:
                val_losses.append(float(row["val_loss"]))
                train_times.append(float(row["time"]))

    return val_losses, train_times

def main():
    parser = argparse.ArgumentParser(description="Train a model with specific parameters.")
    parser.add_argument("--model", type=str, required=True, help="Type of model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--trials", type=int, default=2, help="Number of trials")
    parser.add_argument("--downsampling", type=int, default=2, help="Downsampling reduction scale")

    args = parser.parse_args()
    model_type = args.model
    epochs = args.epochs
    trials = args.trials
    downsampling_scale = args.downsampling

    os.makedirs(PATH_TIMINGS, exist_ok=True)
    timing_csv = f"{PATH_TIMINGS}/{model_type}.csv"

    with open(timing_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gpu_count", "avg_train_time", "avg_val_loss"])

    num_gpus = torch.cuda.device_count()

    for gpu_count in [1, 2, 4]:
        if gpu_count > num_gpus: break
        print(f"\nTraining with {gpu_count} GPU(s)...")
        
        for trial in range(trials):
            os.system(f"bash launch.sh -p {gpu_count} -m {model_type} -e {epochs}")

        # Parse results from generated metrics file
        val_losses, train_times = read_metrics_from_csv(model_type, gpu_count)

        if len(val_losses) < epochs * trials or len(train_times) < epochs * trials:
            print(f"Warning: Expected {epochs * trials} entries, found {len(val_losses)}.")

        val_loss = val_losses[-1]
        avg_train_time = sum(train_times) / len(train_times)

        print(f"[{gpu_count} GPU(s)] Avg time per epoch: {avg_train_time:.2f} s")
        print(f"[{gpu_count} GPU(s)] Avg val loss: {val_loss:.4f}")

        with open(timing_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([gpu_count, avg_train_time, val_loss])

if __name__ == "__main__":
    main()
