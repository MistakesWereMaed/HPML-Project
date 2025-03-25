import csv
import argparse
import torch

from model_trainer import train

PATH_TRAIN = "../Data/Processed/Test.nc"
PATH_VAL = "../Data/Processed/Val.nc"
PATH_TIMINGS = "../Models/Timings"

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
    
    print("\nTraining with 1 GPU...")
    total_time = 0
    total_val_loss = 0
    
    # Run multiple trials
    for trial in range(trials):
        # Initialize and train model
        val_loss, train_time = train(
            model_type=model_type, epochs=epochs,
            path_train=PATH_TRAIN, path_val=PATH_VAL, downsampling_scale=downsampling_scale, 
            experiment=True, show_progress_bar=True
        )

        total_time += train_time
        total_val_loss += val_loss

        del model_kwargs
        torch.cuda.empty_cache()
    
    # Compute averages
    avg_time = total_time / trials
    avg_val_loss = total_val_loss / trials
    
    print(f"Average time per epoch: {avg_time:.2f} seconds")
    print(f"Average validation loss: {avg_val_loss:.4f}")
    
    # Save results to CSV
    with open(f"{PATH_TIMINGS}/{model_type}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gpu_count", "train_time", "avg_val_loss"])
        writer.writerow([1, avg_time, avg_val_loss])

if __name__ == "__main__":
    main()