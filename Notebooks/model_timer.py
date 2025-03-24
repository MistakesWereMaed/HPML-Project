import csv
import argparse
import torch

from model_trainer import train
from models import PICPModel
from data_loader import load_dataset

PATH_TRAIN = "../Data/Processed/Test.nc"
PATH_VAL = "../Data/Processed/Val.nc"
PATH_TIMINGS = "../Models/Timings"

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
    
    print("\nTraining with 1 GPU...")
    total_time = 0
    total_val_loss = 0
    
    # Run multiple trials
    for trial in range(trials):
        # Initialize and train model
        model_kwargs = model_class.initialize_model(image_size, params)
        val_loss, train_time = train(train_ds=train_ds, val_ds=val_ds, batch_size=params["batch_size"], epochs=epochs, 
                                     experiment=False, show_progress_bar=True, **model_kwargs)

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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a model with specific parameters.")
    parser.add_argument("--model", type=str, required=True, help="Type of model")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    args = parser.parse_args()

    main(vars(args))