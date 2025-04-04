import argparse
import numpy as np
import xarray as xr
import torch

from tqdm import tqdm
from models import initialize_model
from model_trainer import load_checkpoint
from data_loader import load_data, get_image_size, get_dataset

PATH_TEST = "../Data/Processed/Test.nc"
PATH_WEIGHTS = "../Models/Weights"
PATH_RESULTS = "../Models/Results"

INPUT_DAYS = 7
TARGET_DAYS = 15

def test(model_type, path_test, downsampling_scale):
    
    image_size = get_image_size(path_test, downsampling_scale)
    model, optimizer, loss_function, _ = initialize_model(image_size, model_type)
    load_checkpoint(f"{PATH_WEIGHTS}/{model.name}.ckpt", model, optimizer, experiment=False)
    
    test_set = get_dataset(path=path_test, downsampling_scale=downsampling_scale, splits=1)
    test_loader = load_data(test_set)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Initialize lists to store daily losses
    daily_losses = np.zeros(TARGET_DAYS)
    all_predictions = []
    all_targets = []
    
    i = 0
    progress_bar = tqdm(test_loader, desc="Testing", leave=True)
    with torch.no_grad():
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            i += 1
            # Loop through each day in the prediction horizon
            for day in range(TARGET_DAYS):
                # Calculate the loss for each individual day
                day_loss = loss_function(predictions[:, :, day], targets[:, :, day])
                daily_losses[day] += day_loss.item()
            
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            
            progress_bar.set_postfix(loss=np.mean(daily_losses) / i)

    # Calculate the average loss per day across all batches
    daily_losses.sort()
    daily_losses /= len(test_loader)
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    print(f"Average loss by day of lead time: {daily_losses}")
    return daily_losses, all_predictions, all_targets


def save_results(model_type, loss, predictions, targets):
    results_path = f"{PATH_RESULTS}/{model_type}.nc"
    
    all_predictions_np = predictions.detach().cpu().numpy().astype(np.float32)
    all_targets_np = targets.detach().cpu().numpy().astype(np.float32)
    # Define dimensions based on shape
    num_samples, num_channels, num_time_steps, lat_size, lon_size = all_predictions_np.shape
    # Create an xarray dataset with explicit dimensions
    ds = xr.Dataset(
    {
        "loss": (("lead_time",), loss.astype(np.float32)),
        "predictions": (("sample", "channel", "time", "latitude", "longitude"), all_predictions_np),
        "targets": (("sample", "channel", "time", "latitude", "longitude"), all_targets_np),
    },
    coords={
        "sample": np.arange(num_samples),
        "channel": np.arange(num_channels),
        "time": np.arange(num_time_steps),
        "latitude": np.linspace(-90, 90, lat_size),
        "longitude": np.linspace(-180, 180, lon_size),
    },
)
    
    ds.to_netcdf(results_path)

def main():
    parser = argparse.ArgumentParser(description="Train a model with specific parameters.")
    parser.add_argument("--model", type=str, required=True, help="Type of model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--downsampling", type=int, default=2, help="Downsampling reduction scale")

    args = parser.parse_args()
    model_type = args.model
    downsampling_scale = args.downsampling
 
    loss, predictions, targets = test(model_type, PATH_TEST, downsampling_scale)
    save_results(model_type, loss, predictions, targets)

if __name__ == "__main__":
    main()