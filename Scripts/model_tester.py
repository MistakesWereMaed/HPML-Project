import argparse
import numpy as np
import torch

from tqdm import tqdm
from models import load_and_initialize
from model_trainer import load_checkpoint

PATH_TEST = "../Data/Processed/Test.nc"
PATH_WEIGHTS = "../Models/Weights"
PATH_RESULTS = "../Models/Results"

def test(model_type, path_test, downsampling_scale):
    
    model_dict = load_and_initialize(model_type=model_type, path1=path_test, downsampling_scale=downsampling_scale)
    model_kwargs = model_dict["model_kwargs"]

    name = model_kwargs["name"]
    model = model_kwargs["model"]
    optimizer = model_kwargs["optimizer"]
    loss_function = model_kwargs["loss_function"]
    target_days = model_kwargs["hyperparameters"]["target_days"]
    
    test_loader = model_dict["loaders"][0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    load_checkpoint(f"{PATH_WEIGHTS}/{name}-Best.ckpt", model, optimizer)
    
    # Initialize lists to store daily losses
    daily_losses = np.zeros(target_days)
    all_predictions = []
    all_targets = []
    
    i = 0
    progress_bar = tqdm(test_loader[0], desc="Testing", leave=True)
    with torch.no_grad():
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            i += 1
            # Loop through each day in the prediction horizon
            for day in range(target_days):
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


def main():
    parser = argparse.ArgumentParser(description="Train a model with specific parameters.")
    parser.add_argument("--model", type=str, required=True, help="Type of model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--downsampling", type=int, default=2, help="Downsampling reduction scale")

    args = parser.parse_args()
    model_type = args.model
    downsampling_scale = args.downsampling
 
    loss, predictions, targets = test(model_type, PATH_TEST, downsampling_scale)
    
    results_path = f"{PATH_RESULTS}/{model_type}.npz"
    np.savez_compressed(
        results_path,
        loss=loss,
        predictions=predictions.numpy(),
        targets=targets.numpy()
    )

if __name__ == "__main__":
    main()