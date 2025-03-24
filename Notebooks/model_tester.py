import argparse
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from models import PICPModel
from model_trainer import load_checkpoint
from data_loader import load_dataset

PATH_TEST = "../Data/Processed/Test.nc"
PATH_WEIGHTS = "../Models/Weights"
PATH_RESULTS = "../Models/Results"

def test(model, name, loss_function, optimizer, test_ds, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    load_checkpoint(f"{PATH_WEIGHTS}/{name}-Base.ckpt", model, optimizer)
    
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    all_loss = []
    all_predictions = []
    all_targets = []
    
    progress_bar = tqdm(test_loader, desc="Testing", leave=True)
    with torch.no_grad():
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            loss = loss_function(predictions, targets)

            all_loss.append(loss.item())
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())

            progress_bar.set_postfix(loss=loss.item())
    
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    return all_loss, all_predictions, all_targets

def main(args):
    model_type = args["model"]
    match model_type:
        case "PINN":
            model_class = PICPModel
        case _:
            raise ValueError(f"Unknown model type")
    
    params = model_class.load_params()
    test_ds, image_size = load_dataset(path=PATH_TEST, input_days=params["input_days"], target_days=params["target_days"])
    
    model_kwargs = model_class.initialize_model(image_size, params)
    name = model_kwargs["name"]
    model = model_kwargs["model"]
    optimizer = model_kwargs["optimizer"]
    loss_function = model_kwargs["loss_function"]
    
    loss, predictions, targets = test(model, name, loss_function, optimizer, test_ds, params["batch_size"])
    
    results_path = f"{PATH_RESULTS}/{model_type}-Base.npz"
    np.savez_compressed(
        results_path,
        loss=loss,
        predictions=predictions.numpy(),
        targets=targets.numpy()
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a model with specific parameters.")
    parser.add_argument("--model", type=str, required=True, help="Type of model")
    args = parser.parse_args()
    main(vars(args))