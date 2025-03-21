import argparse
import numpy as np
import torch

from tqdm import tqdm
from models import PICPModel
from model_trainer import load_checkpoint

PATH_WEIGHTS = "../Models/Weights"
PATH_RESULTS = "../Models/Results"

# Testing Function
def test(model, test_loader, loss_function, device="cuda"):
    model.to(device)
    model.eval()
    # Initialize return variables
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    # Testing loop
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            # Forward pass
            predictions = model(inputs)
            loss = loss_function(predictions, targets)
            # Progress update
            total_loss += loss.item()
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
    # Testing summary
    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}")

    return avg_loss, torch.cat(all_predictions), torch.cat(all_targets)

def main(args):
    # Select model
    model_type = args["model"]
    match model_type:
        case "PINN":
            model_class = PICPModel
        #case "GNN": 
        #case "FNO":
        case _:
            raise ValueError(f"Unknown model type")
    # Initialize model
    model_kwargs = model_class.initialize_model(testing=True)
    name = model_kwargs["name"]
    model = model_kwargs["model"]
    optimizer = model_kwargs["optimizer"]
    loss_function = model_kwargs["loss_function"]
    test_ds = model_kwargs["data"][0]
    # Load weights and test
    load_checkpoint(f"{PATH_WEIGHTS}/{name}-current.ckpt", model, optimizer)
    loss, predictions, targets = test(model, test_ds, loss_function)
    # Save results
    results_path = f"{PATH_RESULTS}/{model_type}.npz"
    np.savez_compressed(
        results_path,
        loss=np.array(loss, dtype=np.float32),
        predictions=predictions.cpu().numpy(),
        targets=targets.cpu().numpy()
    )

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a model with specific parameters.")
    parser.add_argument("--model", type=str, required=True, help="Type of model")
    args = parser.parse_args()
    
    main(vars(args))