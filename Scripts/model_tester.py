import argparse
import numpy as np
import xarray as xr
import torch

from tqdm import tqdm
from models import initialize_model
from model_trainer import load_checkpoint
from data_loader import get_image_size

PATH_TEST = "../Data/Processed/Test.nc"
PATH_WEIGHTS = "../Models/Weights"
PATH_RESULTS = "../Models/Results"

TARGET_DAYS = 15
INPUT_VARS = ["zos", "u10", "v10"]
TARGET_VARS = ["uo", "vo", "zos"]

def test(model_type, path_test, downsampling_scale):
    # Load full dataset
    ds = xr.open_dataset(path_test)
    if downsampling_scale >= 1:
        ds = ds.interp(latitude=ds.latitude[::downsampling_scale], longitude=ds.longitude[::downsampling_scale], method="nearest")

    # Convert input vars to tensor and stack along channel dim
    inputs = torch.stack([torch.tensor(ds[var].values).float() for var in INPUT_VARS], dim=1)  # [T, C=3, H, W]

    # Convert target vars to tensor and stack along channel dim
    targets = torch.stack([torch.tensor(ds[var].values).float() for var in TARGET_VARS], dim=1)  # [T, C=3, H, W]

    # Make sure time length is divisible by TARGET_DAYS
    total_days = inputs.shape[0]
    usable_days = total_days - TARGET_DAYS
    batch_size = usable_days  # One sample for each valid starting day

    # Prepare input_t0: [B, C, H, W] for day t
    input_t0 = inputs[:usable_days]  # shape: [B, C, H, W]

    # Prepare target sequence: next 15 days per sample
    target_sequence = torch.stack([targets[t+1:t+1+TARGET_DAYS] for t in range(usable_days)])  # [B, T, C, H, W]
    target_sequence = target_sequence.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]

    # Initialize model and other components
    image_size = get_image_size(path_test, downsampling_scale)
    model, _, loss_function, _, _ = initialize_model(image_size, model_type)
    load_checkpoint(f"{PATH_WEIGHTS}/{model.name}.ckpt", model, experiment=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_t0 = input_t0.to(device)
    target_sequence = target_sequence.to(device)

    all_predictions = []
    daily_losses = []

    with torch.no_grad():
        for i in tqdm(range(input_t0.shape[0]), desc="Testing"):
            input_t = input_t0[i:i+1]  # [1, C, H, W]
            target_seq = target_sequence[i:i+1]  # [1, C, T, H, W]
            sample_predictions = []
            losses = []

            for t in range(TARGET_DAYS):
                output = model(input_t)  # [1, C, H, W]
                sample_predictions.append(output.unsqueeze(2))  # [1, C, 1, H, W]
                loss = loss_function(output, target_seq[:, :, t])
                losses.append(loss.item())

                input_t = output.detach()  # autoregressive input update

            predictions = torch.cat(sample_predictions, dim=2)  # [1, C, T, H, W]
            all_predictions.append(predictions)
            daily_losses.append(losses)

    all_predictions = torch.cat(all_predictions, dim=0)  # [B, C, T, H, W]
    all_targets = target_sequence.cpu()
    daily_losses = torch.tensor(daily_losses).mean(dim=0).numpy()

    print(f"Average loss by lead time: {daily_losses}")
    return daily_losses, all_predictions.cpu(), all_targets

def save_results(model_type, loss, predictions, targets):
    results_path = f"{PATH_RESULTS}/{model_type}.nc"

    preds_np = predictions.numpy().astype(np.float32)
    targets_np = targets.numpy().astype(np.float32)
    B, C, T, H, W = preds_np.shape

    ds = xr.Dataset(
        {
            "loss": (("lead_time",), loss.astype(np.float32)),
            "predictions": (("sample", "channel", "time", "latitude", "longitude"), preds_np),
            "targets": (("sample", "channel", "time", "latitude", "longitude"), targets_np),
        },
        coords={
            "sample": np.arange(B),
            "channel": np.arange(C),
            "time": np.arange(T),
            "latitude": np.linspace(65, 90, H),
            "longitude": np.linspace(-180, 180, W),
        },
    )

    ds.to_netcdf(results_path)

def main():
    parser = argparse.ArgumentParser(description="Test autoregressive model.")
    parser.add_argument("--model", type=str, required=True, help="Model type")
    parser.add_argument("--downsampling", type=int, default=2, help="Downsampling scale")
    args = parser.parse_args()

    loss, preds, targets = test(args.model, PATH_TEST, args.downsampling)
    save_results(args.model, loss, preds, targets)

if __name__ == "__main__":
    main()