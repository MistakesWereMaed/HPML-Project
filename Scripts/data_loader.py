import xarray as xr
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

class XarrayDataset(Dataset):
    def __init__(self, ds, input_vars, target_vars):
        if ds["time"].dtype != np.int64:
            ds["time"] = ds["time"].dt.dayofyear
        self.ds = ds
        self.input_vars = input_vars
        self.target_vars = target_vars

    def __len__(self):
        return self.ds.sizes["time"] - 1

    def __getitem__(self, idx):
        # Input: stack input variables along the channel dimension
        x = self.ds[self.input_vars].isel(time=idx).to_array().values  # shape: [C_in, H, W]

        # Target: stack target variables along the channel dimension
        y = self.ds[self.target_vars].isel(time=idx + 1).to_array().values  # shape: [C_out, H, W]

        # Convert to tensors
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        return x_tensor, y_tensor
    
def get_image_size(path, downsampling_scale=2):
    ds = xr.open_dataset(path, chunks="auto")
    if downsampling_scale >= 1:
        ds = ds.interp(latitude=ds.latitude[::downsampling_scale], longitude=ds.longitude[::downsampling_scale], method="nearest")
    # Image size
    lat_size = ds.sizes.get("latitude", 0)
    lon_size = ds.sizes.get("longitude", 0)
    ds.close()
    return (lat_size, lon_size)

def get_dataset(path=None, downsampling_scale=2, splits=1):
    # Load dataset
    ds = xr.open_dataset(path)
    ds = ds.chunk({"time": ds.sizes["time"] // splits})
    # Apply downsampling
    if downsampling_scale >= 1:
        ds = ds.interp(latitude=ds.latitude[::downsampling_scale], longitude=ds.longitude[::downsampling_scale], method="nearest")
    # Skip splitting for single split
    if splits == 1:
        return ds
    # Split dataset along the "time" dimension
    total_time = ds.sizes["time"]
    split_size = total_time // splits
    # Split data into chunks
    chunks = []
    for i in range(splits):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size
        # Select the subset of the dataset
        chunk = ds.isel(time=slice(start_idx, end_idx))
        chunks.append(chunk)

    return chunks

def load_data(ds, batch_size):
    input_vars = ['zos', 'u10', 'v10']
    target_vars = ['uo', 'vo', 'zos']

    ds.load()
    xr_ds = XarrayDataset(ds, input_vars, target_vars)
    dataloader = DataLoader(xr_ds, batch_size=batch_size, pin_memory=True)

    return dataloader
