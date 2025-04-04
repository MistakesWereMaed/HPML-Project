import xarray as xr
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

BATCH_SIZE = 1
INPUT_DAYS = 7
TARGET_DAYS = 15

class XarrayDataset(Dataset):
    def __init__(self, ds, input_vars, target_vars):
        if ds["time"].dtype != np.int64:
            ds["time"] = ds["time"].dt.dayofyear
        self.ds = ds
        self.input_vars = input_vars
        self.target_vars = target_vars
        self.input_days = INPUT_DAYS
        self.target_days = TARGET_DAYS

    def __len__(self):
        return self.ds.sizes["time"] - (self.input_days + self.target_days - 1)

    def __getitem__(self, idx):
        # Select past `input_days` timesteps for input variables
        x_list = []
        for var in self.input_vars:
            x_list.append(self.ds[var].isel(time=slice(idx, idx + self.input_days)).values)
        x = np.stack(x_list, axis=0)  
        # Select next `target_days` timesteps for target variables
        y_list = []
        for var in self.target_vars:
            y_list.append(self.ds[var].isel(time=slice(idx + self.input_days, idx + self.input_days + self.target_days)).values)
        y = np.stack(y_list, axis=0)
        # Reshape: Merge time into the channel dimension (for Conv2D)
        x = x.transpose(1, 0, 2, 3)
        x = x.reshape(-1, x.shape[2], x.shape[3])
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

def load_data(ds):
    input_vars = ['zos', 'u10', 'v10']
    target_vars = ['uo', 'vo', 'zos']

    ds.load()
    xr_ds = XarrayDataset(ds, input_vars, target_vars)
    dataloader = DataLoader(xr_ds, batch_size=BATCH_SIZE, pin_memory=True)

    return dataloader
