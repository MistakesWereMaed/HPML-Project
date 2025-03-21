import xarray as xr
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader, DistributedSampler

class XarrayDataset(Dataset):
    def __init__(self, ds, input_vars, target_vars, input_days=7, target_days=15):
        ds["time"] = ds["time"].dt.dayofyear
        self.ds = ds
        self.input_vars = input_vars
        self.target_vars = target_vars
        self.input_days = input_days
        self.target_days = target_days

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

def load_dataset(path, downsampling_scale=2, batch_size=2, num_workers=2, input_days=1, target_days=1, rank=0, world_size=1, **kwargs):
    input_vars = ['zos', 'u10', 'v10']
    target_vars = ['uo', 'vo']
    # Load dataset
    ds = xr.open_dataset(path, chunks="auto")
    if downsampling_scale >= 1:
        ds = ds.interp(latitude=ds.latitude[::downsampling_scale], longitude=ds.longitude[::downsampling_scale], method="nearest")
    # Load tensors
    dataset = XarrayDataset(ds, input_vars, target_vars, input_days, target_days)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)

    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
