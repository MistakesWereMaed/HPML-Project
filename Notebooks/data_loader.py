import xarray as xr
import torch

from torch.utils.data import Dataset, DataLoader

class XarrayDataset(Dataset):
    def __init__(self, ds, input_var, target_var):
        self.ds = ds
        self.input_var = input_var
        self.target_var = target_var

    def __len__(self):
        return self.ds.dims["time"]

    def __getitem__(self, idx):
        x = self.ds[self.input_var].isel(time=idx).values
        y = self.ds[self.target_var].isel(time=idx).values

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        return x_tensor, y_tensor
    
def load_dataset(path, batch_size=32, shuffle=False, num_workers=4):
    x = ['zos', 'u10', 'v10', 'sin_day', 'cos_day']
    y = ['uo', 'vo']

    ds = xr.open_dataset(path, chunks="auto")
    xds = XarrayDataset(ds, x, y)
    return DataLoader(xds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)