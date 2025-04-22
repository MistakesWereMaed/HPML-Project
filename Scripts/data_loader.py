import xarray as xr
import torch

from tqdm import tqdm
from axonn import axonn as ax
from torch.utils.data import Dataset

class XarrayTensorDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
def get_image_size(path, downsampling_scale=2):
    ds = xr.open_dataset(path, chunks="auto")
    if downsampling_scale >= 1:
        ds = ds.interp(latitude=ds.latitude[::downsampling_scale], longitude=ds.longitude[::downsampling_scale], method="nearest")
    # Image size
    lat_size = ds.sizes.get("latitude", 0)
    lon_size = ds.sizes.get("longitude", 0)
    ds.close()
    return (lat_size, lon_size)

def load_data(rank=0, world_size=1, path=None, batch_size=8, downsampling_scale=2, shuffle=False):
    input_vars = ['zos', 'u10', 'v10']
    target_vars = ['uo', 'vo', 'zos']
    
    # Load and preprocess dataset
    ds = xr.open_dataset(path)
    ds = ds.chunk({"time": ds.sizes["time"] // world_size})

    if downsampling_scale >= 1:
        ds = ds.interp(
            latitude=ds.latitude[::downsampling_scale],
            longitude=ds.longitude[::downsampling_scale],
            method="nearest"
        )

    total_time = ds.sizes["time"]
    split_size = total_time // world_size

    start_idx = rank * split_size
    end_idx = (rank + 1) * split_size if rank < world_size - 1 else total_time
    
    chunk = ds.isel(time=slice(start_idx, end_idx))
    chunk.load()

    # Preload tensors to GPU
    x_list, y_list = [], []
    time_steps = chunk.sizes["time"] - 1
    for i in tqdm(range(time_steps), desc=f"[Rank {rank}] Loading", leave=False):
        x = chunk[input_vars].isel(time=i).to_array().values
        y = chunk[target_vars].isel(time=i + 1).to_array().values
        
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        x_list.append(x_tensor)
        y_list.append(y_tensor)

    x_all = torch.stack(x_list).to(rank)
    y_all = torch.stack(y_list).to(rank)

    # Dataset with preloaded tensors
    tensor_dataset = XarrayTensorDataset(x_all, y_all)
    dataloader = ax.create_dataloader(
        tensor_dataset,
        global_batch_size=batch_size * world_size,
        num_workers=0
    )

    return dataloader
