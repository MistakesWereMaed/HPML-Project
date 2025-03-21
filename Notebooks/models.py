import json
import torch
import torch.nn as nn
import torch.optim as optim

from linformer import LinformerSelfAttention
from hyperopt import hp
from data_loader import load_dataset

PATH_TRAIN = "../Data/Processed/Train.nc"
PATH_VAL = "../Data/Processed/Val.nc"
PATH_TEST = "../Data/Processed/Test.nc"

PATH_PARAMS = "../Models/Params"

NUM_FEATURES = 3

###### PINN ######
def calculate_seq_length(image_size, kernel_size, padding=0, stride=1):
    height_in, width_in = image_size
    kernel_height, kernel_width = kernel_size
    # Compute output height and width after Conv2D
    height_out = (height_in + 2 * padding - kernel_height) // stride + 1
    width_out = (width_in + 2 * padding - kernel_width) // stride + 1
    # Compute sequence length
    return height_out * width_out

class DataDrivenModule(nn.Module):
    def __init__(self, image_size, in_channels=3, kernel_size=(5, 10), num_heads=4, embed_dim=768, linformer_k=256, dropout_p=0.1, mlp_hidden_dim=768, out_channels=3, **kwargs):
        super(DataDrivenModule, self).__init__()

        embed_dim = int(embed_dim)
        linformer_k = int(linformer_k)
        mlp_hidden_dim = int(mlp_hidden_dim)

        self.conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=embed_dim, 
            kernel_size=kernel_size
        )
        self.attn = LinformerSelfAttention(
            dim=embed_dim,        
            seq_len=calculate_seq_length(image_size, kernel_size),
            k=linformer_k,
            heads=num_heads
        )
        self.dropout = nn.Dropout(p=dropout_p)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.LeakyReLU(negative_slope=0.01)
        )
        self.transconv = nn.ConvTranspose2d(in_channels=mlp_hidden_dim, out_channels=out_channels, kernel_size=kernel_size)

    def forward(self, x):
        x = self.conv(x)
        batch_size, channels, height, width = x.shape
        seq_len = height * width
        x = x.view(batch_size, channels, seq_len).transpose(1, 2)
        self.attn.seq_len = seq_len
        x = self.attn(x, context=x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.mlp(x)
        x = x.permute(0, 2, 1).view(batch_size, -1, height, width)
        x = self.transconv(x)
        
        return x  # Returns u', v', SSH
    
class PhysicsInformedModule(nn.Module):
    def __init__(self, g=9.81, f=1e-4):
        super(PhysicsInformedModule, self).__init__()
        self.g = g
        self.f = f

    def forward(self, ssh):
        # Compute gradients ∂SSH/∂x and ∂SSH/∂y using finite differences
        dudx = torch.diff(ssh, dim=-1, append=ssh[:, :, :, -1:])
        dvdy = torch.diff(ssh, dim=-2, append=ssh[:, :, -1:, :])
        
        # Compute geostrophic velocity components
        u_g = self.g / self.f * dvdy
        v_g = -self.g / self.f * dudx

        if torch.isnan(u_g).any() or torch.isinf(u_g).any():
            print("NaN detected in u_g (geostrophic velocity)")

        if torch.isnan(v_g).any() or torch.isinf(v_g).any():
            print("NaN detected in v_g (geostrophic velocity)")
            exit(1)
            
        return u_g, v_g
    
class SumModule(nn.Module):
    def forward(self, u_g, v_g, u_prime, v_prime):
        with torch.autocast("cuda"):
            u = u_g + u_prime
            v = v_g + v_prime
            return u, v
        
class PICPModel(nn.Module):
    def __init__(self, **kwargs):
        super(PICPModel, self).__init__()
        self.data_module = DataDrivenModule(**kwargs)
        self.physics_module = PhysicsInformedModule()
        self.sum_module = SumModule()

    def forward(self, x):
        # Step 1: Data-driven module
        output = self.data_module(x)
        u_prime, v_prime, ssh = torch.chunk(output, chunks=3, dim=1)
        # Step 2: Physics-informed module
        u_g, v_g = self.physics_module(ssh)
        # Step 3: Sum module
        u, v = self.sum_module(u_g, v_g, u_prime, v_prime)

        return torch.stack((u, v), dim=1)

    @staticmethod
    def get_hyperparam_space():
        return {
            "input_days": hp.choice("input_days", [1, 3, 7]),
            "target_days": hp.choice("target_days", [1, 7, 15]),
            "batch_size": hp.choice("batch_size", [1, 2, 4]),
            "kernel_size": hp.choice("kernel_size", [(3, 3), (5, 10), (7, 7)]),
            "linformer_k": hp.quniform("linformer_k", 128, 512, 128),
            "num_heads": hp.choice("num_heads", [1, 2, 4]),
            "embed_dim": hp.quniform("embed_dim", 128, 512, 128),
            "mlp_hidden_dim": hp.quniform("mlp_hidden_dim", 128, 512, 128),
            "learning_rate": hp.loguniform("learning_rate", -7, -4),
        }
    
    @staticmethod
    def load_params():
        try:
            with open(f"{PATH_PARAMS}/PINN.json", "r") as f:
                params = json.load(f)
        except FileNotFoundError:
            params = {
                "input_days": 1,
                "target_days": 1,
                "batch_size": 2,
                "kernel_size": (5, 10),
                "linformer_k": 256,
                "num_heads": 1,
                "embed_dim": 128,
                "mlp_hidden_dim": 128,
                "learning_rate": 7e-4,
            }
        return params
    
    @staticmethod
    def initialize_model(downsampling_scale=2, path_train=PATH_TRAIN, path_val=PATH_VAL, path_test=PATH_TEST, testing=False, **kwargs):
        params = PICPModel.load_params()
        params.update(kwargs)
        # Load dataset
        if testing:
            data = (load_dataset(path_test, downsampling_scale=downsampling_scale, **params))
        else:
            train_loader = load_dataset(path_train, downsampling_scale=downsampling_scale, **params)
            val_loader = load_dataset(path_val, downsampling_scale=downsampling_scale, **params)
            data = (train_loader, val_loader)
        # Get image size
        features, _ = next(iter(data[0]))
        image_size = features.shape[2:]

        # Calculate channel dimensions
        in_channels = NUM_FEATURES * params["input_days"]
        out_channels = NUM_FEATURES * params["target_days"]
        # Initialize model
        model = PICPModel(image_size=image_size, in_channels=in_channels, out_channels=out_channels, **params)
        loss_function = nn.SmoothL1Loss(beta=1.0)
        optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
        # Package returns
        model_kwargs = {
            "model": model,
            "loss_function": loss_function,
            "optimizer": optimizer,
            "hyperparameters": params,
            "data": data,
        }
        return model_kwargs
    


###### GNN ######



###### FNO ######