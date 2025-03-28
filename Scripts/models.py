import json
import torch
import torch.nn as nn
import torch.optim as optim

from linformer import LinformerSelfAttention
from hyperopt import hp
from data_loader import get_dataset

PATH_PARAMS = "../Models/Params"

NUM_FEATURES = 3

def load_and_initialize(model_type="PINN", path1=None, path2=None, downsampling_scale=2, splits=12, hyperparameters=None):
    # Select model
    match model_type:
        case "PINN":
            model_class = PICPModel
        case _:
            raise ValueError(f"Unknown model type")
    # Load params
    params = model_class.load_params() if hyperparameters is None else hyperparameters
    return_dict = {
        "datasets": [],
        "model_kwargs": None
    }
    # Load data
    train_set, image_size = get_dataset(path=path1, downsampling_scale=downsampling_scale, splits=splits)
    model_kwargs = model_class.initialize_model(image_size, params)

    return_dict["datasets"].append(train_set)
    return_dict["model_kwargs"] = model_kwargs
    # Load validation data
    if path2:
        val_set, _ = get_dataset(path=path2, downsampling_scale=downsampling_scale, splits=1)
        return_dict["datasets"].append(val_set)

    return return_dict

###### PINN ######
def calculate_seq_length(image_size, kernel_size, padding=0, stride=1):
    height_in, width_in = image_size
    kernel_height, kernel_width = kernel_size
    # Compute output height and width after Conv2D
    height_out = (height_in + 2 * padding - kernel_height) // stride + 1
    width_out = (width_in + 2 * padding - kernel_width) // stride + 1
    # Compute sequence length
    return height_out * width_out

class PICPModel(nn.Module):
    def __init__(self, image_size, in_channels=3, kernel_size=(5, 10), num_heads=4, embed_dim=768, 
                 linformer_k=256, dropout_p=0.1, mlp_hidden_dim=768, out_channels=3, g=9.81, f=1e-4, **kwargs):
        super(PICPModel, self).__init__()
        embed_dim = int(embed_dim)
        linformer_k = int(linformer_k)
        mlp_hidden_dim = int(mlp_hidden_dim)
        # Data-Driven Components
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=kernel_size)
        self.attn = LinformerSelfAttention(dim=embed_dim, seq_len=calculate_seq_length(image_size, kernel_size), k=linformer_k, heads=num_heads)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout_p)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.LeakyReLU(negative_slope=0.01)
        )
        self.transconv = nn.ConvTranspose2d(mlp_hidden_dim, out_channels, kernel_size=kernel_size)
        # Physics-Informed Components
        self.g = g
        self.f = f
        # Ensure model is on CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        # Step 1: Data-Driven Computation
        x = self.conv(x)
        batch_size, channels, height, width = x.shape
        seq_len = height * width
        x = x.view(batch_size, channels, seq_len).transpose(1, 2)
        x = self.attn(x, context=x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.mlp(x)
        x = x.permute(0, 2, 1).view(batch_size, -1, height, width)
        x = self.transconv(x)
        # Extract u', v', SSH from Data-Driven output
        u_prime, v_prime, ssh = torch.chunk(x, chunks=3, dim=1)
        # Step 2: Physics-Informed Computation (∇SSH → geostrophic velocity)
        dudx = torch.diff(ssh, dim=-1, append=ssh[:, :, :, -1:])
        dvdy = torch.diff(ssh, dim=-2, append=ssh[:, :, -1:, :])
        u_g = self.g / self.f * dvdy
        v_g = -self.g / self.f * dudx
        # Step 3: Sum Module (Combining Data-Driven and Physics-Informed velocities)
        with torch.autocast("cuda"):
            u = u_g + u_prime
            v = v_g + v_prime

        return torch.stack((u, v, ssh), dim=1)

    @staticmethod
    def get_hyperparam_space():
        return {
            "input_days": hp.choice("input_days", [1, 3, 7]),
            "target_days": hp.choice("target_days", [1, 7, 15]),
            "batch_size": hp.choice("batch_size", [1]),
            "kernel_size": hp.choice("kernel_size", [(3, 3), (5, 10), (7, 7)]),
            "linformer_k": hp.quniform("linformer_k", 128, 528, 128),
            "num_heads": hp.choice("num_heads", [1, 2, 4]),
            "embed_dim": hp.quniform("embed_dim", 128, 528, 128),
            "mlp_hidden_dim": hp.quniform("mlp_hidden_dim", 128, 528, 128),
            "learning_rate": hp.loguniform("learning_rate", -6, -2),
        }
    
    @staticmethod
    def load_params():
        try:
            with open(f"{PATH_PARAMS}/PINN-Base.json", "r") as f:
                print("Loading saved params")
                params = json.load(f)
        except FileNotFoundError:
            params = {
                "input_days": 7,
                "target_days": 15,
                "batch_size": 2,
                "kernel_size": (5, 10),
                "linformer_k": 256,
                "num_heads": 1,
                "embed_dim": 256,
                "mlp_hidden_dim": 256,
                "learning_rate": 7e-4,
            }
        return params
    
    @staticmethod
    def initialize_model(image_size, params):
        # Calculate channel dimensions
        in_channels = NUM_FEATURES * params["input_days"]
        out_channels = NUM_FEATURES * params["target_days"]
        # Initialize model
        model = PICPModel(image_size=image_size, in_channels=in_channels, out_channels=out_channels, **params)
        loss_function = nn.SmoothL1Loss(beta=1.0)
        optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"]) # Add scheduler
        # Package returns
        model_kwargs = {
            "name": "PINN",
            "model": model,
            "loss_function": loss_function,
            "optimizer": optimizer,
            "hyperparameters": params,
        }
        return model_kwargs
    


###### GNN ######



###### FNO ######