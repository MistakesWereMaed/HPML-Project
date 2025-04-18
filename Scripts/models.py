import json
import torch
import numpy as np
import torch.nn as nn
import torch.fft
import torch.optim as optim
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.utils import grid
from torch_geometric.data import Data, Batch

from linformer import LinformerSelfAttention
from hyperopt import hp

PATH_PARAMS = "../Models/Params"

def initialize_model(image_size, model_type="PINN", hyperparameters=None):
    # Select model
    match model_type:
        case "PINN":
            model_class = PICPModel
        case "GNN":
            model_class = GNN
        case "FNO":
            model_class = FNO
        case _:
            raise ValueError(f"Unknown model type")
    # Load params
    params = model_class.load_params() if hyperparameters is None else hyperparameters
    # Initialize model
    model = model_class(image_size=image_size, **params)
    loss_function = nn.SmoothL1Loss(beta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

    return model, optimizer, loss_function, params["batch_size"]

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
    def __init__(self, image_size, kernel_size, num_heads, embed_dim, 
                 linformer_k, mlp_hidden_dim, dropout_p=0.1, g=9.81, f=1e-4, **kwargs):
        super(PICPModel, self).__init__()
        embed_dim = int(embed_dim)
        linformer_k = int(linformer_k)
        mlp_hidden_dim = int(mlp_hidden_dim)
        # Data-Driven Components
        self.conv = nn.Conv2d(3, embed_dim, kernel_size=kernel_size)
        self.attn = LinformerSelfAttention(dim=embed_dim, seq_len=calculate_seq_length(image_size, kernel_size), k=linformer_k, heads=num_heads)
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout_p)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU()
        )
        self.transconv = nn.ConvTranspose2d(mlp_hidden_dim, 3, kernel_size=kernel_size)
        # Physics-Informed Components
        self.g = g
        self.f = f
        self.name = "PINN"

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
        x = torch.clamp(x, min=1e-5, max=1e5)
        # Extract u', v', SSH from Data-Driven output
        u_prime, v_prime, ssh = torch.chunk(x, chunks=3, dim=1)
        u_prime = u_prime.squeeze(1)
        v_prime = v_prime.squeeze(1)
        ssh = ssh.squeeze(1)
        # Step 2: Physics-Informed Computation (∇SSH → geostrophic velocity)
        dudx = torch.diff(ssh, dim=-1, append=ssh[:, :, -1:])
        dvdy = torch.diff(ssh, dim=-2, append=ssh[:, -1:, :])
        u_g = self.g / self.f * dvdy
        v_g = -self.g / self.f * dudx
        # Step 3: Sum Module (Combining Data-Driven and Physics-Informed velocities)
        u = u_g + u_prime
        v = v_g + v_prime

        return torch.stack((u, v, ssh), dim=1)

    @staticmethod
    def get_hyperparam_space():
        return {
            "batch_size": hp.choice("batch_size", [1, 4, 8]),
            "kernel_size": hp.choice("kernel_size", [(3, 3), (5, 10), (7, 7)]),
            "linformer_k": hp.quniform("linformer_k", 128, 528, 128),
            "num_heads": hp.choice("num_heads", [1, 2, 4]),
            "embed_dim": hp.quniform("embed_dim", 128, 528, 128),
            "mlp_hidden_dim": hp.quniform("mlp_hidden_dim", 128, 528, 128),
            "learning_rate": hp.loguniform("learning_rate", np.log(1e-6), np.log(1e-3)),
        }
    
    @staticmethod
    def load_params():
        try:
            with open(f"{PATH_PARAMS}/PINN.json", "r") as f:
                params = json.load(f)
        except FileNotFoundError:
            params = {
                "batch_size": 8,
                "kernel_size": (3, 3),
                "linformer_k": 128,
                "num_heads": 1,
                "embed_dim": 128,
                "mlp_hidden_dim": 128,
                "learning_rate": 1e-5,
            }
        return params
    


###### GNN ######
class GNN(nn.Module):
    def __init__(self, image_size, **kwargs):
        super().__init__()
        hidden_dim = int(kwargs["hidden_dim"])
        num_gnn_layers = int(kwargs["num_gnn_layers"])

        height, width = image_size
        self.H, self.W = height, width
        self.n_nodes = height * width
        self.dropout = 0.1

        # Define edge_index
        edge_index = grid(height, width)[0]

        self.input_proj = nn.Linear(3, hidden_dim)
        self.gnn_layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_gnn_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, 3)
        self.name = "GNN"
        self.edge_index = edge_index

    def forward(self, x):
        device = x.device
        edge_index = self.edge_index.to(device)

        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).permute(0, 2, 1)  # [B, N, C]

        # Create a list of PyG Data objects, one per sample
        data_list = [
            Data(x=xb, edge_index=edge_index) for xb in x
        ]

        # Batch the graphs into one large graph
        batch = Batch.from_data_list(data_list).to(device)  # batch.x: [B*N, C]

        # Apply GNN layers on the full batch
        h = self.input_proj(batch.x)
        for conv in self.gnn_layers:
            h = conv(h, batch.edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        out = self.output_proj(h)  # [B*N, 3]

        # Unbatch and reshape
        out = out.view(B, self.n_nodes, -1)  # [B, N, 3]
        out = out.permute(0, 2, 1).reshape(B, -1, H, W)  # [B, 3, H, W]

        return out
    
    @staticmethod
    def get_hyperparam_space():
        return {
            "batch_size": hp.choice("batch_size", [1, 4, 8, 16]),
            "hidden_dim": hp.quniform("hidden_dim", 32, 256, 32),
            "num_gnn_layers": hp.choice("num_gnn_layers", [2, 3, 4]),
            "learning_rate": hp.loguniform("learning_rate", -6, -3),
        }
    
    @staticmethod
    def load_params():
        try:
            with open(f"{PATH_PARAMS}/GNN.json", "r") as f:
                params = json.load(f)
        except FileNotFoundError:
            params = {
                "batch_size": 8,
                "hidden_dim": 64,
                "num_gnn_layers": 1,
                "learning_rate": 1e-4,
            }
        return params


###### FNO ######
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat),
            requires_grad=False
        )

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        B, C, H, W = x.shape

        with torch.amp.autocast(device_type="cuda", enabled=False):
            x_fp32 = x.float()
            x_ft = torch.fft.rfft2(x_fp32)

            out_ft = torch.zeros(
                B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device
            )

            out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
                x_ft[:, :, :self.modes1, :self.modes2], self.weights1
            )

            x = torch.fft.irfft2(out_ft, s=(H, W))
            x = x.to(dtype=x_fp32.dtype)

        return x

class FNO(nn.Module):
    def __init__(self, image_size, **kwargs):
        super().__init__()

        self.modes1 = int(kwargs["num_fourier_modes"])
        self.modes2 = int(kwargs["num_fourier_modes"])
        self.width = int(kwargs["fno_width"])
        self.depth = int(kwargs["num_fno_layers"])
        self.mlp_hidden_dim = int(kwargs["mlp_hidden_dim"])
        self.dropout = nn.Dropout(0.1)

        self.fc0 = nn.Conv2d(3, self.width, kernel_size=1)

        self.convs = nn.ModuleList([
            SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
            for _ in range(self.depth)
        ])
        self.ws = nn.ModuleList([
            nn.Conv2d(self.width, self.width, 1)
            for _ in range(self.depth)
        ])

        self.fc1 = nn.Linear(self.width, self.mlp_hidden_dim)
        self.fc2 = nn.Linear(self.mlp_hidden_dim, 3)

        self.name = "FNO"

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        pad_h = (self.modes1 - H % self.modes1) % self.modes1
        pad_w = (self.modes2 - W % self.modes2) % self.modes2

        x = F.pad(x, [0, pad_w, 0, pad_h])  # Pad width (last), then height (first)
        x = self.fc0(x)

        for i, (conv, w) in enumerate(zip(self.convs, self.ws)):
            x1 = conv(x)
            x2 = w(x)
            x = torch.relu(x1 + x2)
            x = self.dropout(x)

        x = x.permute(0, 2, 3, 1)  # [B, H, W, width]
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)  # [B, out_channels, H, W]

        if pad_h > 0: x = x[..., :-pad_h, :]
        if pad_w > 0: x = x[..., :, :-pad_w]

        return x

    @staticmethod
    def get_hyperparam_space():
        return {
            "batch_size": hp.choice("batch_size", [1, 4, 8, 16]),
            "num_fourier_modes": hp.quniform("num_fourier_modes", 8, 32, 4),
            "num_fno_layers": hp.choice("num_fno_layers", [1, 2, 4]),
            "fno_width": hp.quniform("fno_width", 32, 128, 32),
            "mlp_hidden_dim": hp.quniform("mlp_hidden_dim", 64, 256, 64),
            "learning_rate": hp.loguniform("learning_rate", -6, -3),
        }

    @staticmethod
    def load_params():
        try:
            with open(f"{PATH_PARAMS}/FNO.json", "r") as f:
                params = json.load(f)
        except FileNotFoundError:
            params = {
                "batch_size": 4,
                "num_fourier_modes": 20,
                "num_fno_layers": 2,
                "fno_width": 64,
                "mlp_hidden_dim": 128,
                "learning_rate": 1e-3,
            }
        return params