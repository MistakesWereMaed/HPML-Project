import json
import torch
import torch.nn as nn
import torch.fft
import torch.optim as optim
import torch.nn.functional as F

from linformer import LinformerSelfAttention
from hyperopt import hp

PATH_PARAMS = "../Models/Params"

NUM_FEATURES = 3
INPUT_DAYS = 7
TARGET_DAYS = 15

def initialize_model(image_size, model_type="PINN", hyperparameters=None):
    # Select model
    match model_type:
        case "PINN":
            model_class = PICPModel
        case "FNO":
            model_class = FNO2d
        case _:
            raise ValueError(f"Unknown model type")
    # Load params
    params = model_class.load_params() if hyperparameters is None else hyperparameters
    # Initialize model
    model = model_class(image_size=image_size, **params)
    loss_function = nn.SmoothL1Loss(beta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

    return model, optimizer, loss_function, params

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
    def __init__(self, image_size, kernel_size=(5, 10), num_heads=4, embed_dim=768, 
                 linformer_k=256, dropout_p=0.1, mlp_hidden_dim=768, g=9.81, f=1e-4, **kwargs):
        super(PICPModel, self).__init__()
        embed_dim = int(embed_dim)
        linformer_k = int(linformer_k)
        mlp_hidden_dim = int(mlp_hidden_dim)
        in_channels = NUM_FEATURES * INPUT_DAYS
        out_channels = NUM_FEATURES * TARGET_DAYS
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
        # Extract u', v', SSH from Data-Driven output
        u_prime, v_prime, ssh = torch.chunk(x, chunks=3, dim=1)
        # Step 2: Physics-Informed Computation (∇SSH → geostrophic velocity)
        dudx = torch.diff(ssh, dim=-1, append=ssh[:, :, :, -1:])
        dvdy = torch.diff(ssh, dim=-2, append=ssh[:, :, -1:, :])
        u_g = self.g / self.f * dvdy
        v_g = -self.g / self.f * dudx
        # Step 3: Sum Module (Combining Data-Driven and Physics-Informed velocities)
        u = u_g + u_prime
        v = v_g + v_prime

        return torch.stack((u, v, ssh), dim=1)

    @staticmethod
    def get_hyperparam_space():
        return {
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
            with open(f"{PATH_PARAMS}/PINN.json", "r") as f:
                params = json.load(f)
        except FileNotFoundError:
            params = {
                "kernel_size": (5, 10),
                "linformer_k": 256,
                "num_heads": 1,
                "embed_dim": 256,
                "mlp_hidden_dim": 256,
                "learning_rate": 7e-4,
            }
        return params
    


###### GNN ######



###### FNO ######
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        x = torch.fft.irfft2(out_ft, s=(H, W))
        return x

class FNO2d(nn.Module):
    def __init__(self, image_size, **kwargs):
        super().__init__()
        in_channels = NUM_FEATURES * INPUT_DAYS
        out_channels = NUM_FEATURES * TARGET_DAYS

        self.modes1 = int(kwargs["num_fourier_modes"])
        self.modes2 = int(kwargs["num_fourier_modes"])
        self.width = int(kwargs["fno_width"])
        self.depth = int(kwargs["num_fno_layers"])
        self.mlp_hidden_dim = int(kwargs["mlp_hidden_dim"])

        self.fc0 = nn.Conv2d(in_channels, self.width, kernel_size=1)

        self.convs = nn.ModuleList([
            SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
            for _ in range(self.depth)
        ])
        self.ws = nn.ModuleList([
            nn.Conv2d(self.width, self.width, 1)
            for _ in range(self.depth)
        ])

        self.fc1 = nn.Linear(self.width, self.mlp_hidden_dim)
        self.fc2 = nn.Linear(self.mlp_hidden_dim, out_channels)

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

        x = x.permute(0, 2, 3, 1)  # [B, H, W, width]
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)  # [B, out_channels, H, W]

        if pad_h > 0: x = x[..., :-pad_h, :]
        if pad_w > 0: x = x[..., :, :-pad_w]

        u, v, ssh = torch.chunk(x, chunks=3, dim=1)
        return torch.stack((u, v, ssh), dim=1)

    @staticmethod
    def get_hyperparam_space():
        return {
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
                "num_fourier_modes": 20,
                "num_fno_layers": 2,
                "fno_width": 64,
                "mlp_hidden_dim": 128,
                "learning_rate": 1e-3,
            }
        return params