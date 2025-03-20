import torch
import torch.nn as nn
from linformer import LinformerSelfAttention

class DataDrivenModule(nn.Module):
    def __init__(self, in_channels=3, embed_dim=768, kernel_size=(5, 10), num_heads=4, linformer_k=256, dropout_p=0.1, mlp_hidden_dim=768, transconv_out_channels=3):
        super(DataDrivenModule, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=embed_dim, 
            kernel_size=kernel_size
        )
        self.attn = LinformerSelfAttention(
            dim=embed_dim,        
            seq_len=316197,
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
        self.transconv = nn.ConvTranspose2d(in_channels=mlp_hidden_dim, out_channels=transconv_out_channels, kernel_size=kernel_size)

    def forward(self, x):
        with torch.autocast("cuda"):
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
        with torch.autocast("cuda"):
            # Compute gradients ∂SSH/∂x and ∂SSH/∂y using finite differences
            dudx = torch.diff(ssh, dim=-1, append=ssh[:, :, :, -1:])
            dvdy = torch.diff(ssh, dim=-2, append=ssh[:, :, -1:, :])
            
            # Compute geostrophic velocity components
            u_g = self.g / self.f * dvdy
            v_g = -self.g / self.f * dudx
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
        with torch.autocast("cuda"):
            # Step 1: Data-driven module
            output = self.data_module(x)
            u_prime, v_prime, ssh = torch.chunk(output, chunks=3, dim=1)
            # Step 2: Physics-informed module
            u_g, v_g = self.physics_module(ssh)
            # Step 3: Sum module
            u, v = self.sum_module(u_g, v_g, u_prime, v_prime)

            return torch.stack((u, v), dim=1)