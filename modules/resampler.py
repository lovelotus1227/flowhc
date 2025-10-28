import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()

        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim),
        )

    def forward(self, x):
        return self.ff(x)


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.proj_q = nn.Linear(dim, dim, bias=False)
        self.proj_k = nn.Linear(dim, dim, bias=False)
        self.proj_v = nn.Linear(dim, dim, bias=False)
        self.proj_out = nn.Linear(dim, dim, bias=False)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(self, z, x_f):
        z = self.norm_q(z)
        x_f = self.norm_kv(x_f)

        q = rearrange(self.proj_q(z), "b k (h p) -> b h k p", h=self.num_heads)
        k = rearrange(self.proj_k(x_f), "b n (h p) -> b h n p", h=self.num_heads)
        v = rearrange(self.proj_v(x_f), "b n (h p) -> b h n p", h=self.num_heads)

        dot = torch.matmul(q, k.transpose(-1, -2)) * self.scale # (B, H, K, N)
        attn = F.softmax(dot, dim=-1) # (B, H, K, N)

        update = torch.matmul(attn, v) # (B, H, K, P)
        update = rearrange(update, "b h k p -> b k (h p)")

        return self.proj_out(update)


class Resampler(nn.Module):
    def __init__(self, dim, num_heads, num_latents, num_layers, eta):
        super().__init__()

        self.latents = nn.Parameter(torch.randn(1, num_latents, dim))
        self.temporal_pos = nn.Parameter(torch.randn(1, 4, 1, dim))
        self.spatial_pos = nn.Parameter(torch.randn(1, 1, 49, dim))

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList([
                    CrossAttention(dim, num_heads),
                    FeedForward(dim),
                ])
            )

        self.eta = eta
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(dim)

    def compute_grad_expd(self, z):
        B, K, D = z.shape
        z = F.normalize(z, dim=-1).transpose(-1, -2).contiguous() # (B, D, K)
        I = torch.eye(D).repeat(B, 1, 1).to(z.device) # (B, D, D)
        alpha = D / (K * 0.1)
        E = alpha * torch.inverse(I + alpha * torch.bmm(z, z.transpose(-1, -2)).contiguous()) # (B, D, D)
        grad = torch.bmm(E, z).transpose(-1, -2) # (B, K, D)
        return grad
    
    def compute_grad_comp(self, z):
        z = rearrange(z, "b k (h p) -> b h k p", h=self.num_heads) # (B, H, K, P)
        B, H, K, P = z.shape
        I = torch.eye(P).repeat(B, 1, 1).to(z.device) # (B, P, P)
        alpha = P / (K * 0.1)
        grad_comp = []
        for i in range(self.num_heads):
            z_i = z[:,i,:,:] # (B, K, P)
            z_i = F.normalize(z_i, dim=-1).transpose(-1, -2).contiguous() # (B, P, K)
            E = alpha * torch.inverse(I + alpha * torch.bmm(z_i, z_i.transpose(-1, -2)).contiguous()) # (B, P, P)
            grad = torch.bmm(E, z_i).transpose(-1, -2).contiguous() # (B, K, P)
            grad_comp.append(grad)
        grad_comp = torch.stack(grad_comp, dim=1) # (B, H, K, P)
        grad_comp = rearrange(grad_comp, "b h k p -> b k (h p)", h=self.num_heads).contiguous()
        return grad_comp

    def compute_grad_rr(self, z):
        grad_expd = self.compute_grad_expd(z).mean(dim=(0,1), keepdim=True) # (1, 1, D)
        grad_comp = self.compute_grad_comp(z).mean(dim=(0,1), keepdim=True) # (1, 1, D)
        return grad_expd - grad_comp

    def forward(self, x):
        x = rearrange(x, "b d t h w -> b t (h w) d") # (B, T, N, D)
        x = x + self.temporal_pos
        x = x + self.spatial_pos
        x = rearrange(x, "b t n d -> b (t n) d") # (B, T*N, D)

        B, L, D = x.shape
        z = self.latents.repeat(B, 1, 1) # (B, K, D)

        for attn, ff in self.layers:
            z = attn(z, x) + z
            z = ff(z) + z
            z = z + self.eta * self.compute_grad_rr(z)

        return z