import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim),
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

        dot = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dot, dim=-1)
        update = torch.matmul(attn, v)
        update = rearrange(update, "b h k p -> b k (h p)")
        return self.proj_out(update)


# -----------------------------
# 🔥 升级：双分支交互流匹配（创新版）
# -----------------------------
class FlowMatchingModule(nn.Module):
    def __init__(self, dim, num_heads, num_latents, num_layers, eta=None, is_image=False, num_frames=8):
        super().__init__()
        self.num_latents = num_latents
        self.dim = dim
        self.is_image = is_image
        self.num_frames = num_frames

        # 时序位置编码（视频专用 → 创新1）
        if not is_image and num_frames > 1:
            self.temporal_emb = nn.Parameter(torch.randn(1, num_frames, dim) * 0.02)
        else:
            self.temporal_emb = None

        # 属性/物体 双分支可学习 latent
        self.latents_v = nn.Parameter(torch.randn(1, num_latents, dim))
        self.latents_o = nn.Parameter(torch.randn(1, num_latents, dim))

        # 共享底层特征提取
        self.shared_layers = nn.ModuleList([])
        for _ in range(num_layers // 2):
            self.shared_layers.append(nn.ModuleList([
                CrossAttention(dim, num_heads),
                FeedForward(dim),
            ]))

        # 【创新2】高层双向交互层：v ↔ o 互相引导
        self.cross_interact_layers = nn.ModuleList([])
        for _ in range((num_layers + 1) // 2):
            self.cross_interact_layers.append(nn.ModuleList([
                CrossAttention(dim, num_heads),  # v 看 o
                CrossAttention(dim, num_heads),  # o 看 v
                FeedForward(dim),
                FeedForward(dim),
            ]))

        # 双分支流预测器
        self.flow_predictor_v = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.flow_predictor_o = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

        self.norm_v = nn.LayerNorm(dim)
        self.norm_o = nn.LayerNorm(dim)

    def add_temporal_pos_emb(self, x):
        """ 视频时序编码 → 让流匹配感知时间变化 """
        if self.temporal_emb is None:
            return x
        B, N, D = x.shape
        T = self.num_frames
        HW = N // T
        x = rearrange(x, "b (t hw) d -> b t hw d", t=T)
        x = x + self.temporal_emb.unsqueeze(2)
        return rearrange(x, "b t hw d -> b (t hw) d")

    # --------------------------
    # 标准流匹配（保留原版）
    # --------------------------
    def flow_match(self, z_target_v, z_target_o):
        B, K, D = z_target_v.shape
        z_noise_v = torch.randn_like(z_target_v)
        z_noise_o = torch.randn_like(z_target_o)
        t = torch.rand(B, 1, 1, device=z_target_v.device)

        z_t_v = (1 - t) * z_noise_v + t * z_target_v
        z_t_o = (1 - t) * z_noise_o + t * z_target_o

        v_pred_v = self.flow_predictor_v(z_t_v)
        v_pred_o = self.flow_predictor_o(z_t_o)

        v_target_v = z_target_v - z_noise_v
        v_target_o = z_target_o - z_noise_o

        loss_flow_v = F.mse_loss(v_pred_v, v_target_v)
        loss_flow_o = F.mse_loss(v_pred_o, v_target_o)
        return (loss_flow_v + loss_flow_o) / 2

    # --------------------------
    # 🔥 创新3：双向引导流损失（比 leakage 更强）
    # --------------------------
    def bilateral_guide_loss(self, z_v, z_o):
        # 属性 → 物体 引导
        pred_o = self.flow_predictor_o(z_v)
        # 物体 → 属性 引导
        pred_v = self.flow_predictor_v(z_o)

        loss_o = F.mse_loss(pred_o, z_o)
        loss_v = F.mse_loss(pred_v, z_v)
        return (loss_v + loss_o) * 0.5

    # --------------------------
    # 🔥 创新4：对比正则化损失（增强特征判别性）
    # --------------------------
    def contrastive_reg_loss(self, feat, labels, tau=0.1):
        sim = torch.matmul(feat, feat.transpose(-1, -2)) / tau
        return F.cross_entropy(sim, labels)

    # --------------------------
    # 保留你原版 leakage（兼容）
    # --------------------------
    def leakage_flow_match(self, z_v, z_o):
        B, K, D = z_v.shape
        z_noise_v = torch.randn_like(z_v)
        z_noise_o = torch.randn_like(z_o)
        t = torch.rand(B, 1, 1, device=z_v.device)

        z_t_v = (1 - t) * z_noise_v + t * z_v
        v_pred_v_to_o = self.flow_predictor_o(z_t_v)
        v_target_v_to_o = z_o - z_noise_v
        loss_mse_v_to_o = F.mse_loss(v_pred_v_to_o, v_target_v_to_o)

        z_t_o = (1 - t) * z_noise_o + t * z_o
        v_pred_o_to_v = self.flow_predictor_v(z_t_o)
        v_target_o_to_v = z_v - z_noise_o
        loss_mse_o_to_v = F.mse_loss(v_pred_o_to_v, v_target_o_to_v)

        return 0.5 * (loss_mse_v_to_o + loss_mse_o_to_v)

    def forward(self, x):
        if x.dim() == 5:
            x = rearrange(x, "b d t h w -> b (t h w) d")
        elif x.dim() == 2:
            BT, D = x.shape
            B = BT // self.num_frames
            x = x.reshape(B, self.num_frames, D)
        elif x.dim() != 3:
            raise ValueError(f"不支持的输入维度: {x.shape}")

        # 加入时序编码
        x = self.add_temporal_pos_emb(x)
        B = x.shape[0]

        z_v = self.latents_v.repeat(B, 1, 1)
        z_o = self.latents_o.repeat(B, 1, 1)

        # 共享层
        for attn, ff in self.shared_layers:
            z_v = attn(z_v, x) + z_v
            z_v = ff(z_v) + z_v
            z_o = attn(z_o, x) + z_o
            z_o = ff(z_o) + z_o

        # 🔥 创新5：双向交叉交互层（v ↔ o 信息互通）
        for attn_vo, attn_ov, ff_v, ff_o in self.cross_interact_layers:
            z_v = attn_vo(z_v, z_o) + z_v  # 属性关注物体
            z_o = attn_ov(z_o, z_v) + z_o  # 物体关注属性
            z_v = ff_v(z_v) + z_v
            z_o = ff_o(z_o) + z_o

        z_v = self.norm_v(z_v)
        z_o = self.norm_o(z_o)

        if self.training:
            loss_flow = self.flow_match(z_v, z_o)
            return z_v, z_o, loss_flow
        else:
            return z_v, z_o


# ================================================
# 🔥 升级：通道门控组合器 GatedComposer（强创新）
# ================================================
class Composer(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int = None,
            dropout: float = 0.1,
            composer_type: str = "gated"  # gated / simple / mlp
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or dim
        self.composer_type = composer_type

        # 全局系数预测
        self.coff_net = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 2)
        )

        # 🔥 核心创新：逐通道门控（细粒度融合）
        self.gate_v = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.gate_o = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

        # 深度融合投影
        self.fuse_proj = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU()
        )

        # 初始化系数接近 0.5
        nn.init.constant_(self.coff_net[-1].weight, 0.0)
        nn.init.constant_(self.coff_net[-1].bias, 0.0)

    def forward(self, v_feat, o_feat, normalize=True):
        if v_feat.dim() == 3:
            v_feat = v_feat.mean(dim=1)
            o_feat = o_feat.mean(dim=1)

        if normalize:
            v_feat = F.normalize(v_feat, dim=-1)
            o_feat = F.normalize(o_feat, dim=-1)

        # 全局系数 a + b = 1
        combined = torch.cat([v_feat, o_feat], dim=-1)
        logits = self.coff_net(combined)
        a, b = torch.softmax(logits, dim=-1).unbind(-1)
        return a, b

    def compose(self, v_feat, o_feat, a=None, b=None, normalize=True):
        if a is None or b is None:
            a, b = self.forward(v_feat, o_feat, normalize)

        if v_feat.dim() == 3:
            v_feat = v_feat.mean(dim=1)
            o_feat = o_feat.mean(dim=1)
        if normalize:
            v_feat = F.normalize(v_feat, -1)
            o_feat = F.normalize(o_feat, -1)

        # 通道门控
        gv = self.gate_v(v_feat)
        go = self.gate_o(o_feat)

        # 加权组合
        a = a.unsqueeze(-1)
        b = b.unsqueeze(-1)
        linear = a * gv * v_feat + b * go * o_feat

        # 深度融合
        fused = self.fuse_proj(torch.cat([v_feat, o_feat], -1))

        # 最终组合特征
        return linear + fused