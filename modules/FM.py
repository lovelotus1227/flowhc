import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


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
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
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


class FeatureRefinementAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_gate = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        x_norm = self.norm_attn(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + self.attn_gate(x_norm) * attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x


class FlowMatchingModule(nn.Module):
    def __init__(self, dim, num_heads, num_latents, num_layers, eta=None, is_image=False, num_frames=8):
        super().__init__()
        self.num_latents = num_latents
        self.dim = dim
        self.is_image = is_image
        self.num_frames = num_frames
        self.eta = eta

        if not is_image and num_frames > 1:
            self.temporal_emb = nn.Parameter(torch.randn(1, num_frames, dim) * 0.02)
        else:
            self.temporal_emb = None

        self.latents_v = nn.Parameter(torch.randn(1, num_latents, dim) * 0.02)
        self.latents_o = nn.Parameter(torch.randn(1, num_latents, dim) * 0.02)

        self.shared_layers = nn.ModuleList([
            nn.ModuleList([CrossAttention(dim, num_heads), FeedForward(dim)])
            for _ in range(num_layers // 2)
        ])

        self.cross_interact_layers = nn.ModuleList([
            nn.ModuleList([
                CrossAttention(dim, num_heads),
                CrossAttention(dim, num_heads),
                FeedForward(dim),
                FeedForward(dim),
            ])
            for _ in range((num_layers + 1) // 2)
        ])

        self.flow_predictor_v = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.flow_predictor_o = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.time_embed = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.feature_refiner = (
            nn.Identity() if is_image else FeatureRefinementAttention(dim, num_heads)
        )

        self.norm_v = nn.LayerNorm(dim)
        self.norm_o = nn.LayerNorm(dim)

    def add_temporal_pos_emb(self, x, num_frames=None):
        if self.temporal_emb is None:
            return x

        _, num_tokens, _ = x.shape
        frames = num_frames or self.num_frames
        if frames <= 1 or num_tokens % frames != 0:
            return x

        temporal_emb = self.temporal_emb
        if temporal_emb.shape[1] != frames:
            temporal_emb = F.interpolate(
                temporal_emb.transpose(1, 2),
                size=frames,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        x = rearrange(x, "b (t hw) d -> b t hw d", t=frames)
        x = x + temporal_emb.to(device=x.device, dtype=x.dtype).unsqueeze(2)
        return rearrange(x, "b t hw d -> b (t hw) d")

    def add_time_condition(self, z_t, t):
        time_emb = self.time_embed(t.to(dtype=z_t.dtype))
        return z_t + time_emb.to(dtype=z_t.dtype)

    def flow_match(self, z_target_v, z_target_o):
        batch_size = z_target_v.shape[0]
        z_noise_v = torch.randn_like(z_target_v)
        z_noise_o = torch.randn_like(z_target_o)
        t = torch.rand(
            batch_size,
            1,
            1,
            device=z_target_v.device,
            dtype=z_target_v.dtype,
        )

        z_t_v = (1 - t) * z_noise_v + t * z_target_v
        z_t_o = (1 - t) * z_noise_o + t * z_target_o

        v_pred_v = self.flow_predictor_v(self.add_time_condition(z_t_v, t))
        v_pred_o = self.flow_predictor_o(self.add_time_condition(z_t_o, t))

        v_target_v = z_target_v - z_noise_v
        v_target_o = z_target_o - z_noise_o

        loss_flow_v = F.mse_loss(v_pred_v, v_target_v)
        loss_flow_o = F.mse_loss(v_pred_o, v_target_o)
        return 0.5 * (loss_flow_v + loss_flow_o)

    def contrastive_reg_loss(self, feat, labels, tau=0.1):
        feat = F.normalize(feat, dim=-1)
        sim = torch.matmul(feat, feat.transpose(-1, -2)) / tau
        return F.cross_entropy(sim, labels)

    @staticmethod
    def _different_label_orthogonal_loss(feat, labels, margin=0.2):
        if labels is None:
            return feat.new_tensor(0.0)

        labels = labels.reshape(-1).to(device=feat.device)
        if feat.shape[0] != labels.numel():
            raise ValueError(
                f"label count ({labels.numel()}) must match feature batch size ({feat.shape[0]})"
            )

        feat = F.normalize(feat, dim=-1)
        sim = torch.matmul(feat, feat.t())
        different_label = labels.unsqueeze(0) != labels.unsqueeze(1)
        off_diagonal = ~torch.eye(
            labels.numel(),
            dtype=torch.bool,
            device=labels.device,
        )
        mask = different_label & off_diagonal
        if not mask.any():
            return feat.new_tensor(0.0)
        return F.relu(sim[mask] - margin).pow(2).mean()

    def orthogonal_flow_loss(
            self,
            z_v,
            z_o,
            verb_labels=None,
            obj_labels=None,
            branch_margin=0.1,
            class_margin=0.2,
    ):
        if z_v.dim() == 3:
            z_v = z_v.mean(dim=1)
        if z_o.dim() == 3:
            z_o = z_o.mean(dim=1)

        z_v = F.normalize(z_v, dim=-1)
        z_o = F.normalize(z_o, dim=-1)
        cross_sim = (z_v * z_o).sum(dim=-1).abs()
        cross_branch = F.relu(cross_sim - branch_margin).pow(2).mean()
        verb_orth = self._different_label_orthogonal_loss(z_v, verb_labels, class_margin)
        obj_orth = self._different_label_orthogonal_loss(z_o, obj_labels, class_margin)
        return cross_branch + 0.5 * (verb_orth + obj_orth)

    def _flatten_encoder_output(self, x):
        effective_num_frames = self.num_frames
        if x.dim() == 5:
            effective_num_frames = x.shape[2]
            x = rearrange(x, "b d t h w -> b (t h w) d")
        elif x.dim() == 2:
            tokens, dim = x.shape
            if tokens % self.num_frames != 0:
                raise ValueError(
                    f"2D input first dimension ({tokens}) must be divisible by num_frames ({self.num_frames})"
                )
            x = x.reshape(tokens // self.num_frames, self.num_frames, dim)
        elif x.dim() != 3:
            raise ValueError(f"Unsupported input dimensions: {x.shape}")
        return x, effective_num_frames

    def forward(self, x):
        x, effective_num_frames = self._flatten_encoder_output(x)
        x = self.add_temporal_pos_emb(x, effective_num_frames)
        x = self.feature_refiner(x)

        batch_size = x.shape[0]
        z_v = self.latents_v.expand(batch_size, -1, -1)
        z_o = self.latents_o.expand(batch_size, -1, -1)

        for attn, ff in self.shared_layers:
            z_v = z_v + attn(z_v, x)
            z_v = z_v + ff(z_v)
            z_o = z_o + attn(z_o, x)
            z_o = z_o + ff(z_o)

        for attn_vo, attn_ov, ff_v, ff_o in self.cross_interact_layers:
            z_v_prev, z_o_prev = z_v, z_o
            z_v = z_v_prev + attn_vo(z_v_prev, z_o_prev)
            z_o = z_o_prev + attn_ov(z_o_prev, z_v_prev)
            z_v = z_v + ff_v(z_v)
            z_o = z_o + ff_o(z_o)

        z_v = self.norm_v(z_v)
        z_o = self.norm_o(z_o)

        if self.training:
            loss_flow = self.flow_match(z_v, z_o)
            return z_v, z_o, loss_flow
        return z_v, z_o


class Composer(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int = None,
            dropout: float = 0.1,
            composer_type: str = "gated",
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or dim
        self.composer_type = composer_type

        self.coff_net = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 2),
        )

        self.gate_v = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.gate_o = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())

        self.fuse_proj = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
        )

        nn.init.constant_(self.coff_net[-1].weight, 0.0)
        nn.init.constant_(self.coff_net[-1].bias, 0.0)

    def forward(self, v_feat, o_feat, normalize=True):
        if v_feat.dim() == 3:
            v_feat = v_feat.mean(dim=1)
            o_feat = o_feat.mean(dim=1)

        if normalize:
            v_feat = F.normalize(v_feat, dim=-1)
            o_feat = F.normalize(o_feat, dim=-1)

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
            v_feat = F.normalize(v_feat, dim=-1)
            o_feat = F.normalize(o_feat, dim=-1)

        gv = self.gate_v(v_feat)
        go = self.gate_o(o_feat)

        a = a.unsqueeze(-1)
        b = b.unsqueeze(-1)
        linear = a * gv * v_feat + b * go * o_feat
        fused = self.fuse_proj(torch.cat([v_feat, o_feat], dim=-1))

        return linear + fused
