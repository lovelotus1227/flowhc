import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt.manifolds.stereographic as poincare


class HyperProjector(nn.Module):
    def __init__(self, dim=768, curvature=1.0, max_tangent_norm=1.0):
        super().__init__()
        self.manifold = poincare.PoincareBall(c=curvature)
        self.dim = dim
        self.max_tangent_norm = max_tangent_norm

        self.proj_v = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.proj_o = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.hyper_comp_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def _clip_tangent(self, tangent):
        norm = tangent.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        scale = (self.max_tangent_norm / norm).clamp(max=1.0)
        return tangent * scale

    def to_hyperbolic(self, z_euclidean, proj_net):
        tangent = self._clip_tangent(proj_net(z_euclidean))
        point = self.manifold.expmap0(tangent)
        return self.manifold.projx(point)

    def hyper_composition_loss(self, z_v, z_o):
        h_v = self.to_hyperbolic(z_v, self.proj_v)
        h_o = self.to_hyperbolic(z_o, self.proj_o)

        h_comp_mobius = self.manifold.mobius_add(h_v, h_o)
        if hasattr(self.manifold, "mobius_scalar_mul"):
            h_comp_mobius = self.manifold.mobius_scalar_mul(0.5, h_comp_mobius)
        h_comp_mobius = self.manifold.projx(h_comp_mobius)
        h_comp_target = self.to_hyperbolic(0.5 * (z_v + z_o), self.hyper_comp_proj)

        dist = self.manifold.dist(h_comp_mobius, h_comp_target)
        return torch.nan_to_num(dist, nan=0.0, posinf=1e3, neginf=0.0).mean()

    def hyper_contrastive_loss(self, z_v, z_o, tau=0.1):
        h_v = self.to_hyperbolic(z_v, self.proj_v)
        h_o = self.to_hyperbolic(z_o, self.proj_o)

        z_v_norm = F.normalize(z_v, dim=-1)
        z_o_norm = F.normalize(z_o, dim=-1)
        sim_matrix = torch.matmul(z_v_norm, z_o_norm.t()) / tau

        batch_size = z_v.size(0)
        labels = torch.arange(batch_size, device=z_v.device)

        loss_v2o = F.cross_entropy(sim_matrix, labels)
        loss_o2v = F.cross_entropy(sim_matrix.t(), labels)

        hyper_dist = self.manifold.dist(h_v.unsqueeze(1), h_o.unsqueeze(0))
        hyper_dist = torch.nan_to_num(hyper_dist, nan=1e3, posinf=1e3, neginf=0.0)
        hyper_sim = -hyper_dist / tau

        loss_hyper_v2o = F.cross_entropy(hyper_sim, labels)
        loss_hyper_o2v = F.cross_entropy(hyper_sim.t(), labels)

        return 0.25 * (loss_v2o + loss_o2v + loss_hyper_v2o + loss_hyper_o2v)
