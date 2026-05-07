import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt.manifolds.stereographic as poincare


class HyperProjector(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.manifold = poincare.PoincareBall(c=1.0)
        self.dim = dim

        self.proj_v = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.proj_o = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        self.hyper_comp_proj = nn.Linear(dim, dim)

    def to_hyperbolic(self, z_euclidean, proj_net):
        tangent = proj_net(z_euclidean)
        return self.manifold.expmap0(tangent)

    def hyper_composition_loss(self, z_v, z_o):
        h_v = self.to_hyperbolic(z_v, self.proj_v)
        h_o = self.to_hyperbolic(z_o, self.proj_o)

        h_comp_mobius = self.manifold.mobius_add(h_v, h_o)
        h_comp_target = self.to_hyperbolic((z_v + z_o) * 0.5, self.proj_v)

        dist = self.manifold.dist(h_comp_mobius, h_comp_target)
        return dist.mean()

    def hyper_contrastive_loss(self, z_v, z_o, tau=0.1):
        h_v = self.to_hyperbolic(z_v, self.proj_v)
        h_o = self.to_hyperbolic(z_o, self.proj_o)

        sim_matrix = torch.matmul(F.normalize(z_v, dim=-1), F.normalize(z_o, dim=-1).t()) / tau

        B = z_v.size(0)
        labels = torch.arange(B, device=z_v.device)

        loss_v2o = F.cross_entropy(sim_matrix, labels)
        loss_o2v = F.cross_entropy(sim_matrix.t(), labels)

        hyper_dist = self.manifold.dist(h_v.unsqueeze(1), h_o.unsqueeze(0))
        hyper_sim = -hyper_dist / tau

        loss_hyper_v2o = F.cross_entropy(hyper_sim, labels)
        loss_hyper_o2v = F.cross_entropy(hyper_sim.t(), labels)

        return (loss_v2o + loss_o2v + loss_hyper_v2o + loss_hyper_o2v) * 0.25
