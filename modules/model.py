import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.word_embedding import load_word_embeddings
from modules.video_models.swin_transformer_mmaction import get_swinvideo
from modules.resampler import Resampler


class MCR2CVL(nn.Module):
    def __init__(
        self, 
        emb_dim,
        feat_dim,
        num_heads,
        num_latents,
        num_layers,
        eta,
        feat_extractor,
        dset,
        emb_init="fasttext",
        static_inp=False,
        train_only=True, # closed-world 
    ):
        super(MCR2CVL, self).__init__()

        self.video_encoder = get_swinvideo(feat_extractor)
        self.dset = dset

        def get_all_ids(relevant_pairs):
            # Precompute validation pairs
            attrs, objs = zip(*relevant_pairs)
            attrs = [dset.attr2idx[attr] for attr in attrs]
            objs = [dset.obj2idx[obj] for obj in objs]
            pairs = [a for a in range(len(relevant_pairs))]
            attrs = torch.Tensor(attrs)
            objs = torch.Tensor(objs)
            pairs = torch.Tensor(pairs)
            return attrs, objs, pairs

        # Validation
        val_attrs, val_objs, val_pairs = get_all_ids(self.dset.pairs)
        self.register_buffer('val_attrs', val_attrs)
        self.register_buffer('val_objs', val_objs)
        self.register_buffer('val_pairs', val_pairs)

        # for indivual projections
        uniq_attrs, uniq_objs = torch.arange(len(self.dset.attrs)), \
                                torch.arange(len(self.dset.objs))
        self.register_buffer('uniq_attrs', uniq_attrs)
        self.register_buffer('uniq_objs', uniq_objs)
        self.factor = 2

        self.train_forward = self.train_forward_closed

        # Precompute training compositions
        if train_only:
            train_attrs, train_objs, train_pairs = get_all_ids(self.dset.train_pairs)
        else:
            train_attrs, train_objs, train_pairs = val_attrs, val_objs, val_pairs

        self.register_buffer('train_attrs', train_attrs)
        self.register_buffer('train_objs', train_objs)
        self.register_buffer('train_pairs', train_pairs)

        self.attr_embedder = nn.Embedding(len(dset.attrs), emb_dim)
        self.obj_embedder = nn.Embedding(len(dset.objs), emb_dim)

        # init with word embeddings
        if emb_init:
            pretrained_weight = load_word_embeddings(emb_init, dset.attrs)
            self.attr_embedder.weight.data.copy_(pretrained_weight)
            pretrained_weight = load_word_embeddings(emb_init, dset.objs)
            self.obj_embedder.weight.data.copy_(pretrained_weight)

        # static inputs
        if static_inp:
            for param in self.attr_embedder.parameters():
                param.requires_grad = False
            for param in self.obj_embedder.parameters():
                param.requires_grad = False

        # Composition MLP
        self.o_projection1 = nn.Linear(emb_dim, emb_dim)
        self.v_projection1 = nn.Linear(emb_dim, emb_dim)

        # feature map -> latent
        self.resampler = Resampler(
            dim=feat_dim,
            num_heads=num_heads,
            num_latents=num_latents,
            num_layers=num_layers,
            eta=eta,
        )
        
        # attention pooling
        self.fc_v = nn.Linear(feat_dim, emb_dim)
        self.fc_o = nn.Linear(feat_dim, emb_dim)

    def freeze_representations(self):
        print('Freezing representations')
        # for param in self.video_embedder.parameters():
        #     param.requires_grad = False
        for param in self.attr_embedder.parameters():
            param.requires_grad = False
        for param in self.obj_embedder.parameters():
            param.requires_grad = False

    def val_forward_closed(self, x, pairs, visual=False):
        vid_feat = self.video_encoder(x)  # (B, L, D)

        z = self.resampler(vid_feat) # (B, K, D)
        z = z.mean(dim=1) # (B, D)

        v_feat = self.fc_v(z) # (B, D)
        v_feat_normed = F.normalize(v_feat, dim=-1) # (B, D)
        o_feat = self.fc_o(z) # (B, D)
        o_feat_normed = F.normalize(o_feat, dim=-1) # (B, D)

        all_verbs, all_objs = self.attr_embedder(self.uniq_attrs), self.obj_embedder(self.uniq_objs)

        v_emb = self.v_projection1(all_verbs)
        v_emb_normed = F.normalize(v_emb, dim=-1)
        o_emb = self.o_projection1(all_objs)  # n,c
        o_emb_normed = F.normalize(o_emb, dim=-1)

        p_v = torch.matmul(v_feat_normed, v_emb_normed.permute(1, 0)) * 0.5 + 0.5  # b,nv
        p_o = torch.matmul(o_feat_normed, o_emb_normed.permute(1, 0)) * 0.5 + 0.5  # b,no
        p_vo = p_v.unsqueeze(2) * p_o.unsqueeze(1) # (B, N_v, N_o)

        verb_ids, obj_ids = pairs[:, 0], pairs[:, 1]
        pair_pred = p_vo[:, verb_ids, obj_ids] 

        if visual:
            return p_v, p_o, pair_pred

        return pair_pred

    def train_forward_closed(self, x):
        vid_feat = self.video_encoder(x) # (B, L, D)

        z = self.resampler(vid_feat) # (B, K, D)
        z = z.mean(dim=1) # (B, D)

        v_feat = self.fc_v(z) # (B, D)
        v_feat_normed = F.normalize(v_feat, dim=-1) # (B, D)
        o_feat = self.fc_o(z) # (B, D)
        o_feat_normed = F.normalize(o_feat, dim=-1) # (B, D)

        all_verbs, all_objs = self.attr_embedder(self.uniq_attrs), self.obj_embedder(self.uniq_objs)

        v_emb = self.v_projection1(all_verbs)
        v_emb_normed = F.normalize(v_emb, dim=-1)
        o_emb = self.o_projection1(all_objs)  # n,c
        o_emb_normed = F.normalize(o_emb, dim=-1)

        p_v = torch.matmul(v_feat_normed, v_emb_normed.permute(1, 0)) * 0.5 + 0.5  # b,nv
        p_o = torch.matmul(o_feat_normed, o_emb_normed.permute(1, 0)) * 0.5 + 0.5  # b,no

        pred = p_v.unsqueeze(2) * p_o.unsqueeze(1)

        return p_v, p_o, pred

    def forward(self, x, pair=None):
        if self.training:
            pred = self.train_forward_closed(x)
        else:
            pred = self.val_forward_closed(x, pair)

        return pred