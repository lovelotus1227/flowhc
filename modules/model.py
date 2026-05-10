import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.word_embedding import load_word_embeddings
from modules.video_models.swin_transformer_mmaction import get_swinvideo
from modules.FM import FlowMatchingModule, Composer
from modules.HC import HyperProjector
from modules.Aim import get_aim
import os
import clip
from dataclasses import dataclass
from typing import Optional, Tuple

CLIP_MODEL_NAME = "ViT-B/32"


@dataclass
class SwinTConfig:
    name: str = "swin_tiny"
    requires_grad: bool = True
    output_dim: int = 768

    @staticmethod
    def create_encoder():
        print("🚀 初始化 Swin-T 视频编码器（全参数可训练）")
        return get_swinvideo("swin_tiny")


@dataclass
class CLIPAIMConfig:
    name: str = "clip"
    freeze_clip: bool = True
    output_dim: int = 768

    @staticmethod
    def create_encoder(freeze_clip: bool = True):
        print("🚀 初始化 CLIP+AIM 视频编码器（CLIP骨干冻结）")
        aim_model = get_aim()
        print("✅ 成功加载手写AIM模型（ViT-B/32+Adapter）")

        if freeze_clip:
            for name, param in aim_model.named_parameters():
                if "Adapter" not in name and "temporal_embedding" not in name:
                    param.requires_grad = False
            print("✅ 冻结CLIP骨干权重，仅训练Adapter和时序嵌入层")

        class VideoEncoderWrapper(nn.Module):
            def __init__(self, aim_model):
                super().__init__()
                self.aim_model = aim_model

            def forward(self, x, output_format="swin"):
                return self.aim_model(x, output_format=output_format)

        return VideoEncoderWrapper(aim_model)


def get_clip_text_encoder(freeze_clip: bool = True, local_clip_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if local_clip_path and os.path.exists(local_clip_path):
        clip_model, preprocess = clip.load(local_clip_path, device=device)
    else:
        clip_model, preprocess = clip.load(CLIP_MODEL_NAME, device=device)

    if freeze_clip:
        for param in clip_model.parameters():
            param.requires_grad = False

    clip_tokenizer = clip.tokenize

    class ClipTextEncoder(nn.Module):
        def __init__(self, clip_model, proj, device):
            super().__init__()
            self.clip_model = clip_model
            self.proj = proj
            self.device = device
            for param in self.clip_model.parameters():
                param.requires_grad = False

        def forward(self, prompts: list):
            tokens = clip_tokenizer(prompts).to(self.device)
            with torch.no_grad():
                text_feat = self.clip_model.encode_text(tokens)
            text_feat = self.proj(text_feat)
            return F.normalize(text_feat, dim=-1)

    proj = nn.Linear(512, 300).to(device, dtype=clip_model.dtype)
    encoder = ClipTextEncoder(clip_model, proj, device)
    return encoder


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
            emb_init,
            static_inp,
            train_only,
            is_image,
            dset,
            num_frames=8,
            lambda_flow=0.1,
            lambda_comp=0.1,
            lambda_orth=0.03,
            lambda_hyper_comp=0.05,
            lambda_hyper_contrast=0.02,
            composer_type="gated",
            composer_hidden_dim=None,
            composer_dropout=0.1,
    ):
        super(MCR2CVL, self).__init__()
        self.lambda_flow = lambda_flow
        self.lambda_comp = lambda_comp
        self.lambda_orth = lambda_orth
        self.lambda_hyper_comp = lambda_hyper_comp
        self.lambda_hyper_contrast = lambda_hyper_contrast

        self.feat_extractor = feat_extractor
        self.clip_text_encoder = None

        if feat_extractor == "swin_tiny":
            print("=" * 60)
            print("🎯 使用 Swin-T 视频编码器（全参数可训练）")
            print("=" * 60)
            self.video_encoder = SwinTConfig.create_encoder()
            self.encoder_output_format = "swin"

        elif feat_extractor == "clip":
            print("=" * 60)
            print("🎯 使用 CLIP+AIM 视频编码器（CLIP骨干冻结）")
            print("=" * 60)
            self.video_encoder = CLIPAIMConfig.create_encoder(freeze_clip=True)
            self.encoder_output_format = "flatten"
            self.clip_text_encoder = get_clip_text_encoder(
                freeze_clip=True,
                local_clip_path="/home/ubuntu/wisdom1/jiangwen/CLIP/weights/ViT-B-32.pt"
            )

        else:
            raise ValueError(
                f"视频编码器出错！仅支持 'swin_tiny' 或 'clip'，当前传入：{feat_extractor}"
            )
        self.dset = dset
        self.is_image = is_image

        def get_all_ids(relevant_pairs):
            attrs, objs = zip(*relevant_pairs)
            attrs = [dset.attr2idx[attr] for attr in attrs]
            objs = [dset.obj2idx[obj] for obj in objs]
            pairs = [a for a in range(len(relevant_pairs))]
            attrs = torch.LongTensor(attrs)
            objs = torch.LongTensor(objs)
            pairs = torch.LongTensor(pairs)
            return attrs, objs, pairs

        val_attrs, val_objs, val_pairs = get_all_ids(self.dset.pairs)
        self.register_buffer('val_attrs', val_attrs)
        self.register_buffer('val_objs', val_objs)
        self.register_buffer('val_pairs', val_pairs)

        uniq_attrs, uniq_objs = torch.arange(len(self.dset.attrs)), \
            torch.arange(len(self.dset.objs))
        self.register_buffer('uniq_attrs', uniq_attrs)
        self.register_buffer('uniq_objs', uniq_objs)
        self.factor = 2

        self.train_forward = self.train_forward_closed

        if train_only:
            train_attrs, train_objs, train_pairs = get_all_ids(self.dset.train_pairs)
        else:
            train_attrs, train_objs, train_pairs = val_attrs, val_objs, val_pairs

        self.register_buffer('train_attrs', train_attrs)
        self.register_buffer('train_objs', train_objs)
        self.register_buffer('train_pairs', train_pairs)

        self.attr_embedder = nn.Embedding(len(dset.attrs), emb_dim)
        self.obj_embedder = nn.Embedding(len(dset.objs), emb_dim)

        if emb_init == "clip":
            print(f"📝 用 CLIP 文本嵌入初始化 verb/object {emb_dim}维")

            if self.clip_text_encoder is None:
                raise ValueError(
                    "使用 CLIP 初始化词嵌入时，feat_extractor 必须为 'clip'！"
                )

            verb_prompts = [f"{v}" for v in dset.attrs]
            obj_prompts = [f"{o}" for o in dset.objs]

            verb_emb = self.clip_text_encoder(verb_prompts)
            obj_emb = self.clip_text_encoder(obj_prompts)

            verb_emb = verb_emb.to(self.attr_embedder.weight.device)
            obj_emb = obj_emb.to(self.obj_embedder.weight.device)
            verb_emb = verb_emb.type(self.attr_embedder.weight.dtype)
            obj_emb = obj_emb.type(self.obj_embedder.weight.dtype)

            if verb_emb.shape[1] != emb_dim:
                proj = nn.Linear(verb_emb.shape[1], emb_dim).to(verb_emb.device)
                verb_emb = proj(verb_emb)
                obj_emb = proj(obj_emb)
                print(f"⚠️ CLIP特征维度{verb_emb.shape[1]}自动投影到{emb_dim}维")

            self.attr_embedder.weight.data.copy_(verb_emb)
            self.obj_embedder.weight.data.copy_(obj_emb)
            print(f"✅ CLIP文本嵌入初始化完成：attrs={verb_emb.shape}, objs={obj_emb.shape}")

            del self.clip_text_encoder
            self.clip_text_encoder = None

        else:
            print(f"📝 用 FastText 文本嵌入初始化 verb/object {emb_dim}维")
            pretrained_weight = load_word_embeddings(emb_init, dset.attrs)
            self.attr_embedder.weight.data.copy_(pretrained_weight)
            pretrained_weight = load_word_embeddings(emb_init, dset.objs)
            self.obj_embedder.weight.data.copy_(pretrained_weight)

        if static_inp:
            for param in self.attr_embedder.parameters():
                param.requires_grad = False
            for param in self.obj_embedder.parameters():
                param.requires_grad = False

        self.o_projection1 = nn.Linear(emb_dim, emb_dim)
        self.v_projection1 = nn.Linear(emb_dim, emb_dim)

        print("✅ 使用 FlowMatchingModule (时序感知+双向交互)")
        self.flow_matching = FlowMatchingModule(
            dim=feat_dim,
            num_heads=num_heads,
            num_latents=num_latents,
            num_layers=num_layers,
            eta=eta,
            is_image=self.is_image,
            num_frames=num_frames,
        )

        self.fc_v = nn.Linear(feat_dim, emb_dim)
        self.fc_o = nn.Linear(feat_dim, emb_dim)

        print(f"✅ 使用 Gated Composer (通道门控融合)")
        self.composer = Composer(
            dim=feat_dim,
            hidden_dim=composer_hidden_dim,
            dropout=composer_dropout,
            composer_type=composer_type
        )

        self.fc_comp = nn.Linear(feat_dim, emb_dim)

        print("✅ 使用 HyperProjector (欧式→双曲投影+双曲组合)")
        self.hyper_projector = HyperProjector(dim=feat_dim)

    def freeze_representations(self):
        print('✅ 冻结verb-object嵌入层，仅训练FlowMatching和投影层')
        for param in self.attr_embedder.parameters():
            param.requires_grad = False
        for param in self.obj_embedder.parameters():
            param.requires_grad = False

    def val_forward_closed(self, x, pairs, visual=False):
        if self.feat_extractor == "clip":
            vid_feat = self.video_encoder(x, output_format="flatten")
        else:
            vid_feat = self.video_encoder(x)

        z_v, z_o = self.flow_matching(vid_feat)
        z_v_mean = z_v.mean(dim=1)
        z_o_mean = z_o.mean(dim=1)

        z_comp = self.composer.compose(z_v_mean, z_o_mean, normalize=False)

        v_feat = self.fc_v(z_v_mean)
        o_feat = self.fc_o(z_o_mean)
        comp_feat = self.fc_comp(z_comp)

        v_feat_normed = F.normalize(v_feat, dim=-1)
        o_feat_normed = F.normalize(o_feat, dim=-1)
        comp_feat_normed = F.normalize(comp_feat, dim=-1)

        all_verbs, all_objs = self.attr_embedder(self.uniq_attrs), self.obj_embedder(self.uniq_objs)

        v_emb = self.v_projection1(all_verbs)
        v_emb_normed = F.normalize(v_emb, dim=-1)
        o_emb = self.o_projection1(all_objs)
        o_emb_normed = F.normalize(o_emb, dim=-1)

        p_v = torch.matmul(v_feat_normed, v_emb_normed.t()) * 0.5 + 0.5
        p_o = torch.matmul(o_feat_normed, o_emb_normed.t()) * 0.5 + 0.5
        p_comp_v = torch.matmul(comp_feat_normed, v_emb_normed.t()) * 0.5 + 0.5
        p_comp_o = torch.matmul(comp_feat_normed, o_emb_normed.t()) * 0.5 + 0.5

        p_v = (p_v + p_comp_v) / 2
        p_o = (p_o + p_comp_o) / 2

        p_vo_simple = p_v.unsqueeze(2) * p_o.unsqueeze(1)
        verb_ids, obj_ids = pairs[:, 0].long(), pairs[:, 1].long()
        pair_pred = p_vo_simple[:, verb_ids, obj_ids]

        if visual:
            return p_v, p_o, pair_pred
        return pair_pred

    def train_forward_closed(self, x, verb_labels=None, obj_labels=None):
        if self.feat_extractor == "clip":
            vid_feat = self.video_encoder(x, output_format="flatten")
        else:
            vid_feat = self.video_encoder(x)

        z_v, z_o, loss_flow = self.flow_matching(vid_feat)
        z_v_mean = z_v.mean(dim=1)
        z_o_mean = z_o.mean(dim=1)

        z_comp = self.composer.compose(z_v_mean, z_o_mean, normalize=False)

        v_feat = self.fc_v(z_v_mean)
        o_feat = self.fc_o(z_o_mean)
        comp_feat = self.fc_comp(z_comp)

        v_feat_normed = F.normalize(v_feat, dim=-1)
        o_feat_normed = F.normalize(o_feat, dim=-1)
        comp_feat_normed = F.normalize(comp_feat, dim=-1)

        all_verbs, all_objs = self.attr_embedder(self.uniq_attrs), self.obj_embedder(self.uniq_objs)

        v_emb = self.v_projection1(all_verbs)
        v_emb_normed = F.normalize(v_emb, dim=-1)
        o_emb = self.o_projection1(all_objs)
        o_emb_normed = F.normalize(o_emb, dim=-1)

        p_v = torch.matmul(v_feat_normed, v_emb_normed.t()) * 0.5 + 0.5
        p_o = torch.matmul(o_feat_normed, o_emb_normed.t()) * 0.5 + 0.5
        pred = p_v.unsqueeze(2) * p_o.unsqueeze(1)

        p_comp_v = torch.matmul(comp_feat_normed, v_emb_normed.t()) * 0.5 + 0.5
        p_comp_o = torch.matmul(comp_feat_normed, o_emb_normed.t()) * 0.5 + 0.5
        pred_comp = p_comp_v.unsqueeze(2) * p_comp_o.unsqueeze(1)

        pred = (pred + pred_comp) / 2
        p_v = (p_v + p_comp_v) / 2
        p_o = (p_o + p_comp_o) / 2

        loss_comp = F.mse_loss(z_comp, (z_v_mean + z_o_mean) * 0.5)

        loss_orth = z_v_mean.new_tensor(0.0)
        if self.lambda_orth > 0:
            loss_orth = self.flow_matching.orthogonal_flow_loss(
                z_v_mean,
                z_o_mean,
                verb_labels=verb_labels,
                obj_labels=obj_labels,
            )

        loss_hyper_comp = z_v_mean.new_tensor(0.0)
        if self.lambda_hyper_comp > 0:
            loss_hyper_comp = self.hyper_projector.hyper_composition_loss(z_v_mean, z_o_mean)

        loss_hyper_contrast = z_v_mean.new_tensor(0.0)
        if self.lambda_hyper_contrast > 0:
            loss_hyper_contrast = self.hyper_projector.hyper_contrastive_loss(z_v_mean, z_o_mean)

        total_additional_loss = (
            self.lambda_flow * loss_flow
            + self.lambda_comp * loss_comp
            + self.lambda_orth * loss_orth
            + self.lambda_hyper_comp * loss_hyper_comp
            + self.lambda_hyper_contrast * loss_hyper_contrast
        )
        self.latest_loss_terms = {
            "loss_flow": loss_flow.detach(),
            "loss_comp_aux": loss_comp.detach(),
            "loss_orth": loss_orth.detach(),
            "loss_hyper_comp": loss_hyper_comp.detach(),
            "loss_hyper_contrast": loss_hyper_contrast.detach(),
        }
        return p_v, p_o, pred, total_additional_loss

    def forward(self, x, pair=None, verb_labels=None, obj_labels=None):
        if self.training:
            return self.train_forward_closed(x, verb_labels=verb_labels, obj_labels=obj_labels)
        else:
            return self.val_forward_closed(x, pair)
