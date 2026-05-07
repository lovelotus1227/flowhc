import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.resampler import Resampler
import os
import clip

CLIP_MODEL_NAME = "ViT-B/16"


def get_clip_text_encoder(freeze_clip, clip_model, device, emb_dim=300):
    """
    构建CLIP文本编码器（输出指定维度特征）
    Args:
        freeze_clip: 是否冻结CLIP权重
        clip_model: 已加载的CLIP模型实例
        device: 运行设备（cpu/cuda）
        emb_dim: 输出特征维度
    Returns:
        encode_text: 文本编码函数
    """
    if freeze_clip:
        for param in clip_model.parameters():
            param.requires_grad = False

    # 关键修复：定义为nn.Module并注册，避免临时变量问题
    class TextProjection(nn.Module):
        def __init__(self, in_dim=512, out_dim=emb_dim):
            super().__init__()
            self.proj = nn.Linear(in_dim, out_dim)

        def forward(self, x):
            return self.proj(x)

    # 初始化投影层并移到指定设备
    proj = TextProjection(512, emb_dim).to(device, dtype=clip_model.dtype)
    if freeze_clip:
        for param in proj.parameters():
            param.requires_grad = False

    clip_tokenizer = clip.tokenize

    def encode_text(prompts: list):
        # 文本prompt转token并移到对应设备
        tokens = clip_tokenizer(prompts).to(device)

        # 根据是否冻结控制梯度
        context = torch.no_grad() if freeze_clip else torch.enable_grad()
        with context:
            # CLIP文本编码器输出512维特征
            text_feat = clip_model.encode_text(tokens)
            # 投影到指定维度（和词嵌入层维度对齐）
            text_feat = proj(text_feat)

        # 归一化（和词嵌入层的归一化逻辑一致）
        return F.normalize(text_feat, dim=-1)

    # 绑定属性，方便后续调用和参数管理
    encode_text.clip_model = clip_model
    encode_text.device = device
    encode_text.proj = proj  # 绑定投影层，方便注册为模型子模块
    encode_text.emb_dim = emb_dim

    return encode_text


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
    ):
        super(MCR2CVL, self).__init__()
        print("构建图像模型")
        if feat_extractor == "clip":
            print("✅ 使用CLIP图像编码器")
        else:
            raise ValueError(f"特征提取器出错！仅支持 'clip'，当前传入：{feat_extractor}")

        # 1. 初始化设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weight_path = "/home/ubuntu/wisdom1/jiangwen/CLIP/weights/ViT-B-16.pt"

        # 2. 加载CLIP模型
        if os.path.exists(weight_path):
            # 加载本地CLIP ViT-B/16权重
            clip_model, _ = clip.load(weight_path, device=self.device)
            print(f"✅ 加载本地CLIP权重: {weight_path}")
        else:
            # 自动下载CLIP ViT-B/16权重（备用方案）
            clip_model, _ = clip.load(CLIP_MODEL_NAME, device=self.device)
            print("✅ 自动下载并加载CLIP ViT-B/16权重")

        # 3. 初始化视觉编码器
        self.visual_encoder = clip_model.visual

        # 4. 初始化文本编码器（修复：传入device和emb_dim）
        self.clip_text_encoder = get_clip_text_encoder(
            freeze_clip=True,
            clip_model=clip_model,
            device=self.device,
            emb_dim=emb_dim
        )
        # 关键修复：注册文本投影层为模型子模块
        self.text_proj = self.clip_text_encoder.proj

        self.is_image = is_image
        self.dset = dset
        self.emb_dim = emb_dim

        def get_all_ids(relevant_pairs):
            # Precompute validation pairs
            attrs, objs = zip(*relevant_pairs)
            attrs = [dset.attr2idx[attr] for attr in attrs]
            objs = [dset.obj2idx[obj] for obj in objs]
            pairs = [a for a in range(len(relevant_pairs))]
            attrs = torch.Tensor(attrs).to(self.device)
            objs = torch.Tensor(objs).to(self.device)
            pairs = torch.Tensor(pairs).to(self.device)
            return attrs, objs, pairs

        # Validation
        val_attrs, val_objs, val_pairs = get_all_ids(self.dset.pairs)
        self.register_buffer('val_attrs', val_attrs)
        self.register_buffer('val_objs', val_objs)
        self.register_buffer('val_pairs', val_pairs)

        # for indivual projections
        uniq_attrs = torch.arange(len(self.dset.attrs)).to(self.device)
        uniq_objs = torch.arange(len(self.dset.objs)).to(self.device)
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

        # 初始化词嵌入层
        self.attr_embedder = nn.Embedding(len(dset.attrs), emb_dim).to(self.device)
        self.obj_embedder = nn.Embedding(len(dset.objs), emb_dim).to(self.device)

        # init with word embeddings
        if emb_init == "clip":
            print(f"✅ 用CLIP文本嵌入初始化verb/object {emb_dim}维")
            # 1. 构造CLIP文本prompt（直接用属性/物体名称）
            verb_prompts = [f"{v}" for v in dset.attrs]
            obj_prompts = [f"{o}" for o in dset.objs]

            # 2. 调用CLIP文本编码器获取词嵌入
            verb_emb = self.clip_text_encoder(verb_prompts)  # (num_attrs, emb_dim)
            obj_emb = self.clip_text_encoder(obj_prompts)  # (num_objs, emb_dim)

            # 3. 设备和数据类型对齐
            verb_emb = verb_emb.to(self.attr_embedder.weight.device)
            obj_emb = obj_emb.to(self.obj_embedder.weight.device)
            verb_emb = verb_emb.type(self.attr_embedder.weight.dtype)
            obj_emb = obj_emb.type(self.obj_embedder.weight.dtype)

            # 4. 赋值到嵌入层权重
            self.attr_embedder.weight.data.copy_(verb_emb)
            self.obj_embedder.weight.data.copy_(obj_emb)
            print(f"✅ CLIP文本嵌入初始化完成：attrs={verb_emb.shape}, objs={obj_emb.shape}")
        else:
            # 修复：错误信息匹配判断条件
            raise ValueError(f"词嵌入初始化出错！仅支持 'clip'，当前传入：{emb_init}")

        # static inputs
        if static_inp:
            for param in self.attr_embedder.parameters():
                param.requires_grad = False
            for param in self.obj_embedder.parameters():
                param.requires_grad = False

        # Composition MLP
        self.o_projection1 = nn.Linear(emb_dim, emb_dim).to(self.device)
        self.v_projection1 = nn.Linear(emb_dim, emb_dim).to(self.device)

        # feature map -> latent
        self.resampler = Resampler(
            dim=feat_dim,
            num_heads=num_heads,
            num_latents=num_latents,
            num_layers=num_layers,
            eta=eta,
            is_image=True,
        ).to(self.device)

        # attention pooling
        self.fc_v = nn.Linear(feat_dim, emb_dim).to(self.device)
        self.fc_o = nn.Linear(feat_dim, emb_dim).to(self.device)

        # 预计算CLIP归一化的均值和方差并移到设备
        self.register_buffer(
            'clip_mean',
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'clip_std',
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        )

    def freeze_representations(self):
        print('Freezing representations')
        for param in self.attr_embedder.parameters():
            param.requires_grad = False
        for param in self.obj_embedder.parameters():
            param.requires_grad = False

    def _vit_feature_extractor_no_avg(self, x):
        """适配CLIP ViT-B/16的特征提取逻辑（修复维度错误）"""
        # CLIP 官方归一化（不再依赖卷积层权重）
        x = x / 255.0  # 先转0-1范围
        x = (x - self.clip_mean) / self.clip_std  # 使用固定归一化参数

        # 验证 visual_encoder 是 CLIP 视觉编码器
        if not hasattr(self.visual_encoder, 'conv1'):
            raise ValueError("self.visual_encoder 不是 CLIP 视觉编码器！")

        # CLIP 视觉编码器前向传播
        x = self.visual_encoder.conv1(x)  # (B, 768, 14, 14)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (B, 768, 196)
        x = x.permute(0, 2, 1)  # (B, 196, 768)

        # 添加 class token
        class_token = self.visual_encoder.class_embedding.to(x.dtype) + \
                      torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([class_token, x], dim=1)

        # 添加位置编码
        x = x + self.visual_encoder.positional_embedding.to(x.dtype)
        x = self.visual_encoder.ln_pre(x)

        # Transformer 编码
        x = x.permute(1, 0, 2)  # NLD -> LND
        for resblock in self.visual_encoder.transformer.resblocks:
            x = resblock(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # 去掉 class token，保留 patch 特征
        patch_feats = x[:, 1:, :]  # (B, 196, 768) → 14x14

        # 重塑为空间特征
        B, _, D = patch_feats.shape
        patch_feats = patch_feats.reshape(B, 14, 14, D)  # (B, 14, 14, 768)
        patch_feats = patch_feats.permute(0, 3, 1, 2)  # (B, 768, 14, 14)

        # 转换为 Resampler 需要的 5 维格式
        patch_feats = patch_feats.unsqueeze(2)  # (B, 768, 1, 14, 14)

        return patch_feats

    def val_forward_closed(self, x, pairs, visual=False):
        img_feat = self._vit_feature_extractor_no_avg(x)  # (B, 768, 1, 14, 14)
        z = self.resampler(img_feat)  # (B, K, D)
        z = z.mean(dim=1)  # (B, D)

        v_feat = self.fc_v(z)  # (B, D)
        v_feat_normed = F.normalize(v_feat, dim=-1)  # (B, D)
        o_feat = self.fc_o(z)  # (B, D)
        o_feat_normed = F.normalize(o_feat, dim=-1)  # (B, D)

        all_verbs = self.attr_embedder(self.uniq_attrs)
        all_objs = self.obj_embedder(self.uniq_objs)

        v_emb = self.v_projection1(all_verbs)
        v_emb_normed = F.normalize(v_emb, dim=-1)
        o_emb = self.o_projection1(all_objs)  # n,c
        o_emb_normed = F.normalize(o_emb, dim=-1)

        p_v = torch.matmul(v_feat_normed, v_emb_normed.permute(1, 0)) * 0.5 + 0.5  # b,nv
        p_o = torch.matmul(o_feat_normed, o_emb_normed.permute(1, 0)) * 0.5 + 0.5  # b,no
        p_vo = p_v.unsqueeze(2) * p_o.unsqueeze(1)  # (B, N_v, N_o)

        verb_ids, obj_ids = pairs[:, 0], pairs[:, 1]
        pair_pred = p_vo[:, verb_ids, obj_ids]

        if visual:
            return p_v, p_o, pair_pred
        return pair_pred

    def train_forward_closed(self, x):
        img_feat = self._vit_feature_extractor_no_avg(x)  # (B, 768, 1, 14, 14)
        z = self.resampler(img_feat)  # (B, K, D)
        z = z.mean(dim=1)  # (B, D)

        v_feat = self.fc_v(z)  # (B, D)
        v_feat_normed = F.normalize(v_feat, dim=-1)  # (B, D)
        o_feat = self.fc_o(z)  # (B, D)
        o_feat_normed = F.normalize(o_feat, dim=-1)  # (B, D)

        all_verbs = self.attr_embedder(self.uniq_attrs)
        all_objs = self.obj_embedder(self.uniq_objs)

        v_emb = self.v_projection1(all_verbs)
        v_emb_normed = F.normalize(v_emb, dim=-1)
        o_emb = self.o_projection1(all_objs)  # n,c
        o_emb_normed = F.normalize(o_emb, dim=-1)

        p_v = torch.matmul(v_feat_normed, v_emb_normed.permute(1, 0)) * 0.5 + 0.5  # b,nv
        p_o = torch.matmul(o_feat_normed, o_emb_normed.permute(1, 0)) * 0.5 + 0.5  # b,no
        pred = p_v.unsqueeze(2) * p_o.unsqueeze(1)

        return p_v, p_o, pred

    def forward(self, x, pair=None):
        # x(64,3,224,224)
        if self.training:
            pred = self.train_forward_closed(x)
        else:
            pred = self.val_forward_closed(x, pair)
        return pred