import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.FM import FlowMatchingModule, Composer
from modules.HC import HyperProjector
import os
import clip

CLIP_MODEL_NAME = "ViT-B/16"


def get_clip_text_encoder(freeze_clip, clip_model, device, emb_dim=300):
    """
    鏋勫缓CLIP鏂囨湰缂栫爜鍣紙杈撳嚭鎸囧畾缁村害鐗瑰緛锛?
    Args:
        freeze_clip: 鏄惁鍐荤粨CLIP鏉冮噸
        clip_model: 宸插姞杞界殑CLIP妯″瀷瀹炰緥
        device: 杩愯璁惧锛坈pu/cuda锛?
        emb_dim: 杈撳嚭鐗瑰緛缁村害
    Returns:
        encode_text: 鏂囨湰缂栫爜鍑芥暟
    """
    if freeze_clip:
        for param in clip_model.parameters():
            param.requires_grad = False

    # 鍏抽敭淇锛氬畾涔変负nn.Module骞舵敞鍐岋紝閬垮厤涓存椂鍙橀噺闂
    class TextProjection(nn.Module):
        def __init__(self, in_dim=512, out_dim=emb_dim):
            super().__init__()
            self.proj = nn.Linear(in_dim, out_dim)

        def forward(self, x):
            return self.proj(x)

    # 鍒濆鍖栨姇褰卞眰骞剁Щ鍒版寚瀹氳澶?
    proj = TextProjection(512, emb_dim).to(device, dtype=clip_model.dtype)
    if freeze_clip:
        for param in proj.parameters():
            param.requires_grad = False

    clip_tokenizer = clip.tokenize

    def encode_text(prompts: list):
        # 鏂囨湰prompt杞瑃oken骞剁Щ鍒板搴旇澶?
        tokens = clip_tokenizer(prompts).to(device)

        # 鏍规嵁鏄惁鍐荤粨鎺у埗姊害
        context = torch.no_grad() if freeze_clip else torch.enable_grad()
        with context:
            # CLIP鏂囨湰缂栫爜鍣ㄨ緭鍑?12缁寸壒寰?
            text_feat = clip_model.encode_text(tokens)
            # 鎶曞奖鍒版寚瀹氱淮搴︼紙鍜岃瘝宓屽叆灞傜淮搴﹀榻愶級
            text_feat = proj(text_feat)

        # 褰掍竴鍖栵紙鍜岃瘝宓屽叆灞傜殑褰掍竴鍖栭€昏緫涓€鑷达級
        return F.normalize(text_feat, dim=-1)

    # 缁戝畾灞炴€э紝鏂逛究鍚庣画璋冪敤鍜屽弬鏁扮鐞?
    encode_text.clip_model = clip_model
    encode_text.device = device
    encode_text.proj = proj  # 缁戝畾鎶曞奖灞傦紝鏂逛究娉ㄥ唽涓烘ā鍨嬪瓙妯″潡
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
        print("Building image FlowHC model")
        if feat_extractor == "clip":
            print("Using CLIP image encoder")
        else:
            raise ValueError(f"Unsupported feature extractor: {feat_extractor}; expected clip")

        # 1. 鍒濆鍖栬澶?
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weight_path = "/home/ubuntu/wisdom1/jiangwen/CLIP/weights/ViT-B-16.pt"

        # 2. 鍔犺浇CLIP妯″瀷
        if os.path.exists(weight_path):
            # 鍔犺浇鏈湴CLIP ViT-B/16鏉冮噸
            clip_model, _ = clip.load(weight_path, device=self.device)
            print(f"Loaded local CLIP weights: {weight_path}")
        else:
            # 鑷姩涓嬭浇CLIP ViT-B/16鏉冮噸锛堝鐢ㄦ柟妗堬級
            clip_model, _ = clip.load(CLIP_MODEL_NAME, device=self.device)
            print("Downloaded and loaded CLIP ViT-B/16 weights")

        # 3. 鍒濆鍖栬瑙夌紪鐮佸櫒
        self.visual_encoder = clip_model.visual

        # 4. 鍒濆鍖栨枃鏈紪鐮佸櫒锛堜慨澶嶏細浼犲叆device鍜宔mb_dim锛?
        self.clip_text_encoder = get_clip_text_encoder(
            freeze_clip=True,
            clip_model=clip_model,
            device=self.device,
            emb_dim=emb_dim
        )
        # 鍏抽敭淇锛氭敞鍐屾枃鏈姇褰卞眰涓烘ā鍨嬪瓙妯″潡
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
            attrs = torch.LongTensor(attrs).to(self.device)
            objs = torch.LongTensor(objs).to(self.device)
            pairs = torch.LongTensor(pairs).to(self.device)
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

        # 鍒濆鍖栬瘝宓屽叆灞?
        self.attr_embedder = nn.Embedding(len(dset.attrs), emb_dim).to(self.device)
        self.obj_embedder = nn.Embedding(len(dset.objs), emb_dim).to(self.device)

        # init with word embeddings
        if emb_init == "clip":
            print(f"Initializing verb/object embeddings with CLIP text features: dim={emb_dim}")
            # 1. 鏋勯€燙LIP鏂囨湰prompt锛堢洿鎺ョ敤灞炴€?鐗╀綋鍚嶇О锛?
            verb_prompts = [f"{v}" for v in dset.attrs]
            obj_prompts = [f"{o}" for o in dset.objs]

            # 2. 璋冪敤CLIP鏂囨湰缂栫爜鍣ㄨ幏鍙栬瘝宓屽叆
            verb_emb = self.clip_text_encoder(verb_prompts)  # (num_attrs, emb_dim)
            obj_emb = self.clip_text_encoder(obj_prompts)  # (num_objs, emb_dim)

            # 3. 璁惧鍜屾暟鎹被鍨嬪榻?
            verb_emb = verb_emb.to(self.attr_embedder.weight.device)
            obj_emb = obj_emb.to(self.obj_embedder.weight.device)
            verb_emb = verb_emb.type(self.attr_embedder.weight.dtype)
            obj_emb = obj_emb.type(self.obj_embedder.weight.dtype)

            # 4. 璧嬪€煎埌宓屽叆灞傛潈閲?
            self.attr_embedder.weight.data.copy_(verb_emb)
            self.obj_embedder.weight.data.copy_(obj_emb)
            print(f"CLIP text embedding init complete: attrs={verb_emb.shape}, objs={obj_emb.shape}")
        else:
            # 淇锛氶敊璇俊鎭尮閰嶅垽鏂潯浠?
            raise ValueError(f"Unsupported embedding init: {emb_init}; expected clip")

        # static inputs
        if static_inp:
            for param in self.attr_embedder.parameters():
                param.requires_grad = False
            for param in self.obj_embedder.parameters():
                param.requires_grad = False

        # Composition MLP
        self.o_projection1 = nn.Linear(emb_dim, emb_dim).to(self.device)
        self.v_projection1 = nn.Linear(emb_dim, emb_dim).to(self.device)

        # feature map -> verb/object latents
        self.flow_matching = FlowMatchingModule(
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
        self.composer = Composer(
            dim=feat_dim,
            hidden_dim=composer_hidden_dim,
            dropout=composer_dropout,
            composer_type=composer_type,
        ).to(self.device)
        self.fc_comp = nn.Linear(feat_dim, emb_dim).to(self.device)
        self.hyper_projector = HyperProjector(dim=feat_dim).to(self.device)

        # 棰勮绠桟LIP褰掍竴鍖栫殑鍧囧€煎拰鏂瑰樊骞剁Щ鍒拌澶?
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
        """Extract CLIP ViT patch features without global averaging."""
        if x.detach().amax() > 2.0:
            x = x / 255.0
        if x.detach().amin() >= 0.0 and x.detach().amax() <= 1.0:
            x = (x - self.clip_mean) / self.clip_std
        # 楠岃瘉 visual_encoder 鏄?CLIP 瑙嗚缂栫爜鍣?
        if not hasattr(self.visual_encoder, 'conv1'):
            raise ValueError("visual_encoder is not a CLIP visual encoder")

        # CLIP 瑙嗚缂栫爜鍣ㄥ墠鍚戜紶鎾?
        x = self.visual_encoder.conv1(x)  # (B, 768, 14, 14)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (B, 768, 196)
        x = x.permute(0, 2, 1)  # (B, 196, 768)

        # 娣诲姞 class token
        class_token = self.visual_encoder.class_embedding.to(x.dtype) + \
                      torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([class_token, x], dim=1)

        # 娣诲姞浣嶇疆缂栫爜
        x = x + self.visual_encoder.positional_embedding.to(x.dtype)
        x = self.visual_encoder.ln_pre(x)

        # Transformer 缂栫爜
        x = x.permute(1, 0, 2)  # NLD -> LND
        for resblock in self.visual_encoder.transformer.resblocks:
            x = resblock(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # 鍘绘帀 class token锛屼繚鐣?patch 鐗瑰緛
        patch_feats = x[:, 1:, :]  # (B, 196, 768) 鈫?14x14

        # 閲嶅涓虹┖闂寸壒寰?
        B, _, D = patch_feats.shape
        patch_feats = patch_feats.reshape(B, 14, 14, D)  # (B, 14, 14, 768)
        patch_feats = patch_feats.permute(0, 3, 1, 2)  # (B, 768, 14, 14)

        # Convert to FlowMatchingModule feature-map format.
        patch_feats = patch_feats.unsqueeze(2)  # (B, 768, 1, 14, 14)

        return patch_feats

    def val_forward_closed(self, x, pairs, visual=False):
        img_feat = self._vit_feature_extractor_no_avg(x)  # (B, 768, 1, 14, 14)
        z_v, z_o = self.flow_matching(img_feat)
        z_v_mean = z_v.mean(dim=1)
        z_o_mean = z_o.mean(dim=1)
        z_comp = self.composer.compose(z_v_mean, z_o_mean, normalize=False)

        v_feat = self.fc_v(z_v_mean)  # (B, D)
        v_feat_normed = F.normalize(v_feat, dim=-1)  # (B, D)
        o_feat = self.fc_o(z_o_mean)  # (B, D)
        o_feat_normed = F.normalize(o_feat, dim=-1)  # (B, D)
        comp_feat = self.fc_comp(z_comp)
        comp_feat_normed = F.normalize(comp_feat, dim=-1)

        all_verbs = self.attr_embedder(self.uniq_attrs)
        all_objs = self.obj_embedder(self.uniq_objs)

        v_emb = self.v_projection1(all_verbs)
        v_emb_normed = F.normalize(v_emb, dim=-1)
        o_emb = self.o_projection1(all_objs)  # n,c
        o_emb_normed = F.normalize(o_emb, dim=-1)

        p_v = torch.matmul(v_feat_normed, v_emb_normed.permute(1, 0)) * 0.5 + 0.5  # b,nv
        p_o = torch.matmul(o_feat_normed, o_emb_normed.permute(1, 0)) * 0.5 + 0.5  # b,no
        p_comp_v = torch.matmul(comp_feat_normed, v_emb_normed.permute(1, 0)) * 0.5 + 0.5
        p_comp_o = torch.matmul(comp_feat_normed, o_emb_normed.permute(1, 0)) * 0.5 + 0.5
        p_v = 0.5 * (p_v + p_comp_v)
        p_o = 0.5 * (p_o + p_comp_o)
        p_vo = p_v.unsqueeze(2) * p_o.unsqueeze(1)  # (B, N_v, N_o)

        verb_ids, obj_ids = pairs[:, 0], pairs[:, 1]
        pair_pred = p_vo[:, verb_ids, obj_ids]

        if visual:
            return p_v, p_o, pair_pred
        return pair_pred

    def train_forward_closed(self, x, verb_labels=None, obj_labels=None):
        img_feat = self._vit_feature_extractor_no_avg(x)  # (B, 768, 1, 14, 14)
        z_v, z_o, loss_flow = self.flow_matching(img_feat)
        z_v_mean = z_v.mean(dim=1)
        z_o_mean = z_o.mean(dim=1)
        z_comp = self.composer.compose(z_v_mean, z_o_mean, normalize=False)

        v_feat = self.fc_v(z_v_mean)  # (B, D)
        v_feat_normed = F.normalize(v_feat, dim=-1)  # (B, D)
        o_feat = self.fc_o(z_o_mean)  # (B, D)
        o_feat_normed = F.normalize(o_feat, dim=-1)  # (B, D)
        comp_feat = self.fc_comp(z_comp)
        comp_feat_normed = F.normalize(comp_feat, dim=-1)

        all_verbs = self.attr_embedder(self.uniq_attrs)
        all_objs = self.obj_embedder(self.uniq_objs)

        v_emb = self.v_projection1(all_verbs)
        v_emb_normed = F.normalize(v_emb, dim=-1)
        o_emb = self.o_projection1(all_objs)  # n,c
        o_emb_normed = F.normalize(o_emb, dim=-1)

        p_v = torch.matmul(v_feat_normed, v_emb_normed.permute(1, 0)) * 0.5 + 0.5  # b,nv
        p_o = torch.matmul(o_feat_normed, o_emb_normed.permute(1, 0)) * 0.5 + 0.5  # b,no
        pred = p_v.unsqueeze(2) * p_o.unsqueeze(1)

        p_comp_v = torch.matmul(comp_feat_normed, v_emb_normed.permute(1, 0)) * 0.5 + 0.5
        p_comp_o = torch.matmul(comp_feat_normed, o_emb_normed.permute(1, 0)) * 0.5 + 0.5
        pred_comp = p_comp_v.unsqueeze(2) * p_comp_o.unsqueeze(1)
        pred = 0.5 * (pred + pred_comp)
        p_v = 0.5 * (p_v + p_comp_v)
        p_o = 0.5 * (p_o + p_comp_o)

        loss_comp = F.mse_loss(z_comp, 0.5 * (z_v_mean + z_o_mean))
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

        additional_loss = (
            self.lambda_flow * loss_flow
            + self.lambda_comp * loss_comp
            + self.lambda_orth * loss_orth
            + self.lambda_hyper_comp * loss_hyper_comp
            + self.lambda_hyper_contrast * loss_hyper_contrast
        )

        return p_v, p_o, pred, additional_loss

    def forward(self, x, pair=None, verb_labels=None, obj_labels=None):
        # x(64,3,224,224)
        if self.training:
            pred = self.train_forward_closed(x, verb_labels=verb_labels, obj_labels=obj_labels)
        else:
            pred = self.val_forward_closed(x, pair)
        return pred
