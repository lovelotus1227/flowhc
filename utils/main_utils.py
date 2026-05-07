import torch
import numpy as np
import importlib
import random
from collections import OrderedDict
from omegaconf import OmegaConf, DictConfig


def instantiate(cfg, **overrides):
    module, cls = cfg["_target_"].rsplit(".", 1)
    base = cfg.get("params", {})
    if isinstance(base, DictConfig):
        base = OmegaConf.to_container(base, resolve=True)
    params = {**(base or {}), **overrides}
    return getattr(importlib.import_module(module), cls)(**params)


def get_optimizer_vm(cfg, model):
    param_groups = {
        'comp_param': [],
        'aim_param': [],
        'video_en_param': [],
        'flow_param': [],
        'hyper_param': []
    }
    param_names = {
        'comp_param': [],
        'aim_param': [],
        'video_en_param': [],
        'flow_param': [],
        'hyper_param': []
    }

    is_clip = False
    for name, _ in model.named_parameters():
        if 'Adapter' in name or 'temporal_embedding' in name:
            is_clip = True
            break

    if model.is_image:
        model_type = "Image Model"
    else:
        model_type = "CLIP+AIM + FlowHC" if is_clip else "Swin-T + FlowHC"
    
    print(f"=== 模型检测: {model_type} ===")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'hyper' in name or 'HC' in name:
            param_groups['hyper_param'].append(param)
            param_names['hyper_param'].append(name)
            continue
            
        if model.is_image:
            if 'visual_encoder' in name:
                param_groups['video_en_param'].append(param)
                param_names['video_en_param'].append(name)
            else:
                param_groups['comp_param'].append(param)
                param_names['comp_param'].append(name)
        else:
            if is_clip:
                if 'video_encoder' in name and ('Adapter' in name or 'temporal_embedding' in name):
                    param_groups['aim_param'].append(param)
                    param_names['aim_param'].append(name)
                elif 'flow_predictor' in name or 'flow_matching' in name:
                    param_groups['flow_param'].append(param)
                    param_names['flow_param'].append(name)
                elif 'video_encoder' not in name:
                    param_groups['comp_param'].append(param)
                    param_names['comp_param'].append(name)
            else:
                if 'video_encoder' in name:
                    param_groups['video_en_param'].append(param)
                    param_names['video_en_param'].append(name)
                elif 'flow_predictor' in name or 'flow_matching' in name:
                    param_groups['flow_param'].append(param)
                    param_names['flow_param'].append(name)
                else:
                    param_groups['comp_param'].append(param)
                    param_names['comp_param'].append(name)

    print("=" * 70)
    print(f"📊 {model_type} 参数分配详情")
    print("=" * 70)

    param_info = {
        'comp_param': ('📦', '组合模块'),
        'flow_param': ('🌊', '流匹配'),
        'hyper_param': ('🔮', '双曲模块'),
        'aim_param': ('🎯', 'CLIP+AIM Adapter'),
        'video_en_param': ('🎬', '视频编码器')
    }

    for key, (emoji, desc) in param_info.items():
        if len(param_groups[key]) > 0:
            print(f"\n{emoji} {key} ({desc}) - 数量: {len(param_groups[key])}")
            for name in param_names[key]:
                print(f"  - {name}")

    print("\n" + "=" * 70)
    print("📈 统计:")
    for key, (emoji, _) in param_info.items():
        print(f"  - {key}: {len(param_groups[key])}")
    print("=" * 70)

    optimizer_params = []
    
    if model.is_image:
        if param_groups['comp_param']:
            optimizer_params.append({
                'params': param_groups['comp_param'], 
                'lr': cfg.com_lr, 
                'weight_decay': cfg.com_wd
            })
        if param_groups['video_en_param']:
            optimizer_params.append({
                'params': param_groups['video_en_param'], 
                'lr': cfg.ve_lr, 
                'weight_decay': cfg.ve_wd
            })
    else:
        if is_clip:
            all_comp_param = param_groups['comp_param'] + param_groups['flow_param'] + param_groups['hyper_param']
            if all_comp_param:
                optimizer_params.append({
                    'params': all_comp_param, 
                    'lr': getattr(cfg, 'clip_com_lr', cfg.com_lr), 
                    'weight_decay': getattr(cfg, 'clip_com_wd', cfg.com_wd)
                })
            if param_groups['aim_param']:
                optimizer_params.append({
                    'params': param_groups['aim_param'], 
                    'lr': getattr(cfg, 'clip_aim_lr', cfg.ve_lr), 
                    'weight_decay': getattr(cfg, 'clip_aim_wd', cfg.ve_wd)
                })
            print(f"✅ 使用 CLIP 专用学习率: comp_lr={getattr(cfg, 'clip_com_lr', cfg.com_lr)}, aim_lr={getattr(cfg, 'clip_aim_lr', cfg.ve_lr)}")
        else:
            all_comp_param = param_groups['comp_param'] + param_groups['flow_param'] + param_groups['hyper_param']
            if all_comp_param:
                optimizer_params.append({
                    'params': all_comp_param, 
                    'lr': cfg.com_lr, 
                    'weight_decay': cfg.com_wd
                })
            if param_groups['video_en_param']:
                optimizer_params.append({
                    'params': param_groups['video_en_param'], 
                    'lr': cfg.ve_lr, 
                    'weight_decay': cfg.ve_wd
                })
            print(f"✅ 使用 Swin 通用学习率: com_lr={cfg.com_lr}, ve_lr={cfg.ve_lr}")

    if not optimizer_params:
        raise ValueError("No parameters found to optimize! Check your model setup.")
        
    optimizer = torch.optim.Adam(optimizer_params, eps=1e-8)
    return optimizer


def load_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    return model


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
