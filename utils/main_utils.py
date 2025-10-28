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
    comp_param=[]
    video_en_param=[]
    for name, param in model.named_parameters():
        if 'video_encoder' in name:
            video_en_param.append(param)
        else:
            comp_param.append(param)
    optimizer = torch.optim.Adam([
        {'params': comp_param, 'lr': cfg.com_lr,'weight_decay': cfg.com_wd},
        {'params': video_en_param, 'lr': cfg.ve_lr,'weight_decay': cfg.ve_wd}],
        lr=cfg.ve_lr, eps=1e-8, weight_decay=cfg.ve_wd)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

    return optimizer


def load_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    return model


def set_seed(seed):
    """function sets the seed value
    Args:
        seed (int): seed value
    """
    seed = int(seed)
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)