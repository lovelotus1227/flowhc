import argparse
from omegaconf import OmegaConf
from utils.main_utils import *
from utils.lr_scheduler import WarmupCosineLR
from modules.trainer import Trainer


def main(opt):
    cfg = OmegaConf.load(opt.config)

    train_dataset = instantiate(cfg.data, phase="train")
    val_dataset = instantiate(cfg.data, phase="val")
    test_dataset = instantiate(cfg.data, phase="test")
    
    model = instantiate(cfg.model, dset=train_dataset)
    
    if cfg.load_from is not None:
        model = load_checkpoint(model, cfg.load_from)

    optimizer = get_optimizer_vm(cfg.train, model)

    lr_scheduler = WarmupCosineLR(optimizer=optimizer, milestones=[cfg.train.warmup, cfg.train.epochs],
                                  warmup_iters=cfg.train.warmup, min_ratio=1e-8)
    
    trainer = Trainer(cfg, model, optimizer, lr_scheduler, train_dataset, val_dataset, test_dataset)

    if opt.mode == "train":
        trainer.train()
    elif opt.mode == "eval":
        trainer.evaluate(test_dataset)
    else:
        raise ValueError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/base.yaml")
    parser.add_argument("--mode", type=str, default="train")

    opt = parser.parse_args()

    main(opt)