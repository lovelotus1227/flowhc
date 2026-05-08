import os
import torch
import torch.nn as nn
from datetime import timedelta
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, DataLoaderConfiguration
from utils.main_utils import set_seed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer:
    def __init__(
            self,
            cfg,
            model,
            optimizer,
            lr_scheduler,
            train_dataset,
            val_dataset,
            test_dataset,
    ):
        # 适配分布式训练场景
        self.accelerator = Accelerator(
            mixed_precision="fp16",
            step_scheduler_with_optimizer=False,
            dataloader_config=DataLoaderConfiguration(split_batches=True),
            kwargs_handlers=[
                DistributedDataParallelKwargs(find_unused_parameters=True),
                InitProcessGroupKwargs(timeout=timedelta(hours=5)),
            ],
        )

        self.cfg = cfg
        self.device = self.accelerator.device

        self.attr2idx = train_dataset.attr2idx
        self.obj2idx = train_dataset.obj2idx
        self.train_pairs = torch.tensor([(self.attr2idx[attr], self.obj2idx[obj])
                                         for attr, obj in train_dataset.train_pairs]).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()

        if cfg.seed:
            rank_idx = self.accelerator.local_process_index
            set_seed(cfg.seed + rank_idx)

        # 定义训练数据加载器
        train_dataloader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True,
                                      drop_last=False, num_workers=4, pin_memory=True)

        # 自动将模型、优化器、调度器、数据加载器适配到当前训练环境
        self.model, self.optimizer, self.lr_scheduler, self.train_dataloader = self.accelerator.prepare(
            model, optimizer, lr_scheduler, train_dataloader
        )

        if self.accelerator.is_main_process:
            os.makedirs(cfg.results_path, exist_ok=True)

            log_path = os.path.join(cfg.results_path, "logs")
            os.makedirs(log_path, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_path)

            self.ckpt_path = os.path.join(cfg.results_path, "ckpt")
            os.makedirs(self.ckpt_path, exist_ok=True)
            OmegaConf.save(cfg, os.path.join(self.ckpt_path, "config.yaml"))

            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
            self.test_dataset = test_dataset

    def save_checkpoint(self, epoch):
        data = {
            "state_dict": self.accelerator.get_state_dict(self.model),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
        }
        torch.save(data, os.path.join(self.ckpt_path, f"epoch-{epoch}.pt"))

    def train(self):
        cfg = self.cfg
        accelerator = self.accelerator

        for epoch in range(0, cfg.train.epochs):
            pbar = tqdm(self.train_dataloader, dynamic_ncols=True, desc=f"Epoch {epoch}")

            for step, batch in enumerate(pbar):
                self.model.train()
                batch_verb = batch[1].to(self.device)
                batch_obj = batch[2].to(self.device)
                batch_target = batch[3].to(self.device)
                batch_img = batch[0].to(self.device)
                # print("11111111111111111111111111111111")
                # print(batch_verb.shape)
                # print(batch_obj.shape)
                # print(batch_target.shape)
                # print(batch_img.shape)
                # 64,3,224,224
                # 64 8 3 224 224
                
                with self.accelerator.autocast():
                    # 检查模型返回
                    model_output = self.model(
                        batch_img,
                        verb_labels=batch_verb,
                        obj_labels=batch_obj,
                    )
                   
                    p_v, p_o, p_pred, additional_loss = model_output

                    # component loss
                    loss_verb = self.loss_fn(p_v * cfg.train.cosine_scale, batch_verb)
                    loss_obj = self.loss_fn(p_o * cfg.train.cosine_scale, batch_obj)
                    train_v_inds, train_o_inds = self.train_pairs[:, 0], self.train_pairs[:, 1]
                    pred_com_train = p_pred[:, train_v_inds, train_o_inds]
                    loss_com = self.loss_fn(pred_com_train * cfg.train.cosine_scale, batch_target)

                    # 总损失：原始损失 + 额外损失（additional_loss 已经包含了所有权重）
                    loss = loss_com + 0.2 * (loss_verb + loss_obj) + additional_loss
                    accelerator.backward(loss)
                    # ====================== 真实梯度检查 ======================
                    # if step == 0:  # 只打印第一步，避免刷屏
                    #     print("\n===== 真实梯度是否存在 =====")
                    #     for name, param in self.model.named_parameters():
                    #         if param.requires_grad:
                    #             print(name, " | ", param.grad is not None)
                    # ==========================================================
                pbar.set_postfix_str(f"loss: {loss.item():.4f}")

                if accelerator.is_main_process:
                    self.writer.add_scalar("train/loss", loss.item(), epoch * len(self.train_dataloader) + step)
                    self.writer.add_scalar("train/loss_verb", loss_verb.item(),
                                           epoch * len(self.train_dataloader) + step)
                    self.writer.add_scalar("train/loss_obj", loss_obj.item(), epoch * len(self.train_dataloader) + step)
                    self.writer.add_scalar("train/loss_comp", loss_com.item(),
                                           epoch * len(self.train_dataloader) + step)
                    self.writer.add_scalar("train/loss_additional",
                                           additional_loss.item() if isinstance(additional_loss,
                                                                                torch.Tensor) else additional_loss,
                                           epoch * len(self.train_dataloader) + step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"],
                                           epoch * len(self.train_dataloader) + step)

                accelerator.wait_for_everyone()

                self.optimizer.step()
                self.optimizer.zero_grad()

            self.lr_scheduler.step()

            if accelerator.is_main_process:
                if (epoch + 1) % cfg.train.eval_freq == 0 or (epoch + 1) >= cfg.train.eval_ts:
                    # loss_avg, val_results = self.evaluate(self.train_dataset)
                    # for k, v in val_results.items():
                    #     self.writer.add_scalar(f"val/{k}", v, epoch)
                    loss_avg, val_results = self.evaluate(self.val_dataset)
                    for k, v in val_results.items():
                        self.writer.add_scalar(f"val/{k}", v, epoch)
                    self.save_checkpoint(epoch)

                """
                TODO: run evaluation on the test split when a new best score is achieved 
                      on the validation split
                """
                if (epoch + 1) == cfg.train.epochs:
                    print("Test dataset evaluation:")
                    loss_avg, test_results = self.evaluate(self.test_dataset)
                    for k, v in test_results.items():
                        self.writer.add_scalar(f"test/{k}", v, epoch)

            accelerator.wait_for_everyone()

    def evaluate(self, dataset):
        import modules.evaluator as test

        model = self.accelerator.unwrap_model(self.model).eval()
        evaluator = test.Evaluator(dataset, model=None)

        all_logits, all_attr_gt, all_obj_gt, all_pair_gt, loss_avg = test.predict_logits(model, dataset, self.cfg)
        test_stats = test.test(
            dataset,
            evaluator,
            all_logits,
            all_attr_gt,
            all_obj_gt,
            all_pair_gt,
            self.cfg,
        )

        result = ""
        # if model.is_image:
        #     key_set = ["attr_acc", "obj_acc"]
        # else:
        key_set = ["attr_acc", "obj_acc", "best_seen", "best_unseen", "best_hm" , "AUC"]

        for key in key_set:
            result = result + key + "  " + str(round(test_stats[key], 4)) + "| "
        print(result)

        return loss_avg, test_stats
