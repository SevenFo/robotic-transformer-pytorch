"""
训练逻辑：
"""

import os
from typing import Optional, Dict, Union
from datetime import datetime
import lightning as lg
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from torchvision import transforms
from einops import rearrange

from robotic_transformer_pytorch import MaxViT, RT1
from dataset import create_dataloader
from log import logger, set_log_level
from config import load_config, get_default_config, save_config

# 引入我们新创建的模块
from callbacks import TrainingProgressCallback, EarlyStopping
from data_utils import set_seed, get_train_val_loaders
from lr_utils import get_cosine_schedule_with_warmup


class LitRT1(lg.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        lr_scheduler_type: str = "onecycle",  # 新增参数：学习率调度器类型
    ):
        super().__init__()
        self.save_hyperparameters()

        # 创建视觉主干网络
        vit = MaxViT(
            num_classes=1000,
            dim_conv_stem=64,
            dim=96,
            dim_head=32,
            depth=(2, 2, 5, 2),
            window_size=5,  # 7
            mbconv_expansion_rate=4,
            mbconv_shrinkage_rate=0.25,
            dropout=0.1,
            width=640,
            height=480,
        )

        # 创建 RT1 模型
        self.model = RT1(
            vit=vit,
            num_actions=13,  # 7 (joint) + 6 (end effector)
            depth=6,
            heads=8,
            dim_head=64,
            cond_drop_prob=0.2,
        )

        # 损失函数权重
        self.action_weights = torch.tensor(
            # 关节角度误差权重为1，末端执行器位姿误差权重为1
            [1.0] * 7 + [1.0] * 6,
            device=self.device,
        )

    def forward(self, video, texts, text_embeds):
        b, f, c, h, w = video.size()
        assert f == 6, f"Expected 6 frames, got {f}"
        assert c == 3, f"Expected 3 channels, got {c}"
        video = rearrange(video, "b f c h w -> b c f h w")
        return self.model(video, texts=texts, text_embeds=text_embeds)

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        loss, logs = self._shared_step(batch, "train")

        # 记录当前学习率
        opt = self.optimizers()
        current_lr = opt.param_groups[0]["lr"]
        logs["train/lr"] = current_lr

        self.log_dict(logs, prog_bar=True, sync_dist=False, logger=True)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int):
        loss, logs = self._shared_step(batch, "val")
        self.log_dict(logs, prog_bar=True, sync_dist=False, logger=True)
        return loss

    def _shared_step(self, batch: Dict, mode: str) -> tuple[torch.Tensor, Dict]:
        video = batch["video"]  # [B, T, C, H, W]
        action = batch["action"]  # [B, T, 13]
        texts = batch["language_raw"]  # List[str] of length B
        texts_use = batch["language_use"]  # [B, 512]
        texts_bert = batch["language_bert"]  # [B, 768]
        # 获取模型预测
        logits = self(
            video, texts=None, text_embeds=texts_bert
        )  # [B, T, 13, action_bins]
        action_target_token = self._normalize_action(
            action, num_bins=logits.size(-1)
        )  # [B, T, 13]

        loss_per_dim = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            action_target_token.view(-1),
            reduction="none",
        ).view_as(action_target_token)

        action_weights = self.action_weights.to(logits.device)

        # if loss.isnan():
        #     logger.error(f"Loss is NaN: {loss_per_dim=}, {action_weights=}")

        joint_loss = loss_per_dim[..., :7].mean()
        ee_loss = loss_per_dim[..., 7:].mean()
        # 计算加权总损失
        loss = joint_loss * 0.4 + ee_loss * 0.6

        # if joint_loss.isnan():
        #     logger.error(
        #         f"Joint loss is NaN: {loss_per_dim=}, {action_weights[...,:7]=}"
        #     )

        # if ee_loss.isnan():
        #     logger.error(f"EE loss is NaN: {loss_per_dim=}, {action_weights[...:,7:]=}")

        # 记录日志
        logs = {
            f"{mode}/loss": loss.item(),
            f"{mode}/joint_loss": joint_loss.item(),
            f"{mode}/ee_loss": ee_loss.item(),
        }

        return loss, logs

    def configure_optimizers(self):
        # 创建优化器
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # 根据配置选择学习率调度器
        if self.hparams.lr_scheduler_type == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.lr,
                total_steps=self.hparams.max_steps,
                pct_start=self.hparams.warmup_steps / self.hparams.max_steps,
                cycle_momentum=False,
            )
        elif self.hparams.lr_scheduler_type == "cosine_warmup":
            # 使用我们的自定义调度器
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.hparams.max_steps,
            )
        else:
            raise ValueError(
                f"未知的学习率调度器类型: {self.hparams.lr_scheduler_type}"
            )

        # 返回包含优化器和调度器的字典
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": self.hparams.lr_scheduler_type,
            },
        }

    # 添加一个新方法来记录当前学习率
    def on_before_optimizer_step(self, optimizer):
        # 记录所有参数组的学习率
        opt = self.optimizers()
        scheduler = self.lr_schedulers()

        if isinstance(scheduler, list):
            scheduler = scheduler[0]

        # 记录当前步数和对应的学习率
        lrs = {
            f"lr_group_{i}": param_group["lr"]
            for i, param_group in enumerate(opt.param_groups)
        }

        # 记录到日志
        self.log_dict(lrs, on_step=True, logger=True)

    @staticmethod
    def _normalize_action(x: torch.Tensor, num_bins: int) -> torch.Tensor:
        """将连续的动作值归一化到离散的bin中"""
        # action = np.concatenate([joint_position, end_effector], axis=1)

        def _norm(x: torch.Tensor, min: float, max: float):
            return ((x - min) / (max - min + 1e-6) * (num_bins - 1)).long()

        joints = x[..., :6]
        ee_joint = x[..., 6:7]
        ee_xyz = x[..., 7:10]
        ee_rpy = x[..., 10:]

        rad_min = torch.tensor(-torch.pi, device=x.device)
        rad_max = torch.tensor(torch.pi, device=x.device)

        ee_joint_min = torch.tensor(0.0, device=x.device)
        ee_joint_max = torch.tensor(1.0, device=x.device)

        ee_xyz_min = torch.tensor(-0.8, device=x.device)
        ee_xyz_max = torch.tensor(0.8, device=x.device)

        x_bins = torch.cat(
            [
                _norm(joints, rad_min, rad_max),
                _norm(ee_joint, ee_joint_min, ee_joint_max),
                _norm(ee_xyz, ee_xyz_min, ee_xyz_max),
                _norm(ee_rpy, rad_min, rad_max),
            ],
            dim=-1,
        )

        return torch.clamp(x_bins, 0, num_bins - 1)


def train(
    data_dir: str,
    output_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    max_epochs: int = 100,
    devices: list[int] | str | int = 1,
    resume_from_checkpoint: Optional[str] = None,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 1000,
    seed: int = 42,
    val_check_interval: int = 1000,  # 每1000步验证一次
    save_checkpoint_steps: int = 5000,  # 每5000步保存一次检查点
    val_sample_size: Optional[int] = 700,  # 验证时使用的样本数量
    lr_scheduler_type: str = "onecycle",  # 学习率调度器类型
    early_stopping: bool = True,  # 是否使用早停
    early_stopping_patience: int = 5,  # 早停耐心值
):
    version = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 设置输出目录
    os.makedirs(os.path.join(output_dir, version), exist_ok=True)

    # 设置随机种子
    set_seed(seed)

    # 数据预处理
    transform = transforms.Compose(
        [
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 创建数据加载器
    train_loader, val_loader = create_dataloader(
        data_dir=data_dir,
        split="merge",
        val_ratio=0.1,
        batch_size=batch_size,
        num_workers=num_workers,
        image_transform=transform,
        seed=seed,
        val_sample_size=val_sample_size,  # 添加val_sample_size参数传递到数据加载器
    )

    logger.info(
        f"成功加载数据集。训练集大小: {len(train_loader.dataset)}，"
        f"验证集大小: {len(val_loader.dataset)} "
        f"{'(已采样)' if val_sample_size and val_sample_size < len(val_loader.dataset.dataset) else ''}"
    )

    # 初始化回调列表
    callbacks = []

    # 创建模型
    model = LitRT1(
        lr=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        max_steps=len(train_loader) * max_epochs,
        lr_scheduler_type=lr_scheduler_type,
    )

    # 创建检查点回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, version, "checkpoints"),
        filename="rt1-{step:08d}-{val/loss:.4f}",
        save_top_k=10,
        monitor="val/loss",
        mode="min",
        save_last=True,
        every_n_train_steps=save_checkpoint_steps,
        save_on_train_epoch_end=False,
        auto_insert_metric_name=False,
        verbose=True,
    )

    # 添加学习率监控
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # 创建训练进度回调
    # progress_callback = TrainingProgressCallback(save_dir=output_dir)

    # 添加基础回调
    callbacks.extend(
        [
            checkpoint_callback,
            lr_monitor,
            # progress_callback,
        ]
    )

    # 如果启用早停，添加早停回调
    if early_stopping:
        early_stopping_callback = EarlyStopping(
            monitor="val/loss",
            min_delta=0.0001,
            patience=early_stopping_patience,
            verbose=True,
        )
        callbacks.append(early_stopping_callback)
        logger.info(f"启用早停，耐心值: {early_stopping_patience}")

    # 创建日志记录器
    tb_logger = TensorBoardLogger(
        output_dir, name="rt1", version=version, log_graph=True
    )

    # 创建训练器
    trainer = lg.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu" if devices == "cpu" else "gpu",
        devices=devices,
        callbacks=callbacks,
        logger=tb_logger,
        precision="16-mixed",
        strategy="ddp_find_unused_parameters_true",
        accumulate_grad_batches=8,
        log_every_n_steps=10,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=None,
    )

    # 保存当前配置
    config_to_save = {
        "data_dir": data_dir,
        "output_dir": output_dir,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "max_epochs": max_epochs,
        "devices": devices,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "seed": seed,
        "val_check_interval": val_check_interval,
        "save_checkpoint_steps": save_checkpoint_steps,
        "val_sample_size": val_sample_size,
        "lr_scheduler_type": lr_scheduler_type,
        "early_stopping": early_stopping,
        "early_stopping_patience": early_stopping_patience,
        "model_config": {
            "vit_depth": (2, 2, 5, 2),
            "window_size": 5,
            "width": 640,
            "height": 480,
            "transformer_depth": 6,
            "heads": 8,
        },
    }
    save_config(config_to_save, os.path.join(output_dir, version, "config.yaml"))

    # 开始训练
    trainer.fit(
        model,
        train_loader,
        val_loader,
        ckpt_path=resume_from_checkpoint,
    )

    # 记录训练结果
    logger.info(f"Best model path: {checkpoint_callback.best_model_path}")
    logger.info(f"Best model score: {checkpoint_callback.best_model_score}")

    # 保存最佳模型路径
    with open(os.path.join(output_dir, "best_model_path.txt"), "w") as f:
        f.write(checkpoint_callback.best_model_path)

    # 训练完成后，显式打印最终学习率
    final_lr = model.optimizers().param_groups[0]["lr"]
    logger.info(f"Final learning rate: {final_lr:.8f}")


if __name__ == "__main__":
    """
    480,640,ws=5
    672,448,ws=7
    768,576,ws=6
    """
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    set_log_level("INFO")
    torch.set_float32_matmul_precision("medium")

    # 获取默认配置
    config = get_default_config()

    # 可以在这里修改特定配置
    config.update(
        {
            "early_stopping": True,
            "early_stopping_patience": 100,
            "val_check_interval": 1000,
            "val_sample_size": 700,
            "save_checkpoint_steps": 10000,
        }
    )

    logger.info(f"Using config: {config}")

    # 启动训练
    train(**config)
