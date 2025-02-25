"""
训练逻辑：
"""

import os
from typing import Optional, Dict, Union

import lightning as lg
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torchvision import transforms
from einops import rearrange

from robotic_transformer_pytorch import MaxViT, RT1
from dataset import create_dataloader
from log import logger, set_log_level


class LitRT1(lg.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
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
            # 关节角度误差权重为1，末端执行器位姿误差权重为2
            [1.0] * 7 + [2.0] * 6,
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
        self.log_dict(logs, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict, batch_idx: int):
        loss, logs = self._shared_step(batch, "val")
        self.log_dict(logs, prog_bar=True)
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

        # 计算每个动作维度的交叉熵损失
        losses = []
        for i in range(action.size(-1)):
            # 将连续action归一化到[0, action_bins-1]范围内
            action_target = self._normalize_action(
                action[..., i], num_bins=logits.size(-1)
            )
            loss_i = F.cross_entropy(
                logits[..., i, :].reshape(-1, logits.size(-1)),
                action_target.reshape(-1),
            )
            losses.append(loss_i * self.action_weights[i])

        # 计算加权总损失
        loss = torch.stack(losses).mean()

        # 记录日志
        logs = {
            f"{mode}/loss": loss.item(),
            f"{mode}/joint_loss": torch.stack(losses[:7]).mean().item(),
            f"{mode}/ee_loss": torch.stack(losses[7:]).mean().item(),
        }

        return loss, logs

    def configure_optimizers(self):
        # 创建优化器
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # 创建学习率调度器
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=self.hparams.max_steps,
            pct_start=self.hparams.warmup_steps / self.hparams.max_steps,
            cycle_momentum=False,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    @staticmethod
    def _normalize_action(x: torch.Tensor, num_bins: int) -> torch.Tensor:
        """将连续的动作值归一化到离散的bin中"""
        x_min, x_max = x.min(), x.max()
        x_norm = (x - x_min) / (x_max - x_min + 1e-6)
        x_bin = (x_norm * (num_bins - 1)).long()
        return torch.clamp(x_bin, 0, num_bins - 1)


def train(
    data_dir: str,
    output_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    max_epochs: int = 100,
    devices: list[int] | str | int = 1,
    resume_from_checkpoint: Optional[str] = None,
):
    # 设置输出目录
    os.makedirs(output_dir, exist_ok=True)

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
        batch_size=batch_size,
        num_workers=num_workers,
        image_transform=transform,
    )

    # 创建模型
    model = LitRT1(
        lr=3e-4,
        weight_decay=0.01,
        warmup_steps=1000,
        max_steps=len(train_loader) * max_epochs,
    )

    # 创建回调
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename="rt1-{epoch:02d}-{val/loss:.2f}",
            save_top_k=3,
            monitor="val/loss",
            mode="min",
        ),
    ]

    # 创建日志记录器
    logger = TensorBoardLogger(output_dir, name="rt1")

    # 创建训练器
    trainer = lg.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu" if devices == "cpu" else "gpu",
        devices=devices,
        callbacks=callbacks,
        logger=logger,
        precision="16-mixed",  # 使用混合精度训练
    )

    # 开始训练
    trainer.fit(
        model,
        train_loader,
        val_loader,
        ckpt_path=resume_from_checkpoint,
    )


if __name__ == "__main__":
    """
    480,640,ws=5
    672,448,ws=7
    768,576,ws=6
    """
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    set_log_level("INFO")
    torch.set_float32_matmul_precision("medium")
    # 训练配置
    config = {
        "data_dir": "/data/shared_folder/h5_ur_1rgb/raw_dataset",
        "output_dir": "./outputs",
        "batch_size": 1,
        "num_workers": 4,
        "max_epochs": 100,
        "devices": 1,
        "resume_from_checkpoint": None,  # 或者指定检查点路径
    }

    train(**config)
