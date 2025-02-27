"""
学习率调度工具，提供更多灵活的学习率调度选项
"""

import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    创建带有预热的余弦学习率调度

    Args:
        optimizer: 优化器
        num_warmup_steps: 预热步数
        num_training_steps: 总训练步数
        num_cycles: 在训练中完成的余弦周期数
        last_epoch: 上一轮的epoch数，用于恢复训练
    """

    def lr_lambda(current_step):
        # 预热阶段
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # 余弦退火阶段
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1,
):
    """
    创建带有预热的线性学习率调度

    Args:
        optimizer: 优化器
        num_warmup_steps: 预热步数
        num_training_steps: 总训练步数
        last_epoch: 上一轮的epoch数，用于恢复训练
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class CosineWarmupScheduler:
    """
    学习率调度器，结合预热和余弦退火
    可以直接在训练循环中使用，而不依赖于PyTorch Lightning
    """

    def __init__(self, optimizer, warmup_steps, max_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.current_step = 0

    def step(self):
        # 更新学习率
        if self.current_step < self.warmup_steps:
            # 预热阶段 - 线性增加
            lr_scale = min(1.0, float(self.current_step + 1) / float(self.warmup_steps))
        else:
            # 余弦退火阶段
            progress = float(self.current_step - self.warmup_steps) / float(
                max(1, self.max_steps - self.warmup_steps)
            )
            lr_scale = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = param_group["initial_lr"] * lr_scale

        self.current_step += 1

    def get_lr(self):
        """获取当前学习率"""
        return [param_group["lr"] for param_group in self.optimizer.param_groups]
