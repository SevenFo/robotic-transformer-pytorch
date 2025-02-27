"""
数据处理工具，提供验证集采样、数据加载优化等功能
"""

import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
from typing import Callable, Optional
from lightning.pytorch.callbacks import Callback


class ValidationSampler(Callback):
    """
    验证集采样回调，每次验证时对验证集进行采样

    这可以加速验证过程，同时通过每次使用不同样本提供更全面的评估
    """

    def __init__(
        self,
        val_dataset: Dataset,
        batch_size: int,
        sample_size: int,
        num_workers: int = 4,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.sample_size = min(sample_size, len(val_dataset))
        self.num_workers = num_workers
        self.seed = seed
        self.random_gen = random.Random(seed) if seed is not None else random
        self.initial_loader = self._create_loader()

    def _create_loader(self):
        """创建采样后的验证数据加载器"""
        indices = self.random_gen.sample(range(len(self.val_dataset)), self.sample_size)
        sampled_dataset = Subset(self.val_dataset, indices)
        return DataLoader(
            sampled_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def on_validation_epoch_start(self, trainer, pl_module):
        """每次验证开始时重新采样验证集"""
        trainer.val_dataloaders = self._create_loader()

    def on_validation_epoch_end(self, trainer, pl_module):
        """验证结束后记录采样信息"""
        trainer.logger.experiment.add_scalar(
            "validation/sample_size", self.sample_size, trainer.global_step
        )


def set_seed(seed: int):
    """设置随机种子，确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_train_val_loaders(dataset_fn, train_kwargs, val_kwargs, val_sample_size=None):
    """
    创建训练和验证数据加载器

    Args:
        dataset_fn: 创建数据集的函数
        train_kwargs: 训练数据加载器参数
        val_kwargs: 验证数据加载器参数
        val_sample_size: 验证样本数量，如果为None则使用全部验证集

    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        val_sampler: 验证采样器（如果适用）
    """
    train_dataset = dataset_fn(**train_kwargs)
    val_dataset = dataset_fn(**val_kwargs)

    train_loader = DataLoader(train_dataset, **train_kwargs.get("loader_args", {}))

    val_sampler = None
    if val_sample_size and val_sample_size < len(val_dataset):
        # 创建验证采样器
        val_sampler = ValidationSampler(
            val_dataset=val_dataset,
            batch_size=val_kwargs.get("loader_args", {}).get("batch_size", 32),
            sample_size=val_sample_size,
            num_workers=val_kwargs.get("loader_args", {}).get("num_workers", 4),
            seed=val_kwargs.get("seed"),
        )
        val_loader = val_sampler.initial_loader
    else:
        val_loader = DataLoader(val_dataset, **val_kwargs.get("loader_args", {}))

    return train_loader, val_loader, val_sampler
