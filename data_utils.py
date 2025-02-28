"""
数据处理工具，提供验证集采样、数据加载优化等功能
"""

import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
from typing import Callable, Optional, List, Any
from lightning.pytorch.callbacks import Callback


class SampledDataset(Dataset):
    """
    动态采样数据集包装器

    每次遍历完子集后自动重新采样，可用于验证集加速
    """

    def __init__(
        self,
        dataset: Dataset,
        sample_size: int,
        seed: Optional[int] = None,
        shuffle_after_epoch: bool = True,
    ):
        """
        初始化采样数据集

        Args:
            dataset: 原始数据集
            sample_size: 采样大小，必须小于或等于原始数据集大小
            seed: 随机种子，用于可复现性
            shuffle_after_epoch: 是否在每次完整遍历后重新采样
        """
        self.dataset = dataset
        self.sample_size = min(sample_size, len(dataset))
        self.random_gen = random.Random(seed) if seed is not None else random
        self.shuffle_after_epoch = shuffle_after_epoch
        self.full_indices = list(range(len(dataset)))
        self.sampled_indices = self._sample_indices()
        self.current_idx = 0

    def _sample_indices(self) -> List[int]:
        """生成新的采样索引"""
        return self.random_gen.sample(self.full_indices, self.sample_size)

    def __len__(self) -> int:
        """返回采样后的数据集大小"""
        return self.sample_size

    def __getitem__(self, idx: int) -> Any:
        """获取指定索引的样本"""

        # 检查是否需要重新采样
        if idx >= self.sample_size:
            raise IndexError(
                f"Index {idx} out of range for dataset with size {self.sample_size}"
            )

        # 获取当前样本
        actual_idx = self.sampled_indices[idx]

        self.current_idx += 1

        if self.current_idx == self.sample_size:
            # 遍历完一轮，重新采样
            self.current_idx = 0
            self.sampled_indices = self._sample_indices()

        return self.dataset[actual_idx]

    def resample(self) -> None:
        """手动触发重新采样"""
        self.sampled_indices = self._sample_indices()


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
    """
    train_dataset = dataset_fn(**train_kwargs)
    val_dataset = dataset_fn(**val_kwargs)

    train_loader = DataLoader(train_dataset, **train_kwargs.get("loader_args", {}))

    # 如果指定了采样大小，使用SampledDataset包装验证集
    if val_sample_size and val_sample_size < len(val_dataset):
        sampled_val_dataset = SampledDataset(
            dataset=val_dataset,
            sample_size=val_sample_size,
            seed=val_kwargs.get("seed"),
        )
        val_loader = DataLoader(
            sampled_val_dataset, **val_kwargs.get("loader_args", {})
        )
    else:
        val_loader = DataLoader(val_dataset, **val_kwargs.get("loader_args", {}))

    return train_loader, val_loader


# 可以移除原来的ValidationSampler类，因为它已经被SampledDataset替代
