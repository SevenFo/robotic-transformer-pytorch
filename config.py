"""
训练配置管理
"""

import os
import yaml
from datetime import datetime


def load_config(config_path):
    """从YAML文件加载配置"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, save_path):
    """保存配置到YAML文件"""
    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def get_default_config():
    """获取默认配置"""
    return {
        "data_dir": "/data/shared_folder/h5_ur_1rgb/raw_dataset",
        "output_dir": "./outputs/run_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        "batch_size": 1,
        "num_workers": 32,
        "max_epochs": 100,
        "devices": [0, 1, 2, 3],
        "resume_from_checkpoint": None,
        "learning_rate": 3e-4,
        "weight_decay": 0.01,
        "warmup_steps": 1000,
        "seed": 42,
        "val_check_interval": 1000,  # 每1000步验证一次
        "save_checkpoint_steps": 5000,  # 每5000步保存一次检查点
        "val_sample_size": 500,  # 验证时使用的样本数量，设为None则使用全部
        "lr_scheduler_type": "onecycle",  # 学习率调度器类型：onecycle 或 cosine_warmup
        "early_stopping": True,  # 是否使用早停
        "early_stopping_patience": 50,  # 早停耐心值
    }
