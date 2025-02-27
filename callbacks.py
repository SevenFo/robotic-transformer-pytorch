"""
自定义回调函数，增强训练过程监控和控制
"""

import os
import time
import torch
import numpy as np
from datetime import datetime
from lightning.pytorch.callbacks import Callback
from log import logger


class TrainingProgressCallback(Callback):
    """训练进度回调，用于记录和报告训练进度、估计剩余时间等"""

    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = save_dir
        self.start_time = None
        self.last_step_time = None
        self.step_times = []
        self.validation_times = []
        self.total_steps = 0
        self.current_step = 0

    def on_fit_start(self, trainer, pl_module):
        self.start_time = time.time()
        self.last_step_time = self.start_time
        self.total_steps = trainer.estimated_stepping_batches
        logger.info(f"开始训练，预计总步数: {self.total_steps}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        current_time = time.time()
        step_time = current_time - self.last_step_time
        self.last_step_time = current_time
        self.current_step = trainer.global_step

        # 记录步时间
        self.step_times.append(step_time)
        if len(self.step_times) > 100:  # 只保留最近的100个步骤时间
            self.step_times.pop(0)

        # 每100步记录一次进度
        if self.current_step % 100 == 0:
            elapsed = current_time - self.start_time
            avg_step_time = np.mean(self.step_times)
            remaining_steps = self.total_steps - self.current_step
            eta = remaining_steps * avg_step_time

            # 格式化时间
            elapsed_str = str(datetime.timedelta(seconds=int(elapsed)))
            eta_str = str(datetime.timedelta(seconds=int(eta)))

            logger.info(
                f"Step {self.current_step}/{self.total_steps} "
                f"({self.current_step / self.total_steps * 100:.1f}%) | "
                f"Elapsed: {elapsed_str} | ETA: {eta_str} | "
                f"Step time: {avg_step_time:.3f}s"
            )

    def on_validation_start(self, trainer, pl_module):
        self.validation_start_time = time.time()
        logger.info(f"开始验证 (Step {trainer.global_step})")

    def on_validation_end(self, trainer, pl_module):
        validation_time = time.time() - self.validation_start_time
        self.validation_times.append(validation_time)
        if len(self.validation_times) > 5:  # 只保留最近的5次验证时间
            self.validation_times.pop(0)

        avg_val_time = np.mean(self.validation_times)
        logger.info(
            f"验证完成，用时: {validation_time:.2f}s (平均: {avg_val_time:.2f}s)"
        )

        # 记录验证结果到单独文件，方便监控
        val_loss = trainer.callback_metrics.get("val/loss", float("nan"))
        with open(os.path.join(self.save_dir, "validation_log.csv"), "a") as f:
            if os.stat(os.path.join(self.save_dir, "validation_log.csv")).st_size == 0:
                f.write("step,timestamp,val_loss\n")  # 写入CSV头
            f.write(f"{trainer.global_step},{datetime.now().isoformat()},{val_loss}\n")

    def on_fit_end(self, trainer, pl_module):
        total_time = time.time() - self.start_time
        hours = total_time // 3600
        minutes = (total_time % 3600) // 60
        seconds = total_time % 60

        logger.info(
            f"训练结束! 总时间: {int(hours)}h {int(minutes)}m {int(seconds)}s "
            f"({total_time:.2f}s)"
        )


class EarlyStopping(Callback):
    """早停回调，如果指定指标在一定轮数内没有改善则停止训练"""

    def __init__(
        self,
        monitor="val/loss",
        min_delta=0.0,
        patience=5,
        mode="min",
        verbose=True,
        check_on_train_epoch_end=False,
    ):
        super().__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        self.check_on_train_epoch_end = check_on_train_epoch_end
        self.wait_count = 0
        self.stopped_epoch = 0
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.stop_training = False

    def on_validation_end(self, trainer, pl_module):
        if self.check_on_train_epoch_end:
            return
        self._check_metrics(trainer)

    def on_train_epoch_end(self, trainer, pl_module):
        if not self.check_on_train_epoch_end:
            return
        self._check_metrics(trainer)

    def _check_metrics(self, trainer):
        if self.monitor not in trainer.callback_metrics:
            return

        current = trainer.callback_metrics[self.monitor].item()

        if self.mode == "min":
            improved = current < (self.best_score - self.min_delta)
        else:
            improved = current > (self.best_score + self.min_delta)

        if improved:
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                self.stopped_epoch = trainer.current_epoch
                trainer.should_stop = True
                self.stop_training = True
                reason = (
                    f"Early stopping triggered: {self.monitor} didn't improve "
                    f"for {self.wait_count} evaluations. "
                    f"Best score: {self.best_score:.6f}"
                )
                if self.verbose:
                    logger.info(reason)

    def on_fit_end(self, trainer, pl_module):
        if self.stop_training and self.verbose:
            logger.info(f"Training stopped early at epoch {self.stopped_epoch}")
