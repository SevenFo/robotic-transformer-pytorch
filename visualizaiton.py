from typing import Optional, List, Tuple, Union
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path
import datetime

from log import logger


def visualize_samples(
    data_source: Union[Dataset, DataLoader],
    num_samples: int = 3,
    samples_per_row: int = 1,
    max_frames: int = 6,
    output_dir: Optional[str] = None,
    fig_size: Tuple[int, int] = (15, 15),  # 增加高度以适应多行子图
    dpi: int = 100,
) -> List[Figure]:
    """
    可视化数据集或数据加载器中的样本，包括图像序列和动作数据

    参数:
    - data_source: Dataset或DataLoader实例
    - num_samples: 要可视化的样本数量
    - samples_per_row: 每行显示的样本数(对于多样本图)
    - max_frames: 每个样本最多显示的帧数
    - output_dir: 输出目录，如果为None则尝试在notebook中显示
    - fig_size: 图像大小
    - dpi: 图像分辨率

    返回:
    - 生成的matplotlib图像列表
    """
    # 确定是Dataset还是DataLoader
    if isinstance(data_source, DataLoader):
        # 从DataLoader获取样本
        batch = next(iter(data_source))
        # 限制样本数量
        num_samples = min(num_samples, batch["video"].shape[0])
        samples = [{k: v[i] for k, v in batch.items()} for i in range(num_samples)]
    else:
        # Dataset - 直接索引
        indices = torch.randperm(len(data_source))[:num_samples].tolist()
        samples = [data_source[i] for i in indices]

    # 判断是否在notebook环境中
    in_notebook = False
    try:
        from IPython import get_ipython

        if get_ipython() is not None:
            in_notebook = True
    except ImportError:
        pass

    # 创建输出目录
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    figures = []
    for i, sample in enumerate(samples):
        video = sample["video"]  # 预期形状: [frames, channels, height, width]
        action = sample[
            "action"
        ]  # shape: [frames, 13] (7 joint positions + 6 end effector)

        num_frames = min(video.shape[0], max_frames)
        frames_per_row = min(5, num_frames)  # 每行最多显示5帧

        num_video_rows = (num_frames + frames_per_row - 1) // frames_per_row  # 向上取整

        # 计算action维度和所需的行数
        action_dims = action.shape[1]  # 13维
        num_action_rows = (
            action_dims + frames_per_row - 1
        ) // frames_per_row  # 向上取整

        # 创建子图网格：1行用于图像 + N行用于action
        total_rows = num_video_rows + num_action_rows
        fig = plt.figure(figsize=(fig_size[0], fig_size[1] * total_rows / 2))

        # 1. 显示视频帧
        for frame_idx in range(num_frames):
            ax = plt.subplot(total_rows, frames_per_row, frame_idx + 1)
            frame = video[frame_idx]
            if isinstance(frame, torch.Tensor):
                frame = frame.permute(1, 2, 0).cpu().numpy()
                if frame.max() > 1.0:
                    frame = frame / 255.0
            ax.imshow(frame)
            ax.set_title(f"Frame {frame_idx}")
            ax.axis("off")

        # 2. 显示action数据
        action_names = (
            [f"Joint {i + 1}" for i in range(7)]  # 7个关节
            + ["EE x", "EE y", "EE z", "EE rx", "EE ry", "EE rz"]  # 末端执行器6自由度
        )

        action_dim_offset = num_video_rows * frames_per_row + 1
        for dim in range(action_dims):
            dim_ = dim + action_dim_offset
            row = dim_ // frames_per_row
            col = dim_ % frames_per_row
            plot_idx = row * frames_per_row + col

            ax = plt.subplot(total_rows, frames_per_row, plot_idx)
            # 绘制该维度随时间的变化
            time_steps = range(num_frames)
            ax.plot(time_steps, action[:num_frames, dim], "b-", linewidth=2)
            ax.set_title(action_names[dim])
            ax.grid(True)
            ax.set_xlabel("Frame")

            # 设置y轴范围，使其对称（如果数据接近0）
            y_max = max(
                abs(action[:num_frames, dim].max()), abs(action[:num_frames, dim].min())
            )
            if y_max < 1e-3:  # 如果数据非常接近0
                ax.set_ylim(-0.1, 0.1)
            else:
                ax.set_ylim(-y_max * 1.1, y_max * 1.1)

        # 显示文本信息
        if "language_raw" in sample and sample["language_raw"] is not None:
            text = sample["language_raw"]
            fig.suptitle(f"Sample {i} - Text: {text}", fontsize=12, y=0.99)
        else:
            fig.suptitle(f"Sample {i}", fontsize=12, y=0.99)

        fig.tight_layout()
        figures.append(fig)

        # 保存或显示xs
        if output_dir is not None:
            save_path = (
                Path(output_dir)
                / f"sample_{i}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            fig.savefig(save_path)
            logger.info(f"Saved visualization to {save_path}")
        elif in_notebook:
            plt.show()
        else:
            # 服务器环境但未指定输出目录，默认保存到当前目录的vis_output
            default_output = Path("./vis_output")
            default_output.mkdir(parents=True, exist_ok=True)
            save_path = default_output / f"sample_{i}_{timestamp}.png"
            fig.savefig(save_path)
            logger.info(f"Saved visualization to {save_path}")

    # 关闭图形以释放内存（如果不在notebook中）
    if not in_notebook:
        for fig in figures:
            plt.close(fig)

    return figures


def visualize_batch(
    dataloader: DataLoader, output_dir: Optional[str] = None, **kwargs
) -> List[Figure]:
    """
    可视化数据加载器中的一个批次

    参数:
    - dataloader: DataLoader实例
    - output_dir: 输出目录
    - **kwargs: 传递给visualize_samples的其他参数

    返回:
    - 生成的matplotlib图像列表
    """
    return visualize_samples(dataloader, output_dir=output_dir, **kwargs)
