from typing import Optional, Callable, Dict, List, Tuple, Literal
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import h5py
import os
import numpy as np
import torchvision
from log import logger


def list_all_keys(h5file, path="/"):
    """
    递归列出HDF5文件中所有组和数据集的键。

    参数:
    - h5file: h5py.File 对象
    - path: 当前遍历的路径，默认为根目录 '/'
    """
    items = []

    def visitor_func(name):
        items.append(name)

    h5file[path].visit(visitor_func)
    return items


class RobotMindDataset(Dataset):
    def __init__(
        self,
        dir: str,
        split: Literal["train", "val", "merge"] = "merge",
        window_size: int = 6,
        stride: int = 1,
        image_transform: Optional[Callable] = None,
        text_transform: Optional[Callable] = None,
        file_cache_size: int = 50,
    ):
        """
        参数说明：
        - split:
            - "train"：仅加载原始 train 目录的数据
            - "val"：仅加载原始 val 目录的数据
            - "merge"：合并所有数据（默认）
        - image_transform: 图像数据的转换函数
            - 输入：(n_frames, C, H, W)
            - 输出：(n_frames, C, H, W)
        """
        self.dir: str = dir
        self.split: str = split
        self.window_size: int = window_size
        self.stride: int = stride
        self.image_transform: Optional[Callable] = image_transform
        self.text_transform: Optional[Callable] = text_transform

        # 存储文件元数据和全局索引
        self.file_index: List[Dict] = []
        self.index_map: List[Tuple[int, int]] = []  # (file_idx, window_start)

        # 文件句柄缓存
        self.file_cache: Dict[str, h5py.File] = {}
        self.cache_size = file_cache_size

        # 遍历目录收集数据
        for dirpath, _, filenames in os.walk(dir):
            for fname in filenames:
                if not fname.endswith(".hdf5"):
                    continue

                # 解析路径结构：.../task_name/traj_type/split/episode_id/data/file.h5
                try:
                    *_, task_name, traj_type, split_part, episode_id, _ = dirpath.split(
                        "/"
                    )
                except ValueError:
                    raise RuntimeError(f"Invalid directory structure: {dirpath}")
                if split_part not in ["train", "val"]:
                    logger.debug(
                        f"Skipping {dirpath}: invalid split part '{split_part}', maybe not in split folders"
                    )
                    continue

                # 根据 split 参数过滤数据
                if self.split in ["train", "val"] and split_part != self.split:
                    continue  # 跳过不符合的目录

                file_meta = {
                    "path": os.path.join(dirpath, fname),
                    "task": task_name,
                    "traj_type": traj_type,
                    "split": split_part,
                    "episode_id": episode_id,
                }
                self._add_file_to_index(file_meta)

    def _add_file_to_index(self, file_meta: Dict):
        """将单个文件添加到索引系统"""
        file_path = file_meta["path"]

        try:
            with h5py.File(file_path, "r") as f:
                """
                [
                    ('language_distilbert', <HDF5 dataset "language_distilbert": shape (1, 1, 768), type "<f2">), 
                    ('language_raw', <HDF5 dataset "language_raw": shape (1,), type "|O">), 
                    ('language_use', <HDF5 dataset "language_use": shape (1, 512), type "<f4">), 
                    ('master', <HDF5 group "/master" (1 members)>), 
                    ('observations', <HDF5 group "/observations" (2 members)>), 
                    ('puppet', <HDF5 group "/puppet" (2 members)>)
                ]
                """
                try:
                    traj_length = len(f["/observations/rgb_images/camera_top"])
                except KeyError:
                    logger.warning(
                        f"Skipping {file_path}: missing /observations/rgb_images/camera_top \n Keys: {list_all_keys(f)}"
                    )
        except Exception as e:
            logger.error(f"Error loading {file_path}, {type(e).__name__}: {e}")
            return

        # 计算有效窗口数量
        assert self.stride == 1, "Only stride=1 is supported for now"
        num_windows = (traj_length - self.window_size) // self.stride + 1
        if num_windows < 1:
            logger.warning(
                f"Skipping short trajectory: {file_path} (length={traj_length})"
            )
            return

        # 记录全局索引
        file_idx = len(self.file_index)
        for window_start in range(0, num_windows * self.stride, self.stride):
            self.index_map.append((file_idx, window_start))

        self.file_index.append(file_meta)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx: int):
        file_idx, window_start = self.index_map[idx]
        file_meta = self.file_index[file_idx]
        file_path = file_meta["path"]

        # 文件缓存管理
        if file_path not in self.file_cache:
            if len(self.file_cache) >= self.cache_size:
                # 简单LRU策略：移除第一个元素
                oldest_key = next(iter(self.file_cache))
                self.file_cache[oldest_key].close()
                del self.file_cache[oldest_key]

            self.file_cache[file_path] = h5py.File(file_path, "r")

        f = self.file_cache[file_path]

        # 读取数据窗口
        window_end = window_start + self.window_size
        camera_top_rgb_raw = f["/observations/rgb_images/camera_top"][
            window_start:window_end
        ]
        imgs = []
        for raw in camera_top_rgb_raw:
            try:
                # 3x480x640, uint8
                img = torchvision.io.decode_image(torch.from_numpy(raw))
                imgs.append(img)
            except Exception as e:
                logger.error(
                    f"Error decoding image of {file_path} {type(e).__name__}: {e}"
                )
                raise IndexError(
                    f"Invalid sample at index {idx}: Error decoding image of {file_path}"
                )
        video = torch.stack(imgs)
        # n_win, 6
        end_effector = f["/puppet/end_effector"][window_start:window_end]
        # n_win, 7
        joint_position = f["/puppet/joint_position"][window_start:window_end]
        action = np.concatenate([joint_position, end_effector], axis=1)
        # 1, 1, 768
        language_token = np.array(f["/language_distilbert"][0, 0])
        # 1, 512
        language_use = np.array(f["/language_use"][0])
        # 1
        language_raw = f["/language_raw"][0].decode("utf-8")
        # n_win, 7
        master_joint_position = f["/master/joint_position"][window_start:window_end]
        # 应用变换
        if self.image_transform:
            video = self.image_transform(video)

        return {
            "video": video,
            "action": action,
            "language_bert": language_token,
            "language_use": language_use,
            "language_raw": language_raw,
        }

    def __del__(self):
        for f in self.file_cache.values():
            f.close()


def create_dataloader(
    data_dir: str,
    split: Literal["train", "val", "merge"] = "merge",
    val_ratio: float = 0.2,
    batch_size: int = 32,
    num_workers: int = 4,
    **dataset_kwargs,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    创建DataLoader的智能函数

    返回：
    - 当 split="merge" 时：返回 (train_loader, val_loader)
    - 当 split="train"/"val" 时：返回 (loader, None)
    """
    # 初始化数据集
    dataset = RobotMindDataset(dir=data_dir, split=split, **dataset_kwargs)

    if split == "merge":
        # 动态划分逻辑 - 使用PyTorch的random_split
        # 按episode_id分组
        episode_indices = {}

        # 按照episode_id将数据点分组
        for i, (file_idx, _) in enumerate(dataset.index_map):
            episode_id = dataset.file_index[file_idx]["episode_id"]
            if episode_id not in episode_indices:
                episode_indices[episode_id] = []
            episode_indices[episode_id].append(i)

        # 随机分割episode groups
        episodes = list(episode_indices.keys())
        # 设置随机种子以确保可重复性
        generator = torch.Generator().manual_seed(42)

        # 计算训练集和验证集的episode数量
        num_val = int(len(episodes) * val_ratio)
        num_train = len(episodes) - num_val

        # 随机分割episode
        train_episodes, val_episodes = random_split(
            episodes, [num_train, num_val], generator=generator
        )

        # 获取对应的索引
        train_indices = []
        for ep in train_episodes:
            train_indices.extend(episode_indices[ep])

        val_indices = []
        for ep in val_episodes:
            val_indices.extend(episode_indices[ep])

        # 创建数据加载器
        train_loader = DataLoader(
            Subset(dataset, train_indices),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            Subset(dataset, val_indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        return train_loader, val_loader
    else:
        # 直接使用原始划分
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),  # 仅训练集shuffle
            num_workers=num_workers,
        )
        return loader, None


# 使用示例
if __name__ == "__main__":
    from log import set_log_level
    from visualizaiton import visualize_batch

    set_log_level("INFO")

    # 设置数据集路径
    data_path = "/data/shared_folder/h5_ur_1rgb/raw_dataset"

    # 创建可视化输出目录
    vis_dir = "./visualizations"
    os.makedirs(vis_dir, exist_ok=True)

    logger.info(f"测试数据集: {data_path}")

    # # 测试 1: 加载并可视化单个数据集样本
    # try:
    #     logger.info("测试 1: 初始化数据集并可视化样本")
    #     dataset = RobotMindDataset(dir=data_path, split="merge", window_size=10)
    #     logger.info(f"数据集大小: {len(dataset)} 样本")

    #     # 可视化几个随机样本
    #     visualize_samples(
    #         dataset,
    #         num_samples=3,
    #         max_frames=6,
    #         output_dir=os.path.join(vis_dir, "dataset_samples"),
    #         fig_size=(18, 6),
    #         dpi=120,
    #     )
    #     logger.info("成功可视化数据集样本")
    # except Exception as e:
    #     logger.error(f"加载数据集失败: {type(e).__name__}: {e}")

    # # 测试 2: 使用 merge 模式创建训练和验证加载器
    # try:
    #     logger.info("测试 2: 使用 merge 模式创建数据加载器")
    #     train_loader, val_loader = create_dataloader(
    #         data_dir=data_path,
    #         split="merge",
    #         window_size=10,
    #         val_ratio=0.2,
    #         batch_size=8,
    #         num_workers=4,
    #     )

    #     logger.info(
    #         f"训练集大小: {len(train_loader.dataset)} 样本, {len(train_loader)} 批次"
    #     )
    #     logger.info(
    #         f"验证集大小: {len(val_loader.dataset)} 样本, {len(val_loader)} 批次"
    #     )

    #     # 可视化训练批次
    #     visualize_batch(
    #         train_loader,
    #         num_samples=2,
    #         output_dir=os.path.join(vis_dir, "train_batch"),
    #         fig_size=(18, 6),
    #     )
    #     logger.info("成功可视化训练批次")
    # except Exception as e:
    #     logger.error(f"创建合并数据加载器失败: {type(e).__name__}: {e}")

    # # 测试 3: 分别加载训练集和验证集
    # try:
    #     logger.info("测试 3: 分别加载训练集和验证集")
    #     # 训练集
    #     train_loader_only, _ = create_dataloader(
    #         data_dir=data_path,
    #         split="train",
    #         window_size=10,
    #         batch_size=8,
    #         num_workers=4,
    #     )

    #     # 验证集
    #     val_loader_only, _ = create_dataloader(
    #         data_dir=data_path, split="val", window_size=10, batch_size=8, num_workers=4
    #     )

    #     if train_loader_only is not None:
    #         logger.info(f"训练集大小: {len(train_loader_only.dataset)} 样本")
    #         # 可视化训练样本
    #         visualize_batch(
    #             train_loader_only,
    #             num_samples=2,
    #             output_dir=os.path.join(vis_dir, "train_only"),
    #             fig_size=(18, 6),
    #         )

    #     if val_loader_only is not None:
    #         logger.info(f"验证集大小: {len(val_loader_only.dataset)} 样本")
    #         # 可视化验证样本
    #         visualize_batch(
    #             val_loader_only,
    #             num_samples=2,
    #             output_dir=os.path.join(vis_dir, "val_only"),
    #             fig_size=(18, 6),
    #         )

    #     logger.info("分别测试训练集和验证集完成")
    # except Exception as e:
    #     logger.error(f"加载单独的数据集失败: {type(e).__name__}: {e}")

    try:
        logger.info("测试 4: 正式数据集加载")
        train_loader, val_loader = create_dataloader(
            data_dir=data_path,
            split="merge",
            window_size=6,
            val_ratio=0.2,
            batch_size=8,
            num_workers=4,
        )

        logger.info(
            f"训练集大小: {len(train_loader.dataset)} 样本, {len(train_loader)} 批次"
        )
        logger.info(
            f"验证集大小: {len(val_loader.dataset)} 样本, {len(val_loader)} 批次"
        )

        sample = next(iter(train_loader))
        for key, value in sample.items():
            try:
                logger.info(f"{key}: {value.shape}")
            except AttributeError:
                logger.info(f"{key}: {len(value)}")

        logger.info("可视化训练集样本")
        visualize_batch(
            train_loader,
            num_samples=1,
            output_dir=os.path.join(vis_dir, "train_batch_final"),
            fig_size=(18, 6),
        )
        logger.info("成功可视化训练集样本")

        logger.info("可视化验证集样本")
        visualize_batch(
            val_loader,
            num_samples=1,
            output_dir=os.path.join(vis_dir, "val_batch_final"),
            fig_size=(18, 6),
        )

    except Exception as e:
        logger.error(f"加载数据集失败: {type(e).__name__}: {e}")
    logger.info("所有测试完成")
