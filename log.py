import logging
import os
import sys
from datetime import datetime

# 创建logs目录（如果不存在）
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(logs_dir, exist_ok=True)

# 日志文件名（使用当前日期）
log_file = os.path.join(logs_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")

# 创建logger实例
logger = logging.getLogger("robotic-transformer")
logger.setLevel(logging.DEBUG)

# 定义日志格式
formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# 添加文件处理器
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# 添加控制台处理器
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)  # 控制台只显示INFO及以上级别
console_handler.setFormatter(formatter)

# 将处理器添加到logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


# 提供一个简单的方法来调整日志级别
def set_log_level(level):
    """
    设置日志记录级别
    :param level: 可以是logging的级别常量或对应的字符串
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    logger.setLevel(level)

    # 可选：同时调整处理器级别
    # console_handler.setLevel(level)
    # file_handler.setLevel(level)


# 添加新函数用于设置日志输出目的地
def set_log_destination(console_only=False, file_only=False):
    """
    设置日志输出目的地
    :param console_only: 如果为True，只将日志输出到控制台
    :param file_only: 如果为True，只将日志输出到文件
    注意：如果两个参数都为False（默认），则同时输出到控制台和文件
          如果两个参数都为True，则优先考虑console_only
    """
    # 首先移除所有现有处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 根据参数重新添加处理器
    if console_only:
        # 只添加控制台处理器
        logger.addHandler(console_handler)
    elif file_only:
        # 只添加文件处理器
        logger.addHandler(file_handler)
    else:
        # 两者都添加（默认行为）
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
