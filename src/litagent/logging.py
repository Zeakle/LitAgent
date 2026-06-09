import logging
from litagent.config import LoggingConfig

def setup_logging(config: LoggingConfig) -> None:
    """初始化全局日志配置。

    整个应用生命周期只调用一次。后续直接用 get_logger() 获取 logger。

    Args:
        config: 包含日志级别和格式的配置对象
    """
    logging.basicConfig(
        level=getattr(logging, config.level.upper(), logging.INFO),
        format=config.format,
    )


def get_logger(name: str) -> logging.Logger:
    """获取模块级 logger。

    Args:
        name: 模块名，自动加 litagent. 前缀

    Returns:
        Logger 实例。例: get_logger("config") → logger named "litagent.config"
    """
    return logging.getLogger(f'litagent.{name}')