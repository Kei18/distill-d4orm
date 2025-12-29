import sys
from loguru import logger

from .d4orm import D4ormCfg, d4orm_opt, get_metrics, configure_jax
from .envs import get_env_cls
from .envs.multibase import EnvConfig
from .viz import save_anim, save_img


def configure_logger(
    level: str = "DEBUG",
    colorize: bool = True,
    format_string: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> <level>{message}</level>",
) -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        colorize=colorize,
        format=format_string,
        level=level,
    )


configure_logger()

__all__ = [
    "D4ormCfg",
    "d4orm_opt",
    "EnvConfig",
    "get_env_cls",
    "save_anim",
    "save_img",
    "get_metrics",
    "configure_jax",
]
