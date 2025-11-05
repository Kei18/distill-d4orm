import sys
from loguru import logger


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
