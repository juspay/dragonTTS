"""Loguru logging setup (mirrors clairvoyance's app.core.logger)."""

import sys

from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    enqueue=True,
    backtrace=False,
    diagnose=False,
)

__all__ = ["logger"]
