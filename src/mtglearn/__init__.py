import logging

from .datasets import Card

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


__all__ = ["Card"]
