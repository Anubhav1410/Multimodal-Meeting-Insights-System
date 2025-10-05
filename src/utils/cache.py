from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch


@lru_cache(maxsize=1)
def get_device_str() -> str:
  return "cuda" if torch.cuda.is_available() else "cpu"
