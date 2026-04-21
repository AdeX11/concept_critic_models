"""
Runtime helpers shared across training, comparison, and cluster scripts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


class NullSummaryWriter:
    def add_scalar(self, *args, **kwargs) -> None:
        return None

    def add_text(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None


def make_summary_writer(log_dir: str):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        return NullSummaryWriter()
    return SummaryWriter(log_dir=log_dir)


def get_obs_shape(env):
    obs_space = env.observation_space
    if hasattr(obs_space, "spaces"):
        return {k: v.shape for k, v in obs_space.spaces.items()}
    return obs_space.shape


def write_json(path: str | Path, data: Dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def sanitize_info_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return str(value)


def flatten_terminal_info(info: Dict[str, Any]) -> Dict[str, Any]:
    return {k: sanitize_info_value(v) for k, v in info.items()}
