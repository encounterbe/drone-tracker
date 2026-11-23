"""Configuration loading with deep-merge overrides.

A user config only needs to contain the keys it changes; everything else falls
back to ``config/default.yaml``. Access is attribute-style through a thin dict
wrapper so call sites read as ``cfg.detection.motion.min_area`` rather than a
chain of string subscripts.
"""

from __future__ import annotations

import copy
import os
from typing import Any

import yaml

__all__ = ["Config", "load_config", "DEFAULT_CONFIG_PATH"]

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "config",
    "default.yaml",
)


class Config(dict):
    """Dict with attribute access, recursively applied to nested mappings.

    Nested dicts are converted to ``Config`` *once, at insertion time*, and
    attribute access then returns that stored object. Converting lazily inside
    ``__getattr__`` instead would hand back a fresh shallow copy on every
    access, so ``cfg.zoom["initial"] = 3.0`` would mutate a temporary and be
    silently discarded - a failure mode that produces no error and is very
    expensive to track down.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for key, value in list(self.items()):
            if isinstance(value, dict) and not isinstance(value, Config):
                super().__setitem__(key, Config(value))

    def __setitem__(self, key: str, value: Any) -> None:
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
        super().__setitem__(key, value)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(
                f"No config key {name!r}. Available: {sorted(self.keys())}"
            ) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """Look up ``"detection.motion.min_area"`` style paths."""
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node: dict = self
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce(text: str) -> Any:
    """Parse a CLI override value using YAML scalar rules."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def load_config(
    path: str | None = None,
    overrides: list[str] | None = None,
) -> Config:
    """Load defaults, merge an optional user file, then apply ``key=value`` overrides.

    Args:
        path: Optional YAML file merged on top of the defaults.
        overrides: ``["detection.motion.min_area=25", "ui.theme=nightvision"]``.
    """
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as fh:
        merged: dict = yaml.safe_load(fh) or {}

    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            merged = _deep_merge(merged, yaml.safe_load(fh) or {})

    cfg = Config(merged)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Malformed override {item!r}; expected key.path=value")
        key, _, raw = item.partition("=")
        cfg.set_path(key.strip(), _coerce(raw.strip()))
    return cfg
