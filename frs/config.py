"""YAML configuration loading with inheritance, interpolation and dot-overrides.

Features
--------
* ``_base_: other.yaml`` -- inherit and deep-merge from another config (path is
  relative to the including file). Chains are supported.
* ``${a.b.c}`` -- interpolate another key's value. Resolved after merging.
* ``--set a.b.c=value`` -- CLI dot-overrides, parsed with YAML scalar rules so
  ``true`` / ``3`` / ``null`` arrive as the right Python type.
* Attribute access -- ``cfg.train.epochs`` as well as ``cfg["train"]["epochs"]``.

Typos are caught at startup by :func:`validate` rather than at epoch 12.
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

_INTERP = re.compile(r"\$\{([^}]+)\}")
_MAX_INTERP_PASSES = 10


class ConfigError(Exception):
    """Raised for malformed or invalid configuration."""


class Config(dict):
    """A dict that also supports attribute access, recursively."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(
                f"config has no key '{key}'. Available: {sorted(self)}"
            ) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        del self[key]

    @classmethod
    def _wrap(cls, obj: Any) -> Any:
        if isinstance(obj, Mapping):
            return cls({k: cls._wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [cls._wrap(v) for v in obj]
        return obj

    def to_dict(self) -> dict:
        """Plain nested dict -- safe for torch.save / yaml.dump."""

        def unwrap(obj: Any) -> Any:
            if isinstance(obj, Mapping):
                return {k: unwrap(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [unwrap(v) for v in obj]
            return obj

        return unwrap(self)

    def get_path(self, dotted: str, default: Any = ...) -> Any:
        """Fetch a nested value by ``"a.b.c"``."""
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                if default is ...:
                    raise KeyError(f"config path not found: '{dotted}'")
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        """Set a nested value by ``"a.b.c"``, creating intermediate dicts."""
        parts = dotted.split(".")
        node: Any = self
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], Mapping):
                node[part] = Config()
            node = node[part]
        node[parts[-1]] = value

    def dump(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)


def deep_merge(base: Mapping, override: Mapping) -> dict:
    """Recursively merge ``override`` into ``base``. Lists are replaced wholesale."""
    out = dict(copy.deepcopy(dict(base)))
    for key, value in override.items():
        if key in out and isinstance(out[key], Mapping) and isinstance(value, Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_raw(path: Path, _seen: tuple[Path, ...] = ()) -> dict:
    """Load one YAML file, resolving its ``_base_`` chain."""
    path = path.resolve()
    if path in _seen:
        chain = " -> ".join(p.name for p in (*_seen, path))
        raise ConfigError(f"circular _base_ chain: {chain}")
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(data)}")

    base_ref = data.pop("_base_", None)
    if base_ref is None:
        return data

    refs = [base_ref] if isinstance(base_ref, str) else list(base_ref)
    merged: dict = {}
    for ref in refs:
        base_path = (path.parent / ref).resolve()
        merged = deep_merge(merged, _load_raw(base_path, (*_seen, path)))
    return deep_merge(merged, data)


def _interpolate(cfg: Config) -> Config:
    """Resolve ``${dotted.key}`` references, repeatedly until stable."""

    def resolve_scalar(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        match = _INTERP.fullmatch(value.strip())
        if match:  # whole-string reference -- preserve the referent's type
            return cfg.get_path(match.group(1).strip())

        def sub(m: "re.Match[str]") -> str:
            return str(cfg.get_path(m.group(1).strip()))

        return _INTERP.sub(sub, value)

    def walk(node: Any) -> Any:
        if isinstance(node, Config):
            return Config({k: walk(v) for k, v in node.items()})
        if isinstance(node, list):
            return [walk(v) for v in node]
        return resolve_scalar(node)

    for _ in range(_MAX_INTERP_PASSES):
        new = walk(cfg)
        if new == cfg:
            return new
        cfg = new
    raise ConfigError(
        "config interpolation did not converge -- check for a ${} reference cycle"
    )


def apply_overrides(cfg: Config, overrides: Iterable[str]) -> Config:
    """Apply CLI dot-overrides such as ``a.b=1`` or ``train.amp=false``."""
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"malformed override '{item}', expected key=value")
        key, _, raw = item.partition("=")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        cfg.set_path(key.strip(), Config._wrap(value))
    return cfg


# Every key that must be present after merging. Catches typos and truncated
# configs at startup instead of hours into a run.
REQUIRED_KEYS = (
    "experiment.name",
    "experiment.output_dir",
    "experiment.seed",
    "data.adapter.type",
    "data.input_size",
    "data.batch_size",
    "model.backbone.arch",
    "model.head.type",
    "optim.type",
    "optim.lr",
    "scheduler.type",
    "train.epochs",
)


def validate(cfg: Config) -> Config:
    """Check required keys and cross-field consistency."""
    missing = [k for k in REQUIRED_KEYS if cfg.get_path(k, None) is None]
    if missing:
        raise ConfigError("config is missing required keys: " + ", ".join(missing))

    size = cfg.get_path("data.input_size")
    if not (isinstance(size, list) and len(size) == 2 and size[0] == size[1]):
        raise ConfigError(f"data.input_size must be [N, N]; got {size}")
    if size[0] not in (112, 224):
        raise ConfigError(
            f"data.input_size must be 112 or 224 (IResNet constraint); got {size[0]}"
        )

    if cfg.get_path("data.batch_size") < 1:
        raise ConfigError("data.batch_size must be >= 1")
    if cfg.get_path("train.grad_accum_steps", 1) < 1:
        raise ConfigError("train.grad_accum_steps must be >= 1")
    if cfg.get_path("train.epochs") < 1:
        raise ConfigError("train.epochs must be >= 1")
    if cfg.get_path("optim.lr") <= 0:
        raise ConfigError("optim.lr must be > 0")

    warmup = cfg.get_path("scheduler.warmup_epochs", 0)
    if warmup >= cfg.get_path("train.epochs"):
        raise ConfigError(
            f"scheduler.warmup_epochs ({warmup}) must be < train.epochs "
            f"({cfg.get_path('train.epochs')})"
        )

    primary = cfg.get_path("eval.primary", None)
    if primary is not None:
        targets = [t["name"] for t in cfg.get_path("eval.targets", [])]
        if targets and primary not in targets:
            raise ConfigError(
                f"eval.primary '{primary}' is not among eval.targets {targets}"
            )
    return cfg


def load_config(
    path: str | os.PathLike,
    overrides: Iterable[str] | None = None,
    do_validate: bool = True,
) -> Config:
    """Load, merge, interpolate, override and validate a config file."""
    cfg = Config._wrap(_load_raw(Path(path)))
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    cfg = _interpolate(cfg)
    if do_validate:
        cfg = validate(cfg)
    return cfg
