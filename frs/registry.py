"""A tiny name -> class registry.

This is the mechanism that makes the codebase pluggable. A new dataset format,
margin head or backbone is added by decorating a class with ``@REGISTRY.register("name")``
and referencing that name from YAML. No training code changes.

    @ADAPTERS.register("my_format")
    class MyAdapter(DatasetAdapter):
        def __init__(self, root, **kw): ...

    # configs/whatever.yaml
    data:
      adapter:
        type: my_format
        root: /path/to/data
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")


class Registry:
    """Maps a string name to a class, and builds instances from config dicts."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._entries: dict[str, type] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Class decorator that registers the class under ``name``."""

        def decorator(cls: type[T]) -> type[T]:
            if name in self._entries:
                raise KeyError(
                    f"{self._name}: '{name}' is already registered to "
                    f"{self._entries[name].__name__}"
                )
            self._entries[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> type:
        if name not in self._entries:
            raise KeyError(
                f"{self._name}: unknown type '{name}'. "
                f"Available: {sorted(self._entries)}"
            )
        return self._entries[name]

    def build(self, cfg: dict[str, Any], **overrides: Any) -> Any:
        """Build an instance from ``{"type": "...", **kwargs}``.

        Keyword ``overrides`` win over the config, which lets the caller inject
        runtime values (e.g. ``num_classes``) that YAML cannot know.
        """
        if not isinstance(cfg, dict):
            raise TypeError(f"{self._name}: expected a dict config, got {type(cfg)}")
        if "type" not in cfg:
            raise KeyError(f"{self._name}: config is missing the 'type' key: {cfg}")

        cfg = dict(cfg)
        type_name = cfg.pop("type")
        cfg.update(overrides)

        cls = self.get(type_name)
        try:
            return cls(**cfg)
        except TypeError as exc:
            raise TypeError(
                f"{self._name}: failed to construct '{type_name}' with "
                f"{sorted(cfg)}: {exc}"
            ) from exc

    def names(self) -> list[str]:
        return sorted(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __repr__(self) -> str:
        return f"Registry({self._name!r}, entries={self.names()})"


ADAPTERS = Registry("adapters")
HEADS = Registry("heads")
BACKBONES = Registry("backbones")
DETECTORS = Registry("detectors")
