"""One subdirectory per identity -- the classic ImageFolder layout.

    root/
      alice/  img001.jpg  img002.jpg
      bob/    photo.png   ...

This is the most common format for datasets you assemble yourself, and the one
most public face datasets ship in once unpacked.
"""

from __future__ import annotations

from pathlib import Path

from ...registry import ADAPTERS
from .base import DatasetAdapter, Sample, exclude_identities, filter_by_min_images


@ADAPTERS.register("folder_per_identity")
class FolderPerIdentityAdapter(DatasetAdapter):
    """Identity is the name of the directory containing the image.

    Parameters
    ----------
    root:
        Directory whose immediate subdirectories are identities.
    extensions:
        File extensions to include, case-insensitive.
    min_images_per_identity:
        Drop identities with fewer than this many images.
    exclude_identities_file:
        Optional text file of identities to hold out (validation split).
    recursive:
        If True, images nested deeper than one level are attributed to their
        top-level identity directory. Useful for ``root/id/session/img.jpg``.
    prefix:
        Identity namespace for multi-source merges.
    """

    def __init__(
        self,
        root: str,
        extensions: tuple[str, ...] | list[str] = (".jpg", ".jpeg", ".png", ".bmp"),
        min_images_per_identity: int = 0,
        exclude_identities_file: str | None = None,
        recursive: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        self.root = Path(root)
        self.extensions = tuple(e.lower() for e in extensions)
        self.min_images_per_identity = int(min_images_per_identity)
        self.exclude_identities_file = exclude_identities_file
        self.recursive = bool(recursive)

    @property
    def cache_key(self) -> str:
        return self._hash_config(
            root=str(self.root.resolve()),
            extensions=self.extensions,
            min_images=self.min_images_per_identity,
            exclude=self.exclude_identities_file,
            recursive=self.recursive,
        )

    def _load_excluded(self) -> set[str]:
        if not self.exclude_identities_file:
            return set()
        path = Path(self.exclude_identities_file)
        if not path.is_file():
            raise FileNotFoundError(f"exclude_identities_file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            return {
                line.strip() for line in fh if line.strip() and not line.startswith("#")
            }

    def scan(self) -> list[Sample]:
        if not self.root.is_dir():
            raise FileNotFoundError(f"dataset root not found: {self.root}")

        samples: list[Sample] = []
        for identity_dir in sorted(self.root.iterdir()):
            if not identity_dir.is_dir():
                continue
            walker = (
                identity_dir.rglob("*") if self.recursive else identity_dir.iterdir()
            )
            for path in walker:
                if path.is_file() and path.suffix.lower() in self.extensions:
                    samples.append(
                        Sample(
                            path=str(path),
                            identity=self._make_identity(identity_dir.name),
                        )
                    )

        if not samples:
            raise RuntimeError(
                f"no images found under {self.root} with extensions {self.extensions}. "
                f"Expected one subdirectory per identity."
            )

        excluded = self._load_excluded()
        if excluded:
            expanded = excluded | {self._make_identity(e) for e in excluded}
            samples, _ = exclude_identities(samples, expanded)

        samples, _ = filter_by_min_images(samples, self.min_images_per_identity)
        samples.sort(key=lambda s: s.path)
        return samples

    def __repr__(self) -> str:
        return f"FolderPerIdentityAdapter(root={str(self.root)!r}, prefix={self.prefix!r})"
