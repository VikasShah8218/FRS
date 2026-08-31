"""Flat directory of images whose identity is encoded in the filename.

This is the adapter for the MeGlass dataset that ships with this project:

    MeGlass_120x120/7276470@N03_identity_3@8733080112_2.jpg
    \\________________________________/ \\__________/ \\/
              identity key              photo id    face idx

The default pattern captures everything before the **last** ``@``, which is the
correct identity key here -- note that the Flickr user id itself contains an
``@`` (``7276470@N03``), so a naive split on the first ``@`` is wrong.

Verified against the real dataset: 47,917 files -> 1,710 identities.
"""

from __future__ import annotations

import re
from pathlib import Path

from ...registry import ADAPTERS
from .base import DatasetAdapter, Sample, exclude_identities, filter_by_min_images

#: Everything before the final "@". Correct for MeGlass-style names.
DEFAULT_PATTERN = r"^(?P<identity>.+)@[^@]+$"


@ADAPTERS.register("flat_regex")
class FlatRegexAdapter(DatasetAdapter):
    """Identity parsed from the filename stem with a named-group regex.

    Parameters
    ----------
    root:
        Directory containing the images.
    pattern:
        Regex applied to each file's **stem** (name without extension). It must
        define a named group ``identity``.
    extensions:
        File extensions to include, case-insensitive.
    min_images_per_identity:
        Drop identities with fewer than this many images. ``0``/``1`` disables.
    exclude_identities_file:
        Optional text file, one identity per line, listing identities to remove
        entirely. This is how the held-out validation split is enforced on the
        training set -- excluding identities here, not just from the pair list,
        is what keeps the reported accuracy honest.
    recursive:
        Search subdirectories too. MeGlass is flat, so this defaults to False.
    prefix:
        Identity namespace for multi-source merges. See :class:`DatasetAdapter`.
    """

    def __init__(
        self,
        root: str,
        pattern: str = DEFAULT_PATTERN,
        extensions: tuple[str, ...] | list[str] = (".jpg", ".jpeg", ".png"),
        min_images_per_identity: int = 0,
        exclude_identities_file: str | None = None,
        recursive: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        self.root = Path(root)
        self.pattern = pattern
        self.extensions = tuple(e.lower() for e in extensions)
        self.min_images_per_identity = int(min_images_per_identity)
        self.exclude_identities_file = exclude_identities_file
        self.recursive = bool(recursive)

        try:
            self._regex = re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"invalid identity pattern {pattern!r}: {exc}") from exc
        if "identity" not in self._regex.groupindex:
            raise ValueError(
                f"pattern {pattern!r} must contain a named group '(?P<identity>...)'"
            )

    @property
    def cache_key(self) -> str:
        return self._hash_config(
            root=str(self.root.resolve()),
            pattern=self.pattern,
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
            raise FileNotFoundError(
                f"exclude_identities_file not found: {path}. Generate it with "
                f"scripts/build_meglass_pairs.py before training."
            )
        with path.open("r", encoding="utf-8") as fh:
            return {
                line.strip()
                for line in fh
                if line.strip() and not line.startswith("#")
            }

    def scan(self) -> list[Sample]:
        if not self.root.is_dir():
            raise FileNotFoundError(f"dataset root not found: {self.root}")

        walker = self.root.rglob("*") if self.recursive else self.root.iterdir()
        samples: list[Sample] = []
        unmatched = 0
        for path in walker:
            if not path.is_file() or path.suffix.lower() not in self.extensions:
                continue
            match = self._regex.match(path.stem)
            if match is None:
                unmatched += 1
                continue
            samples.append(
                Sample(
                    path=str(path),
                    identity=self._make_identity(match.group("identity")),
                )
            )

        if not samples:
            raise RuntimeError(
                f"no samples matched in {self.root} "
                f"(extensions={self.extensions}, pattern={self.pattern!r}). "
                f"{unmatched} files were found but did not match the pattern."
            )

        excluded = self._load_excluded()
        if excluded:
            # The exclusion list is written with the prefix already applied by
            # the split builder, but tolerate bare keys too.
            expanded = excluded | {self._make_identity(e) for e in excluded}
            samples, n_removed = exclude_identities(samples, expanded)
            if n_removed == 0:
                raise RuntimeError(
                    f"exclude_identities_file listed {len(excluded)} identities but "
                    f"none matched any sample. Check the adapter prefix "
                    f"({self.prefix!r}) and that the file matches this dataset."
                )

        samples, _ = filter_by_min_images(samples, self.min_images_per_identity)
        samples.sort(key=lambda s: s.path)  # deterministic order
        return samples

    def __repr__(self) -> str:
        return (
            f"FlatRegexAdapter(root={str(self.root)!r}, "
            f"pattern={self.pattern!r}, prefix={self.prefix!r})"
        )
