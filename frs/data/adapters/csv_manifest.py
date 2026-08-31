"""A CSV/TSV manifest listing image paths and their identities.

    path,identity
    images/0001.jpg,alice
    images/0002.jpg,bob

This covers the common real-world case where a client hands over a folder of
images plus a spreadsheet of labels, and any situation where the on-disk layout
carries no identity information at all.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ...registry import ADAPTERS
from .base import DatasetAdapter, Sample, exclude_identities, filter_by_min_images


@ADAPTERS.register("csv_manifest")
class CsvManifestAdapter(DatasetAdapter):
    """Read (path, identity) pairs from a delimited text file.

    Parameters
    ----------
    manifest:
        Path to the CSV/TSV file.
    root:
        Optional base directory. Relative paths in the manifest are resolved
        against it; absolute paths are used as-is. Defaults to the manifest's
        own directory.
    path_column, identity_column:
        Column names (when the file has a header) or 0-based integer indices.
    delimiter:
        Field separator. Defaults to ``","``; use ``"\\t"`` for TSV.
    has_header:
        Whether the first row is a header. Required if columns are named.
    meta_columns:
        Extra columns to carry through into ``Sample.meta``.
    min_images_per_identity, exclude_identities_file, prefix:
        As in the other adapters.
    """

    def __init__(
        self,
        manifest: str,
        root: str | None = None,
        path_column: str | int = "path",
        identity_column: str | int = "identity",
        delimiter: str = ",",
        has_header: bool = True,
        meta_columns: tuple[str, ...] | list[str] = (),
        min_images_per_identity: int = 0,
        exclude_identities_file: str | None = None,
        verify_exists: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        self.manifest = Path(manifest)
        self.root = Path(root) if root else self.manifest.parent
        self.path_column = path_column
        self.identity_column = identity_column
        self.delimiter = delimiter
        self.has_header = bool(has_header)
        self.meta_columns = tuple(meta_columns)
        self.min_images_per_identity = int(min_images_per_identity)
        self.exclude_identities_file = exclude_identities_file
        self.verify_exists = bool(verify_exists)

        if not has_header and not (
            isinstance(path_column, int) and isinstance(identity_column, int)
        ):
            raise ValueError(
                "with has_header=False, path_column and identity_column must be "
                "integer indices"
            )

    @property
    def cache_key(self) -> str:
        stat = self.manifest.stat() if self.manifest.is_file() else None
        return self._hash_config(
            manifest=str(self.manifest.resolve()),
            mtime=stat.st_mtime_ns if stat else None,
            size=stat.st_size if stat else None,
            root=str(self.root.resolve()),
            path_column=self.path_column,
            identity_column=self.identity_column,
            delimiter=self.delimiter,
            has_header=self.has_header,
            min_images=self.min_images_per_identity,
            exclude=self.exclude_identities_file,
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

    def _resolve(self, raw_path: str) -> str:
        p = Path(raw_path)
        return str(p if p.is_absolute() else self.root / p)

    def scan(self) -> list[Sample]:
        if not self.manifest.is_file():
            raise FileNotFoundError(f"manifest not found: {self.manifest}")

        samples: list[Sample] = []
        missing: list[str] = []

        with self.manifest.open("r", encoding="utf-8", newline="") as fh:
            if self.has_header:
                reader = csv.DictReader(fh, delimiter=self.delimiter)
                if reader.fieldnames is None:
                    raise RuntimeError(f"{self.manifest} is empty")
                for col in (self.path_column, self.identity_column):
                    if isinstance(col, str) and col not in reader.fieldnames:
                        raise KeyError(
                            f"column {col!r} not in manifest header "
                            f"{reader.fieldnames}"
                        )
                rows = (
                    (
                        row[self.path_column]
                        if isinstance(self.path_column, str)
                        else list(row.values())[self.path_column],
                        row[self.identity_column]
                        if isinstance(self.identity_column, str)
                        else list(row.values())[self.identity_column],
                        {c: row.get(c) for c in self.meta_columns} or None,
                    )
                    for row in reader
                )
                collected = list(rows)
            else:
                plain = csv.reader(fh, delimiter=self.delimiter)
                collected = [
                    (r[self.path_column], r[self.identity_column], None)
                    for r in plain
                    if r
                ]

        for raw_path, identity, meta in collected:
            if not raw_path or not identity:
                continue
            full = self._resolve(raw_path.strip())
            if self.verify_exists and not Path(full).is_file():
                missing.append(full)
                continue
            samples.append(
                Sample(
                    path=full,
                    identity=self._make_identity(identity.strip()),
                    meta=meta,
                )
            )

        if missing:
            raise FileNotFoundError(
                f"{len(missing)} files listed in {self.manifest} do not exist, "
                f"e.g. {missing[:3]}. Check the 'root' setting."
            )
        if not samples:
            raise RuntimeError(f"no usable rows in {self.manifest}")

        excluded = self._load_excluded()
        if excluded:
            expanded = excluded | {self._make_identity(e) for e in excluded}
            samples, _ = exclude_identities(samples, expanded)

        samples, _ = filter_by_min_images(samples, self.min_images_per_identity)
        return samples

    def __repr__(self) -> str:
        return f"CsvManifestAdapter(manifest={str(self.manifest)!r}, prefix={self.prefix!r})"
