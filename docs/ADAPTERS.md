# Adding a New Dataset Format

Supporting a new on-disk layout takes one file and one import line. No change to
the trainer, the model, the loss, or the checkpoint code.

---

## The contract

A **dataset adapter** answers two questions:

1. `scan()` — what samples exist, and which person is in each?
2. `load_image(sample)` — give me the pixels for one sample.

That is all. Adapters know nothing about batching, class indices, PyTorch or
training.

### The one rule that matters

```python
Sample(path="/data/img_001.jpg", identity="alice")   # correct
Sample(path="/data/img_001.jpg", identity=0)         # WRONG
```

**Identities are strings, never integers.** The `ClassMap` assigns integer class
indices, not the adapter.

This is what makes incremental training possible. The classifier's row *i* is the
learned prototype for class *i*; if adapters assigned indices, then re-scanning a
folder — or merging a second dataset — could silently reassign them, and every
learned row would refer to the wrong person. Because identities are strings, the
`ClassMap` can guarantee an identity's index never changes, and a new dataset
simply appends new rows to an existing head.

---

## Built-in adapters

| `type` | Layout | Key options |
|---|---|---|
| `flat_regex` | Flat folder, identity in the filename | `pattern`, `extensions` |
| `folder_per_identity` | `root/alice/*.jpg` | `recursive` |
| `csv_manifest` | CSV of `path,identity` | `path_column`, `identity_column`, `delimiter` |
| `mxnet_rec` | `.rec`/`.idx` packs (MS1MV3, Glint360K) | `rec_name`, `idx_name` |

Every adapter also supports:

| Option | Purpose |
|---|---|
| `min_images_per_identity` | Drop identities with too few images to learn from |
| `exclude_identities_file` | Hold identities out (the validation split) |
| `prefix` | Namespace identities when merging sources |

### Picking one

```yaml
data:
  adapter:
    type: folder_per_identity
    root: /data/employees
    min_images_per_identity: 3
```

### Merging two sources safely

If two datasets both contain an identity called `"john_smith"` but they are
*different* people, namespace them:

```yaml
# dataset A
adapter: {type: folder_per_identity, root: /data/site_a, prefix: "site_a/"}
# dataset B
adapter: {type: folder_per_identity, root: /data/site_b, prefix: "site_b/"}
```

Without a prefix they would be merged into one class. That is the correct default
when identity keys are globally meaningful (an employee id, a passport number)
and wrong when they are locally assigned names.

---

## Writing a new adapter

Say your data arrives as `<identity>_<date>_<seq>.png` in dated subfolders, plus
a JSON sidecar of quality scores.

### 1. Write the module

`frs/data/adapters/dated_folders.py`:

```python
"""Dated subfolders with identity-prefixed filenames.

    root/2026-01-14/alice_20260114_001.png
    root/2026-01-15/bob_20260115_007.png
"""

from __future__ import annotations

import json
from pathlib import Path

from ...registry import ADAPTERS
from .base import DatasetAdapter, Sample, filter_by_min_images


@ADAPTERS.register("dated_folders")
class DatedFoldersAdapter(DatasetAdapter):
    def __init__(
        self,
        root: str,
        extensions=(".png", ".jpg"),
        quality_file: str | None = None,
        min_images_per_identity: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        self.root = Path(root)
        self.extensions = tuple(e.lower() for e in extensions)
        self.quality_file = quality_file
        self.min_images_per_identity = int(min_images_per_identity)

    @property
    def cache_key(self) -> str:
        # Must change whenever anything that affects scan() output changes,
        # or a stale cached scan will be reused silently.
        return self._hash_config(
            root=str(self.root.resolve()),
            extensions=self.extensions,
            quality=self.quality_file,
            min_images=self.min_images_per_identity,
        )

    def scan(self) -> list[Sample]:
        if not self.root.is_dir():
            raise FileNotFoundError(f"dataset root not found: {self.root}")

        quality = {}
        if self.quality_file:
            quality = json.loads(Path(self.quality_file).read_text(encoding="utf-8"))

        samples = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in self.extensions:
                continue
            identity = path.stem.split("_")[0]
            samples.append(
                Sample(
                    path=str(path),
                    identity=self._make_identity(identity),
                    meta={"quality": quality.get(path.name)},
                )
            )

        if not samples:
            raise RuntimeError(f"no images found under {self.root}")

        samples, _ = filter_by_min_images(samples, self.min_images_per_identity)
        return samples
```

`load_image` is inherited — the default reads the path with PIL and returns RGB
`HWC uint8`. Override it only for packed formats (see `mxnet_rec.py`, which
interprets a `rec://<index>` URI).

### 2. Register it

Add one line to `frs/data/adapters/__init__.py`:

```python
from . import dated_folders  # noqa: F401  "dated_folders"
```

### 3. Use it

```yaml
data:
  adapter:
    type: dated_folders
    root: /data/captures
    quality_file: /data/quality.json
    min_images_per_identity: 3
```

### 4. Test it

```python
def test_dated_folders(tmp_path):
    write_image(tmp_path / "2026-01-14" / "alice_20260114_001.png")
    write_image(tmp_path / "2026-01-15" / "alice_20260115_002.png")
    write_image(tmp_path / "2026-01-15" / "bob_20260115_003.png")

    samples = DatedFoldersAdapter(root=str(tmp_path)).scan()
    assert len(samples) == 3
    assert {s.identity for s in samples} == {"alice", "bob"}
```

Done. Training, checkpointing, evaluation and reporting all work unchanged.

---

## Verifying a new adapter before training

```python
from frs.data.adapters.base import DatasetAdapter
from frs.registry import ADAPTERS
import frs.data.adapters  # registers everything

adapter = ADAPTERS.build({"type": "dated_folders", "root": "/data/captures"})
samples = adapter.scan()

print(DatasetAdapter.summarize(samples))
# {'num_samples': 12043, 'num_identities': 430, 'min_per_identity': 2, ...}

img = adapter.load_image(samples[0])
print(img.shape, img.dtype)   # (112, 112, 3) uint8  <- must be RGB HWC uint8
```

Sanity checks worth doing:

- **Identity count** — plausible? A count equal to the image count means your
  parsing is producing a unique identity per file.
- **`min_per_identity`** — identities with 1 image cannot be learned
  discriminatively. Set `min_images_per_identity: 2` or higher.
- **Image shape** — must be `HWC uint8` RGB. Grayscale or BGR will train but
  degrade accuracy silently.

---

## Gotchas

**Scan caching.** Results are cached to `.cache/scan/` keyed by `cache_key`. If
you change adapter logic during development, either bump the cache key inputs or
delete the cache — otherwise you will keep loading a stale scan.

**Identity leakage.** If you use `exclude_identities_file`, the excluded strings
must match what your adapter produces *including any prefix*. The built-in
adapters tolerate both forms, but a custom adapter should too.

**Deterministic ordering.** Sort your samples before returning them. The
`ClassMap` sorts identities anyway, but a stable sample order makes runs
reproducible and cache keys meaningful.

**Large datasets.** For millions of images, `scan()` runs once and is cached, so
a slow scan is acceptable — but avoid loading image *data* during the scan.
`mxnet_rec` reads only record labels, not pixels.
