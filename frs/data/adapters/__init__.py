"""Dataset adapters.

Every adapter module is imported here so that its ``@ADAPTERS.register(...)``
decorator runs and the type name becomes available to YAML configs.

**To add a new dataset format:** write the adapter module, then add one import
line below. Nothing else in the codebase changes. See ``docs/ADAPTERS.md``.
"""

from .base import DatasetAdapter, Sample, exclude_identities, filter_by_min_images

# --- registered adapters (import for side effect) ---
from . import csv_manifest  # noqa: F401  "csv_manifest"
from . import flat_regex  # noqa: F401  "flat_regex"
from . import folder_per_identity  # noqa: F401  "folder_per_identity"
from . import mxnet_rec  # noqa: F401  "mxnet_rec"

__all__ = [
    "DatasetAdapter",
    "Sample",
    "exclude_identities",
    "filter_by_min_images",
]
