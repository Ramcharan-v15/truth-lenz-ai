"""V5 real-world dataset subsystem (additive; V4 datasets/ modules untouched)."""
from .registry import (
    LICENSE_STATES,
    get_dataset,
    is_legally_usable,
    list_datasets,
    register_dataset,
    registry_path,
    update_license_status,
)

__all__ = [
    "LICENSE_STATES", "get_dataset", "is_legally_usable", "list_datasets",
    "register_dataset", "registry_path", "update_license_status",
]
