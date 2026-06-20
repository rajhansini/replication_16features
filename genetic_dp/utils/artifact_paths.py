from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Iterable


ONEDRIVE_SAFE_PATH_CHARS = 390
MAX_SAFE_FILENAME_CHARS = 240


def safe_path_component(value: object, *, fallback: str = "artifact") -> str:
    """Return a filesystem-safe, readable path component."""

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return safe or fallback


def _onedrive_display_path(path: Path) -> str:
    absolute = path if path.is_absolute() else Path.cwd() / path
    text = str(absolute)
    home = str(Path.home())
    cloud_prefix = f"{home}/Library/CloudStorage/OneDrive-UniversityofCincinnati"
    display_prefix = f"{home}/OneDrive - University of Cincinnati"
    if text == cloud_prefix or text.startswith(f"{cloud_prefix}/"):
        return f"{display_prefix}{text[len(cloud_prefix):]}"
    return text


def projected_sync_path_chars(path: Path, *, root_prefix_chars: int | None = None) -> int:
    """Length of the path OneDrive is likely to evaluate for sync."""

    if root_prefix_chars is not None:
        return root_prefix_chars + len(str(path))
    return len(_onedrive_display_path(path))


def safe_artifact_path(
    directory: Path,
    stem_parts: Iterable[object],
    *,
    suffix: str,
    max_path_chars: int | None = None,
    digest_chars: int = 10,
    root_prefix_chars: int | None = None,
) -> Path:
    """Build a readable artifact path and shorten the filename when needed.

    The shortening keeps the output directory stable, preserves a readable stem
    prefix, and appends a digest from the full unshortened stem so collisions are
    unlikely.
    """

    limit = max_path_chars
    if limit is None:
        limit = int(os.getenv("ONEDRIVE_SAFE_PATH_CHARS", str(ONEDRIVE_SAFE_PATH_CHARS)))
    directory = Path(directory)
    safe_parts = [safe_path_component(part) for part in stem_parts]
    stem = "_".join(part for part in safe_parts if part) or "artifact"
    candidate = directory / f"{stem}{suffix}"
    if (
        projected_sync_path_chars(candidate, root_prefix_chars=root_prefix_chars) <= limit
        and len(candidate.name) <= MAX_SAFE_FILENAME_CHARS
    ):
        return candidate

    digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:digest_chars]
    parent_chars = projected_sync_path_chars(directory, root_prefix_chars=root_prefix_chars)
    name_budget = min(MAX_SAFE_FILENAME_CHARS, limit - parent_chars - 1)
    reserve = len("_") + digest_chars + len(suffix)
    if name_budget < reserve + 8:
        raise ValueError(
            f"Output directory is too long for a sync-safe artifact path: {directory}"
        )
    prefix_budget = name_budget - reserve
    shortened_stem = stem[:prefix_budget].rstrip("._-") or "artifact"
    shortened = directory / f"{shortened_stem}_{digest}{suffix}"
    if projected_sync_path_chars(shortened, root_prefix_chars=root_prefix_chars) > limit:
        raise ValueError(f"Could not build a sync-safe artifact path under {directory}")
    return shortened
