import os
from typing import Optional


def format_display_filename(source_file: Optional[str], max_parts: int = 3) -> str:
    """Return a shortened display filename for a source file path.

    Keeps the last `max_parts` components of the path joined with '/'.
    If source_file is None or empty, returns 'Unknown'.
    Normalizes path separators to '/'.
    """
    if not source_file:
        return "Unknown"

    # Normalize separators and split
    parts = source_file.replace("\\", "/").split("/")
    # Filter out empty parts
    parts = [p for p in parts if p]
    if not parts:
        return "Unknown"

    return "/".join(parts[-max_parts:])
