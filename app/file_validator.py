# file_validator.py
# Validates uploaded files beyond just checking the file extension.
# This is a security layer that protects against:
#   - Oversized files that could exhaust disk/memory
#   - Files pretending to be something they're not (renamed executables)
#   - Malicious filenames (path traversal, shell injection)
#
# Uses magic bytes (file signatures) for type verification —
# no external dependencies required.

import os
import re
import logging

from app.config import MAX_FILE_SIZE_MB

logger = logging.getLogger(__name__)

# file signatures (magic bytes) for supported document types
# these are the first few bytes of each file format
MAGIC_SIGNATURES = {
    ".pdf":  [b"%PDF"],
    ".docx": [b"PK\x03\x04", b"PK\x05\x06"],  # ZIP-based (Office Open XML)
    ".txt":  [],  # text files have no magic bytes, validated by encoding
    ".md":   [],  # same as text
    ".html": [],  # same as text
}

# characters that are dangerous in filenames
_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def validate_file_size(file_path: str) -> None:
    """Reject files that exceed the configured maximum size."""
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    actual_size = os.path.getsize(file_path)

    if actual_size > max_bytes:
        size_mb = actual_size / (1024 * 1024)
        logger.warning(
            f"File rejected: {size_mb:.1f} MB exceeds limit of {MAX_FILE_SIZE_MB} MB"
        )
        raise ValueError(
            f"File is too large ({size_mb:.1f} MB). "
            f"Maximum allowed size is {MAX_FILE_SIZE_MB} MB."
        )


def validate_file_type(file_path: str, expected_ext: str) -> None:
    """
    Verify a file's actual type matches its extension using magic bytes.

    For binary formats (PDF, DOCX), we check the file signature.
    For text formats (TXT, MD, HTML), we verify the file is valid UTF-8 text.
    """
    expected_ext = expected_ext.lower()

    if expected_ext not in MAGIC_SIGNATURES:
        raise ValueError(f"Unsupported file type: {expected_ext}")

    signatures = MAGIC_SIGNATURES[expected_ext]

    if signatures:
        # binary format — check magic bytes
        with open(file_path, "rb") as f:
            header = f.read(8)  # read first 8 bytes

        if not any(header.startswith(sig) for sig in signatures):
            logger.warning(
                f"File rejected: content doesn't match {expected_ext} format "
                f"(magic bytes: {header[:4].hex()})"
            )
            raise ValueError(
                f"File content doesn't match the {expected_ext} format. "
                f"The file may be corrupted or mislabeled."
            )
    else:
        # text format — verify it's actually readable text
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                f.read(4096)  # sample the first 4KB
        except UnicodeDecodeError:
            logger.warning(
                f"File rejected: claimed to be {expected_ext} but "
                f"isn't valid UTF-8 text"
            )
            raise ValueError(
                f"File claims to be {expected_ext} but contains "
                f"non-text content. It may be corrupted or mislabeled."
            )


def sanitize_filename(filename: str) -> str:
    """
    Clean up a filename to prevent path traversal and shell injection.

    - Strips directory components (no ../ tricks)
    - Removes dangerous characters
    - Limits length
    - Falls back to 'unnamed' if nothing is left
    """
    # strip any directory components
    name = os.path.basename(filename)

    # remove dangerous characters
    name = _UNSAFE_CHARS.sub("_", name)

    # limit length (preserving extension)
    base, ext = os.path.splitext(name)
    if len(base) > 100:
        base = base[:100]
    name = base + ext

    # fallback if empty
    if not base:
        name = f"unnamed{ext}" if ext else "unnamed.txt"

    return name
