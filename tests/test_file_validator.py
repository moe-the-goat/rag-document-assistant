import os
import pytest
import tempfile
from app.file_validator import validate_file_size, validate_file_type, sanitize_filename
from unittest.mock import patch

def test_sanitize_filename():
    assert sanitize_filename("normal.pdf") == "normal.pdf"
    assert sanitize_filename("../../../etc/passwd.txt") == "passwd.txt"
    assert sanitize_filename('malicious<script>.html') == 'malicious_script_.html'
    # Test very long filename
    long_name = "a" * 150 + ".txt"
    sanitized = sanitize_filename(long_name)
    assert len(sanitized) == 104  # 100 base + 4 ext

def test_validate_file_size():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"0" * 1024)  # 1 KB
        temp_path = f.name

    try:
        # Should pass
        validate_file_size(temp_path)

        # Test rejection (mock MAX_FILE_SIZE_MB to 0)
        with patch("app.file_validator.MAX_FILE_SIZE_MB", 0):
            with pytest.raises(ValueError, match="File is too large"):
                validate_file_size(temp_path)
    finally:
        os.remove(temp_path)

def test_validate_file_type_pdf():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(b"%PDF-1.4...")
        temp_path = f.name

    try:
        # Should pass
        validate_file_type(temp_path, ".pdf")
    finally:
        os.remove(temp_path)

def test_validate_file_type_invalid_pdf():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(b"This is not a PDF")
        temp_path = f.name

    try:
        # Should fail magic byte check
        with pytest.raises(ValueError, match="File content doesn't match"):
            validate_file_type(temp_path, ".pdf")
    finally:
        os.remove(temp_path)

def test_validate_file_type_txt():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        f.write("Valid text content".encode("utf-8"))
        temp_path = f.name

    try:
        validate_file_type(temp_path, ".txt")
    finally:
        os.remove(temp_path)

def test_validate_file_type_invalid_txt():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
        f.write(b"\xff\xfe\x00\x00")  # Invalid UTF-8 sequence
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="contains non-text content"):
            validate_file_type(temp_path, ".txt")
    finally:
        os.remove(temp_path)
