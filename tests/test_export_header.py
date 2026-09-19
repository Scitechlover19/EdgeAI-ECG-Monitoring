"""Unit tests for C++ Header Export module (P1-1).

Tests byte-for-byte fidelity between .tflite flatbuffer and exported C++ unsigned char[] array.
Governed by AGENTS.md §3 and Plan.md P1-1.
"""

from pathlib import Path
import re
import pytest

from src.models.export_header import (
    export_deployed_model_header,
    export_tflite_to_cpp_header,
)


def test_export_dummy_tflite_to_header(tmp_path: Path) -> None:
    """Test exporting arbitrary binary flatbuffer bytes to a C++ header."""
    dummy_bytes = bytes([0x1C, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4C, 0x33, 0xAA, 0xBB, 0xCC, 0xDD, 0x01, 0x02])
    dummy_tflite = tmp_path / "model.tflite"
    dummy_tflite.write_bytes(dummy_bytes)

    out_header = tmp_path / "model_data.h"
    bytes_exported = export_tflite_to_cpp_header(
        tflite_path=dummy_tflite,
        output_header_path=out_header,
        array_name="test_model_data",
        len_var_name="test_model_data_len",
    )

    assert bytes_exported == len(dummy_bytes)
    assert out_header.exists()

    header_text = out_header.read_text(encoding="utf-8")
    assert "#ifndef MODEL_DATA_H_" in header_text
    assert "#define MODEL_DATA_H_" in header_text
    assert "const unsigned int test_model_data_len = 14;" in header_text
    assert "const unsigned char test_model_data[] = {" in header_text

    # Extract hex bytes and verify byte-for-byte roundtrip
    hex_matches = re.findall(r"0x([0-9a-fA-F]{2})", header_text)
    reconstructed_bytes = bytes(int(h, 16) for h in hex_matches)
    assert reconstructed_bytes == dummy_bytes


def test_export_deployed_model_header(tmp_path: Path) -> None:
    """Test exporting the actual deployed INT8 student model to header."""
    model_path = Path("models/student_model_int8.tflite")
    if not model_path.exists():
        pytest.skip("models/student_model_int8.tflite not found")

    out_header = tmp_path / "student_model_int8.h"
    export_deployed_model_header(output_path=out_header)

    assert out_header.exists()
    header_content = out_header.read_text(encoding="utf-8")

    expected_size = model_path.stat().st_size
    assert f"const unsigned int g_student_model_data_len = {expected_size};" in header_content

    hex_matches = re.findall(r"0x([0-9a-fA-F]{2})", header_content)
    assert len(hex_matches) == expected_size
    reconstructed_bytes = bytes(int(h, 16) for h in hex_matches)
    assert reconstructed_bytes == model_path.read_bytes()


def test_export_missing_file_raises(tmp_path: Path) -> None:
    """Test FileNotFoundError is raised for non-existent input path."""
    with pytest.raises(FileNotFoundError):
        export_tflite_to_cpp_header(
            tflite_path=tmp_path / "missing.tflite",
            output_header_path=tmp_path / "missing.h",
        )
