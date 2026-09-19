"""C++ Header Export utility for TensorFlow Lite FlatBuffer (P1-1).

Converts exported INT8 `.tflite` models into C/C++ byte arrays (`unsigned char[]`)
suitable for embedding in simulated or bare-metal microcontroller firmware images.
Governed by PRD.md (FR-8), Architecture.md §5.3/§9, and AGENTS.md §1.
"""

from pathlib import Path
from typing import Optional, Union
import numpy as np

from src.pipeline.config import load_config, setup_logging

logger = setup_logging()


def export_tflite_to_cpp_header(
    tflite_path: Union[str, Path],
    output_header_path: Union[str, Path],
    array_name: str = "g_student_model_data",
    len_var_name: Optional[str] = None,
    line_bytes: int = 12,
) -> int:
    """Convert a .tflite flatbuffer file to a C/C++ header (.h) with an unsigned char array.

    Args:
        tflite_path: Path to existing .tflite flatbuffer file.
        output_header_path: Destination path for generated .h header file.
        array_name: C-identifier for the unsigned char byte array.
        len_var_name: Optional identifier for length constant (default: array_name + "_len").
        line_bytes: Number of hex bytes to write per line in the array definition.

    Returns:
        Total number of bytes exported.

    Raises:
        FileNotFoundError: If tflite_path does not exist.
        ValueError: If file is empty or parameters are invalid.
    """
    in_path = Path(tflite_path)
    out_path = Path(output_header_path)

    if not in_path.is_file():
        raise FileNotFoundError(f"TFLite model file not found at '{in_path}'.")

    model_bytes = in_path.read_bytes()
    num_bytes = len(model_bytes)

    if num_bytes == 0:
        raise ValueError(f"TFLite file '{in_path}' is empty (0 bytes).")

    if len_var_name is None:
        len_var_name = f"{array_name}_len"

    header_guard = f"{out_path.stem.upper()}_H_"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "// Automatically generated C++ header from TensorFlow Lite FlatBuffer.",
        f"// Source file : {in_path.name}",
        f"// Model size  : {num_bytes:,} bytes ({num_bytes / 1024.0:.2f} KB) [MEASURED]",
        "",
        f"#ifndef {header_guard}",
        f"#define {header_guard}",
        "",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
        "",
        f"const unsigned int {len_var_name} = {num_bytes};",
        f"alignas(16) const unsigned char {array_name}[] = {{",
    ]

    # Format bytes in hex lines (e.g. 0x1c, 0x00, ...)
    for i in range(0, num_bytes, line_bytes):
        chunk = model_bytes[i : i + line_bytes]
        hex_chunk = ", ".join(f"0x{b:02x}" for b in chunk)
        if i + line_bytes < num_bytes:
            lines.append(f"    {hex_chunk},")
        else:
            lines.append(f"    {hex_chunk}")

    lines.extend([
        "};",
        "",
        "#ifdef __cplusplus",
        "}",
        "#endif",
        "",
        f"#endif  // {header_guard}",
        "",
    ])

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(
        f"Exported {num_bytes:,} bytes from '{in_path}' to C++ header '{out_path}'."
    )
    return num_bytes


def export_deployed_model_header(
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Export the default deployed INT8 model to a C++ header file.

    Args:
        output_path: Optional custom output path. Defaults to models/student_model_int8.h.

    Returns:
        Path to the exported header file.
    """
    config = load_config()
    models_dir = Path(config.paths.models_dir)
    tflite_path = models_dir / "student_model_int8.tflite"

    if not tflite_path.exists():
        tflite_path = models_dir / "student_model.tflite"

    out_path = Path(output_path) if output_path is not None else models_dir / "student_model_int8.h"

    export_tflite_to_cpp_header(
        tflite_path=tflite_path,
        output_header_path=out_path,
        array_name="g_student_model_data",
        len_var_name="g_student_model_data_len",
    )
    return out_path


if __name__ == "__main__":
    exported = export_deployed_model_header()
    print(f"Header successfully generated at: {exported}")
