"""Portable golden-data recorder for Python to C++ inference bring-up.

Each recorded tensor is a raw, little-endian binary array accompanied by its
shape, dtype and semantic description in JSON.  This deliberately avoids
``torch.save``/pickle so that a small C++ reader can consume the files without
linking against Python or NumPy.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


class GoldenRecorder:
    """Write one self-contained directory per inference invocation.

    Recording is opt-in: when no output directory is supplied the methods are
    cheap no-ops.  Tensors are detached and copied to CPU immediately, which
    also makes this safe for asynchronous CUDA/MUSA execution.
    """

    SCHEMA_VERSION = 1

    def __init__(self, output_dir: str | Path | None, *, flush_each_record: bool = True):
        self.output_dir = Path(output_dir).expanduser() if output_dir else None
        self.flush_each_record = flush_each_record
        self._record_dir: Path | None = None
        self._record_manifest: dict[str, Any] | None = None
        self._next_record = 0
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = self.output_dir / "manifest.json"
            if manifest_path.exists():
                try:
                    root = json.loads(manifest_path.read_text())
                    if root.get("schema") != "motionbricks_golden":
                        raise ValueError(f"{manifest_path} is not a MotionBricks golden directory")
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Cannot parse existing golden manifest {manifest_path}") from exc
            else:
                self._write_root_manifest()
            records_dir = self.output_dir / "records"
            if records_dir.exists():
                indices = [
                    int(path.name.removeprefix("record_"))
                    for path in records_dir.iterdir()
                    if path.is_dir() and path.name.startswith("record_") and path.name[7:].isdigit()
                ]
                self._next_record = max(indices, default=-1) + 1

    @property
    def enabled(self) -> bool:
        return self.output_dir is not None

    @property
    def active(self) -> bool:
        return self.enabled and self._record_dir is not None

    def _write_root_manifest(self) -> None:
        if not self.enabled:
            return
        root = {
            "schema": "motionbricks_golden",
            "schema_version": self.SCHEMA_VERSION,
            "format": "raw little-endian tensor binaries + JSON manifests",
            "byte_order": "little",
            "created_unix": time.time(),
            "python": sys.version,
            "platform": platform.platform(),
            "records": [],
        }
        (self.output_dir / "manifest.json").write_text(json.dumps(root, indent=2) + "\n")

    def start_record(self, *, metadata: dict[str, Any] | None = None) -> bool:
        if not self.enabled:
            return False
        if self.active:
            self.finish_record()
        record_name = f"record_{self._next_record:06d}"
        self._next_record += 1
        self._record_dir = self.output_dir / "records" / record_name
        self._record_dir.mkdir(parents=True, exist_ok=False)
        self._record_manifest = {
            "schema": "motionbricks_golden_record",
            "schema_version": self.SCHEMA_VERSION,
            "record": record_name,
            "created_unix": time.time(),
            "metadata": _json_safe(metadata or {}),
            "tensors": [],
        }
        return True

    def record(self, name: str, value: Any, *, layer: str, semantic: str = "") -> None:
        """Save a tensor/array as ``name.bin`` and append its descriptor."""
        if not self.active or value is None:
            return
        array = _to_numpy(value)
        # C++ readers should never need to handle NumPy's native-endian marker.
        array = np.ascontiguousarray(array)
        if array.dtype.kind in "iu" and array.dtype.itemsize > 1:
            array = array.astype(array.dtype.newbyteorder("<"), copy=False)
        elif array.dtype.kind == "f" and array.dtype.itemsize in (2, 4, 8):
            array = array.astype(array.dtype.newbyteorder("<"), copy=False)
        elif array.dtype.kind == "c" and array.dtype.itemsize in (8, 16):
            array = array.astype(array.dtype.newbyteorder("<"), copy=False)
        safe_name = name.strip("/").replace("/", "__")
        if not safe_name:
            raise ValueError("Golden tensor name must not be empty")
        path = self._record_dir / f"{safe_name}.bin"
        array.tofile(path)
        entry = {
            "name": name,
            "file": str(path.relative_to(self._record_dir)),
            "layer": layer,
            "semantic": semantic,
            "dtype": _dtype_name(array.dtype),
            "shape": list(array.shape),
            "numel": int(array.size),
            "nbytes": int(array.nbytes),
        }
        self._record_manifest["tensors"].append(entry)
        if self.flush_each_record:
            self._write_record_manifest()

    def finish_record(self) -> None:
        if not self.active:
            return
        self._write_record_manifest()
        root_path = self.output_dir / "manifest.json"
        try:
            root = json.loads(root_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            self._write_root_manifest()
            root = json.loads(root_path.read_text())
        records = [r for r in root.get("records", []) if r.get("record") != self._record_manifest["record"]]
        records.append({
            "record": self._record_manifest["record"],
            "path": f"records/{self._record_manifest['record']}/record.json",
            "tensor_count": len(self._record_manifest["tensors"]),
        })
        root["records"] = records
        root_path.write_text(json.dumps(root, indent=2) + "\n")
        self._record_dir = None
        self._record_manifest = None

    def __enter__(self) -> "GoldenRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish_record()

    def _write_record_manifest(self) -> None:
        if self.active:
            (self._record_dir / "record.json").write_text(json.dumps(self._record_manifest, indent=2) + "\n")


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    # Import torch lazily: importing this helper must remain possible in a
    # minimal C++-conversion/documentation environment.
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().to("cpu").numpy()
    except ImportError:
        pass
    if np.isscalar(value):
        return np.asarray(value)
    return np.asarray(value)


def _dtype_name(dtype: np.dtype) -> str:
    # ``<f4`` is unambiguous to a C++ reader and documents the storage format.
    dtype = np.dtype(dtype).newbyteorder("<")
    return dtype.str


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    try:
        return value.item()
    except AttributeError:
        return str(value)
