"""Runtime device selection for MotionBricks inference."""

from __future__ import annotations

import torch


def _musa_is_available() -> bool:
    """Load torch_musa when installed and report whether a MUSA device is usable."""
    try:
        import torch_musa  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch, "musa") and torch.musa.is_available()


def resolve_inference_device(requested: str | torch.device = "auto") -> torch.device:
    """Resolve and activate a CUDA, MUSA, or CPU inference device.

    ``auto`` prefers MUSA so the unmodified official demo command uses the
    Moore Threads GPU when both the runtime and a device are present.
    """
    requested_str = str(requested).strip().lower()
    if requested_str == "auto":
        if _musa_is_available():
            requested_str = "musa:0"
        elif torch.cuda.is_available():
            requested_str = "cuda:0"
        else:
            requested_str = "cpu"

    try:
        device = torch.device(requested_str)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            f"Invalid inference device {requested!r}; use auto, musa[:N], cuda[:N], or cpu"
        ) from exc

    if device.type == "musa":
        if not _musa_is_available():
            raise RuntimeError(
                "MUSA inference was requested, but torch_musa is not installed or no MUSA device is available"
            )
        device_index = 0 if device.index is None else device.index
        if device_index >= torch.musa.device_count():
            raise RuntimeError(
                f"MUSA device index {device_index} is out of range; "
                f"found {torch.musa.device_count()} device(s)"
            )
        torch.musa.set_device(device_index)
        return torch.device("musa", device_index)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA inference was requested, but CUDA is not available")
        device_index = 0 if device.index is None else device.index
        if device_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {device_index} is out of range; "
                f"found {torch.cuda.device_count()} device(s)"
            )
        torch.cuda.set_device(device_index)
        return torch.device("cuda", device_index)

    if device.type != "cpu":
        raise ValueError(
            f"Unsupported inference device type {device.type!r}; use auto, musa[:N], cuda[:N], or cpu"
        )
    return torch.device("cpu")


def describe_inference_device(device: torch.device) -> str:
    """Return a concise user-facing description of the selected device."""
    if device.type == "musa":
        return f"{device} ({torch.musa.get_device_name(device.index or 0)})"
    if device.type == "cuda":
        return f"{device} ({torch.cuda.get_device_name(device.index or 0)})"
    return str(device)
