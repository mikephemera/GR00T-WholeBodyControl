"""Client for the final-input/final-output C++ MotionBricks planner.

The request contains only four MuJoCo qpos frames and controller values.  The
C++ process owns canonicalization, FK/features, spring/clip targets, all three
networks, decoder post-processing and the final qpos blend.
"""

from __future__ import annotations

import struct
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np


QPOS_DIM = 36
CONTEXT_FRAMES = 4
MAX_OUTPUT_FRAMES = 64
_FRAME_HEADER = struct.Struct("<8sII")
_RESPONSE_HEADER = struct.Struct("<iii6dI")
_REQUEST_MAGIC = b"MBREQ1\0\0"
_RESPONSE_MAGIC = b"MBRES1\0\0"
_PROTOCOL_VERSION = 1


def _read_exact(stream, size: int, process: subprocess.Popen) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            code = process.poll()
            detail = ""
            if code is not None and process.stderr is not None:
                detail = process.stderr.read().decode(errors="replace").strip()
            raise RuntimeError(
                f"C++ MotionBricks planner closed its stream (code={code}): {detail}"
            )
        chunks.extend(chunk)
    return bytes(chunks)


def build_request_payload(
    context_qpos: np.ndarray,
    *,
    mode: int,
    movement_direction: float,
    facing_direction: float,
    random_seed: int,
    target_clip_seed: int = -1,
    allowed_pred_num_tokens: Sequence[int] | None = None,
    target_clip_name: str = "",
    specific_target: str | None = None,
) -> bytes:
    context = np.ascontiguousarray(context_qpos, dtype="<f4")
    if context.shape != (CONTEXT_FRAMES, QPOS_DIM):
        raise ValueError(
            f"C++ planner context must be [{CONTEXT_FRAMES},{QPOS_DIM}], got {context.shape}"
        )
    if not np.isfinite(context).all():
        raise ValueError("C++ planner context contains NaN or Inf")
    if not 0 <= random_seed <= 0xFFFFFFFF:
        raise ValueError("random_seed must fit uint32")
    allowed = tuple(allowed_pred_num_tokens or (0,) * 11)
    if len(allowed) != 11:
        raise ValueError("allowed_pred_num_tokens must have 11 entries")
    if specific_target is not None:
        if target_clip_name and target_clip_name != specific_target:
            raise ValueError("target_clip_name and specific_target disagree")
        target_clip_name = specific_target
    target = target_clip_name.encode("utf-8")
    if len(target) > 4096:
        raise ValueError("target_clip_name is too long")

    payload = bytearray(context.tobytes(order="C"))
    payload.extend(
        struct.pack(
            "<iffIi",
            int(mode),
            float(movement_direction),
            float(facing_direction),
            int(random_seed),
            int(target_clip_seed),
        )
    )
    payload.extend(struct.pack("<11q", *(int(value) for value in allowed)))
    payload.extend(struct.pack("<I", len(target)))
    payload.extend(target)
    return bytes(payload)


class CppPlanner:
    """Persistent C++ planner process using final qpos I/O."""

    def __init__(
        self,
        planner_cli: str | Path,
        asset_pack: str | Path,
        model_root: str | Path,
        *,
        device: str = "cpu",
        require_musa: bool = False,
        warmup: int = 1,
    ) -> None:
        self.planner_cli = Path(planner_cli).expanduser().resolve()
        self.asset_pack = Path(asset_pack).expanduser().resolve()
        self.model_root = Path(model_root).expanduser().resolve()
        if not self.planner_cli.is_file():
            raise FileNotFoundError(f"C++ planner CLI not found: {self.planner_cli}")
        if not self.asset_pack.is_file():
            raise FileNotFoundError(f"MotionBricks asset pack not found: {self.asset_pack}")
        for model in ("root_backbone.onnx", "pose_backbone.onnx", "vqvae_decoder.onnx"):
            if not (self.model_root / model).is_file():
                raise FileNotFoundError(f"MotionBricks ONNX model not found: {self.model_root / model}")
        if device not in {"cpu", "musa"} and not device.isdigit():
            raise ValueError("device must be cpu, musa, or a numeric MUSA device id")
        if require_musa and device == "cpu":
            raise ValueError("require_musa cannot be used with CPU")

        command = [
            str(self.planner_cli),
            "--asset-pack",
            str(self.asset_pack),
            "--model-root",
            str(self.model_root),
            "--device",
            device,
            "--warmup",
            str(warmup),
            "--serve",
        ]
        if require_musa:
            command.append("--require-musa")
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("failed to open C++ planner pipes")

    def plan(
        self,
        context_qpos: np.ndarray,
        *,
        mode: int,
        movement_direction: float,
        facing_direction: float,
        random_seed: int,
        target_clip_seed: int = -1,
        allowed_pred_num_tokens: Sequence[int] | None = None,
        target_clip_name: str = "",
        specific_target: str | None = None,
    ) -> tuple[np.ndarray, dict[str, int | float]]:
        if self.process.poll() is not None:
            _read_exact(self.process.stdout, 1, self.process)
        payload = build_request_payload(
            context_qpos,
            mode=mode,
            movement_direction=movement_direction,
            facing_direction=facing_direction,
            random_seed=random_seed,
            target_clip_seed=target_clip_seed,
            allowed_pred_num_tokens=allowed_pred_num_tokens,
            target_clip_name=target_clip_name,
            specific_target=specific_target,
        )
        frame = _FRAME_HEADER.pack(_REQUEST_MAGIC, _PROTOCOL_VERSION, len(payload)) + payload
        try:
            self.process.stdin.write(frame)
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError("failed to send request to C++ MotionBricks planner") from error

        header = _read_exact(self.process.stdout, _FRAME_HEADER.size, self.process)
        magic, version, response_size = _FRAME_HEADER.unpack(header)
        if magic != _RESPONSE_MAGIC or version != _PROTOCOL_VERSION:
            raise RuntimeError(
                f"invalid C++ planner response header: magic={magic!r}, version={version}"
            )
        maximum_size = _RESPONSE_HEADER.size + MAX_OUTPUT_FRAMES * QPOS_DIM * 4
        if response_size > maximum_size:
            raise RuntimeError(f"C++ planner response is too large: {response_size}")
        response = _read_exact(self.process.stdout, response_size, self.process)
        if len(response) < _RESPONSE_HEADER.size:
            raise RuntimeError("truncated C++ planner response")

        unpacked = _RESPONSE_HEADER.unpack_from(response)
        frames, tokens, clip = unpacked[:3]
        timings = unpacked[3:9]
        value_count = unpacked[9]
        if not 1 <= frames <= MAX_OUTPUT_FRAMES or value_count != frames * QPOS_DIM:
            raise RuntimeError(
                f"invalid final qpos response: frames={frames}, values={value_count}"
            )
        expected_size = _RESPONSE_HEADER.size + value_count * 4
        if len(response) != expected_size:
            raise RuntimeError(
                f"unexpected final qpos response length: {len(response)} != {expected_size}"
            )
        qpos = np.frombuffer(
            response, dtype="<f4", count=value_count, offset=_RESPONSE_HEADER.size
        ).copy().reshape(frames, QPOS_DIM)
        if not np.isfinite(qpos).all():
            raise RuntimeError("C++ planner returned NaN or Inf qpos")
        quaternion_error = np.max(np.abs(np.linalg.norm(qpos[:, 3:7], axis=1) - 1.0))
        if quaternion_error > 1.0e-3:
            raise RuntimeError(
                f"C++ planner returned non-unit root quaternion: max_error={quaternion_error}"
            )
        return qpos, {
            "frames": frames,
            "tokens": tokens,
            "clip": clip,
            "total_ms": timings[0],
            "feature_ms": timings[1],
            "root_ms": timings[2],
            "pose_ms": timings[3],
            "decoder_ms": timings[4],
            "postprocess_ms": timings[5],
        }

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        try:
            self.process.stdin.close()
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        if self.process.returncode not in (0, None):
            detail = self.process.stderr.read().decode(errors="replace").strip()
            raise RuntimeError(
                f"C++ MotionBricks planner exited with {self.process.returncode}: {detail}"
            )

    def __enter__(self) -> "CppPlanner":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()
