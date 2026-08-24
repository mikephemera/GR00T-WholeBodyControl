#!/usr/bin/env python3
"""Play G1 reference clips or a recorded qpos NPZ in MuJoCo.

This viewer reads the cached MuJoCo qpos poses directly.  It does not create
the MotionBricks neural networks or run planner inference.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


MOTIONBRICKS_ROOT = Path(__file__).resolve().parents[1]
if str(MOTIONBRICKS_ROOT) not in sys.path:
    sys.path.insert(0, str(MOTIONBRICKS_ROOT))

# pyGLFW selects its native library when it is imported.  Match the existing
# interactive demos by preferring X11 when an XWayland display is available.
if platform.system() == "Linux" and os.environ.get("DISPLAY"):
    os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")

import glfw  # noqa: E402
import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from motionbricks.motion_backbone.demo.clips import clip_holder_G1  # noqa: E402


SOURCE_FPS = 30.0
SPEED_LEVELS = (0.5, 1.0, 2.0)
CLIP_NAMES = tuple(clip_holder_G1.CLIPS)


@dataclass(frozen=True)
class ClipLibrary:
    names: tuple[str, ...]
    qpos: np.ndarray
    lengths: np.ndarray
    fps: float = SOURCE_FPS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clips-ckpt",
        type=Path,
        default=MOTIONBRICKS_ROOT / "out/G1-clip.ckpt",
        help="checkpoint containing mujoco_qpos and num_frames_per_clip",
    )
    parser.add_argument(
        "--qpos-npz",
        type=Path,
        default=None,
        help="recorded qpos NPZ to play instead of --clips-ckpt",
    )
    parser.add_argument(
        "--xml",
        type=Path,
        default=MOTIONBRICKS_ROOT / "assets/skeletons/g1/scene_29dof.xml",
        help="MuJoCo scene for the 29-DoF G1",
    )
    parser.add_argument(
        "--initial-clip",
        choices=CLIP_NAMES,
        default="idle",
        help="clip selected when the viewer starts",
    )
    return parser.parse_args()


def _require_tensor(state_dict: Mapping[str, object], key: str) -> torch.Tensor:
    value = state_dict.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"clip checkpoint is missing tensor {key!r}")
    return value


def load_clip_library(checkpoint: Path, expected_nq: int) -> ClipLibrary:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"clip checkpoint not found: {checkpoint}")

    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, Mapping):
        raise ValueError("clip checkpoint must contain a tensor mapping")

    qpos = _require_tensor(state_dict, "mujoco_qpos")
    lengths = _require_tensor(state_dict, "num_frames_per_clip")

    if qpos.ndim != 3:
        raise ValueError(f"mujoco_qpos must have shape [clips,frames,qpos], got {tuple(qpos.shape)}")
    if qpos.shape[0] != len(CLIP_NAMES):
        raise ValueError(
            f"checkpoint contains {qpos.shape[0]} clips, but clip_holder_G1 defines "
            f"{len(CLIP_NAMES)}"
        )
    if qpos.shape[2] != expected_nq:
        raise ValueError(
            f"checkpoint qpos width is {qpos.shape[2]}, but the MuJoCo model expects {expected_nq}"
        )
    if lengths.ndim != 1 or lengths.shape[0] != len(CLIP_NAMES):
        raise ValueError(
            "num_frames_per_clip must have one entry per clip; "
            f"got shape {tuple(lengths.shape)}"
        )
    if not torch.isfinite(qpos).all().item():
        raise ValueError("mujoco_qpos contains non-finite values")

    lengths_i64 = lengths.to(dtype=torch.int64)
    if not torch.equal(lengths, lengths_i64.to(dtype=lengths.dtype)):
        raise ValueError("num_frames_per_clip must contain integer frame counts")
    if (lengths_i64 <= 0).any().item() or (lengths_i64 > qpos.shape[1]).any().item():
        raise ValueError(
            f"clip lengths must be in [1,{qpos.shape[1]}], got {lengths_i64.tolist()}"
        )

    return ClipLibrary(
        names=CLIP_NAMES,
        qpos=np.ascontiguousarray(qpos.detach().cpu().numpy()),
        lengths=np.ascontiguousarray(lengths_i64.cpu().numpy()),
    )


def load_qpos_recording(recording: Path, expected_nq: int) -> ClipLibrary:
    if not recording.is_file():
        raise FileNotFoundError(f"qpos recording not found: {recording}")

    with np.load(recording, allow_pickle=False) as data:
        if "qpos" not in data:
            raise ValueError("qpos recording is missing the 'qpos' array")
        qpos = np.asarray(data["qpos"])
        if qpos.ndim != 2 or qpos.shape[0] == 0 or qpos.shape[1] != expected_nq:
            raise ValueError(
                f"recorded qpos must have shape [frames,{expected_nq}], got {qpos.shape}"
            )
        if not np.isfinite(qpos).all():
            raise ValueError("recorded qpos contains non-finite values")

        quaternion_norms = np.linalg.norm(qpos[:, 3:7], axis=1)
        if not np.allclose(quaternion_norms, 1.0, atol=1e-3, rtol=0.0):
            raise ValueError("recorded qpos contains invalid root quaternions")

        fps_value = np.asarray(data["fps"]) if "fps" in data else np.asarray(SOURCE_FPS)
        if fps_value.size != 1:
            raise ValueError(f"recording fps must be a scalar, got shape {fps_value.shape}")
        fps = float(fps_value.item())
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"recording fps must be finite and positive, got {fps}")

        name = recording.stem
        if "mode_name" in data:
            mode_name = np.asarray(data["mode_name"])
            if mode_name.size == 1 and mode_name.dtype.kind in {"U", "S"}:
                name = str(mode_name.item())

    return ClipLibrary(
        names=(name,),
        qpos=np.ascontiguousarray(qpos[None]),
        lengths=np.asarray([qpos.shape[0]], dtype=np.int64),
        fps=fps,
    )


class PlaybackController:
    """Thread-safe playback state shared with MuJoCo's key callback."""

    def __init__(self, library: ClipLibrary, initial_clip: str) -> None:
        self._library = library
        self._clip_index = library.names.index(initial_clip)
        self._speed_index = SPEED_LEVELS.index(1.0)
        self._frame_index = 0
        self._revision = 0
        self._lock = threading.Lock()

    def _status_locked(self) -> str:
        clip_name = self._library.names[self._clip_index]
        frame_count = int(self._library.lengths[self._clip_index])
        speed = SPEED_LEVELS[self._speed_index]
        return (
            f"clip={self._clip_index + 1}/{len(self._library.names)} "
            f"name={clip_name} frames={frame_count} speed={speed:.1f}x"
        )

    def print_status(self) -> None:
        with self._lock:
            status = self._status_locked()
        print(status, flush=True)

    def on_key(self, key: int) -> None:
        recognized = True
        with self._lock:
            changed = False
            if key == glfw.KEY_LEFT:
                self._clip_index = (self._clip_index - 1) % len(self._library.names)
                self._frame_index = 0
                changed = True
            elif key == glfw.KEY_RIGHT:
                self._clip_index = (self._clip_index + 1) % len(self._library.names)
                self._frame_index = 0
                changed = True
            elif key == glfw.KEY_UP:
                new_speed_index = min(self._speed_index + 1, len(SPEED_LEVELS) - 1)
                changed = new_speed_index != self._speed_index
                self._speed_index = new_speed_index
            elif key == glfw.KEY_DOWN:
                new_speed_index = max(self._speed_index - 1, 0)
                changed = new_speed_index != self._speed_index
                self._speed_index = new_speed_index
            else:
                recognized = False

            if changed:
                self._revision += 1
            status = self._status_locked() if recognized else ""

        if recognized:
            print(status, flush=True)

    def revision(self) -> int:
        with self._lock:
            return self._revision

    def next_frame(self) -> tuple[int, int, float, int]:
        with self._lock:
            clip_index = self._clip_index
            frame_index = self._frame_index
            speed = SPEED_LEVELS[self._speed_index]
            revision = self._revision
            frame_count = int(self._library.lengths[clip_index])
            self._frame_index = (frame_index + 1) % frame_count
        return clip_index, frame_index, speed, revision


def _prefer_x11_glfw_on_linux() -> None:
    if platform.system() != "Linux" or not os.environ.get("DISPLAY"):
        return
    if hasattr(glfw, "PLATFORM") and hasattr(glfw, "PLATFORM_X11"):
        glfw.init_hint(glfw.PLATFORM, glfw.PLATFORM_X11)


def run_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    library: ClipLibrary,
    initial_clip: str,
) -> None:
    controller = PlaybackController(library, initial_clip)
    initial_clip_index = library.names.index(initial_clip)
    data.qpos[:] = library.qpos[initial_clip_index, 0]
    data.qvel.fill(0.0)
    mujoco.mj_forward(model, data)

    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_id < 0:
        raise ValueError("MuJoCo model does not contain a pelvis body")

    print("Controls: Left/Right=select clip, Up/Down=select 0.5x/1.0x/2.0x speed")
    controller.print_status()

    with mujoco.viewer.launch_passive(model, data, key_callback=controller.on_key) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = pelvis_id
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 145.0
        viewer.cam.elevation = -18.0

        next_frame_at = time.monotonic()
        observed_revision = controller.revision()
        while viewer.is_running():
            now = time.monotonic()
            revision = controller.revision()
            if revision != observed_revision:
                next_frame_at = now
                observed_revision = revision

            if now < next_frame_at:
                viewer.sync()
                time.sleep(min(0.005, next_frame_at - now))
                continue

            clip_index, frame_index, speed, observed_revision = controller.next_frame()
            with viewer.lock():
                data.qpos[:] = library.qpos[clip_index, frame_index]
                data.qvel.fill(0.0)
                mujoco.mj_forward(model, data)
            viewer.sync()

            period = 1.0 / (library.fps * speed)
            next_frame_at += period
            if next_frame_at < time.monotonic() - period:
                next_frame_at = time.monotonic() + period


def main() -> None:
    args = parse_args()
    xml_path = args.xml.expanduser().resolve()
    if not xml_path.is_file():
        raise FileNotFoundError(f"MuJoCo scene not found: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    if args.qpos_npz is not None:
        recording = args.qpos_npz.expanduser().resolve()
        library = load_qpos_recording(recording, expected_nq=model.nq)
        initial_clip = library.names[0]
        print(f"Loaded qpos recording: {recording} ({library.lengths[0]} frames at {library.fps:g} Hz)")
    else:
        checkpoint = args.clips_ckpt.expanduser().resolve()
        library = load_clip_library(checkpoint, expected_nq=model.nq)
        initial_clip = args.initial_clip
    _prefer_x11_glfw_on_linux()
    run_viewer(model, data, library, initial_clip)


if __name__ == "__main__":
    main()
