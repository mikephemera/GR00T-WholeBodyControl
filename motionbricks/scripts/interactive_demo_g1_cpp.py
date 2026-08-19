#!/usr/bin/env python3
"""View the final-qpos C++ MotionBricks planner with continuous replanning.

Python owns only keyboard/camera controls, the four-frame playback context and
MuJoCo visualization. Every plan is produced by the C++ final interface:

    qpos[4,36] + control -> C++ MotionBricksPlanner -> qpos[N,36]

No Python MotionBricks model, feature pipeline, spring model or qpos converter
is created by this script.
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

if platform.system() == "Linux" and os.environ.get("DISPLAY"):
    os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")

import mujoco
import numpy as np

from motionbricks.cpp_planner import CONTEXT_FRAMES, QPOS_DIM, CppPlanner


MOTIONBRICKS_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BFM_ROOT = Path(
    os.environ.get("BFM_CONTROLLER_ROOT", "/home/michael/Work-syncfree/bfm_controller")
)

CLIP_NAMES = (
    "idle",
    "slow_walk",
    "walk",
    "hand_crawling",
    "walk_boxing",
    "elbow_crawling",
    "stealth_walk",
    "injured_walk",
    "walk_stealth",
    "walk_happy_dance",
    "walk_zombie",
    "walk_gun",
    "walk_scared",
    "walk_left",
    "walk_right",
)
MODE_KEYS = {
    "v": 1,
    "z": 3,
    "x": 4,
    "b": 5,
    "r": 6,
    "t": 7,
    "c": 8,
    "e": 9,
    "f": 10,
    "g": 11,
    "q": 12,
}
CONTROL_KEYS = tuple("wasd") + tuple(MODE_KEYS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--planner-cli",
        "--cpp-cli",
        dest="planner_cli",
        type=Path,
        default=MOTIONBRICKS_ROOT / "cpp/build/motionbricks_planner_cli",
    )
    parser.add_argument(
        "--asset-pack",
        type=Path,
        default=DEFAULT_BFM_ROOT / "assets/motionbricks_release/motionbricks_assets.mbpack",
    )
    parser.add_argument(
        "--model-root",
        "--onnx-dir",
        dest="model_root",
        type=Path,
        default=DEFAULT_BFM_ROOT / "assets/motionbricks_release/onnx",
    )
    parser.add_argument(
        "--xml",
        "--humanoid_xml",
        dest="xml",
        type=Path,
        default=MOTIONBRICKS_ROOT / "assets/skeletons/g1/scene_29dof.xml",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu, musa, or a numeric MUSA device id",
    )
    parser.add_argument("--require-musa", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--replan-frames",
        type=int,
        default=16,
        help="playback frames between final-qpos C++ planner calls (original demo default: 16)",
    )
    parser.add_argument(
        "--replan-on-control-change",
        action="store_true",
        help="schedule an earlier asynchronous handoff on key/mode change",
    )
    parser.add_argument(
        "--viewer-replan-mode",
        choices=("async", "sync"),
        default="async",
        help="async keeps rendering during C++ planning; sync is the blocking diagnostic baseline",
    )
    parser.add_argument(
        "--replan-lead-frames",
        type=int,
        default=0,
        help="async planning lead; zero estimates it from measured planner wall time",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--mode", type=int, default=2, help="headless fixed mode")
    parser.add_argument("--movement-direction", type=float, default=0.0)
    parser.add_argument("--facing-direction", type=float, default=0.0)
    parser.add_argument("--target-clip", default="")
    parser.add_argument("--segments", type=int, default=3, help="headless replan count")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--has_viewer",
        type=int,
        choices=(0, 1),
        default=None,
        help="compatibility alias: 0 selects headless and 1 selects viewer",
    )
    parser.add_argument("--max_steps", type=int, default=0, help="zero runs until viewer closes")
    parser.add_argument("--max-viewer-seconds", type=float, default=0.0)
    parser.add_argument("--lookat-movement-direction", action="store_true")
    args = parser.parse_args()
    if args.has_viewer is not None:
        args.headless = not bool(args.has_viewer)
    if args.replan_frames < 1 or args.segments < 1:
        parser.error("--replan-frames and --segments must be positive")
    if args.replan_lead_frames < 0:
        parser.error("--replan-lead-frames cannot be negative")
    if args.fps <= 0.0 or args.speed <= 0.0:
        parser.error("--fps and --speed must be positive")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    if not 0 <= args.seed <= 0xFFFFFFFF:
        parser.error("--seed must fit uint32")
    if not 0 <= args.mode < len(CLIP_NAMES):
        parser.error(f"--mode must be in [0,{len(CLIP_NAMES) - 1}]")
    if args.require_musa and args.device == "cpu":
        parser.error("--require-musa cannot be used with --device cpu")
    return args


def _prefer_x11_glfw() -> None:
    if platform.system() != "Linux" or not os.environ.get("DISPLAY"):
        return
    try:
        import glfw

        if hasattr(glfw, "PLATFORM") and hasattr(glfw, "PLATFORM_X11"):
            glfw.init_hint(glfw.PLATFORM, glfw.PLATFORM_X11)
    except (AttributeError, ImportError):
        pass


class KeyboardHandler:
    def __init__(self) -> None:
        from pynput import keyboard

        self._pressed: set[str] = set()
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.daemon = True
        self._listener.start()
        if not self._listener.is_alive():
            raise RuntimeError("pynput keyboard listener did not start")

    def _on_press(self, key, *_args) -> None:
        char = getattr(key, "char", None)
        if char:
            self._pressed.add(char.lower())

    def _on_release(self, key, *_args) -> None:
        char = getattr(key, "char", None)
        if char:
            self._pressed.discard(char.lower())

    def snapshot(self) -> dict[str, bool]:
        return {key: key in self._pressed for key in CONTROL_KEYS}

    def close(self) -> None:
        self._listener.stop()


def camera_angle(lookat, azimuth: float, elevation: float, distance: float) -> float:
    """Return the camera direction in the MuJoCo XY plane (zero is +X)."""
    lookat = np.asarray(lookat, dtype=np.float64)
    azimuth_radians = -math.radians(azimuth) - math.pi / 2.0
    elevation_radians = -math.radians(elevation)
    camera_position = lookat + distance * np.array(
        [
            math.cos(elevation_radians) * math.sin(azimuth_radians),
            math.cos(elevation_radians) * math.cos(azimuth_radians),
            math.sin(elevation_radians),
        ]
    )
    direction = (lookat - camera_position) * np.array([1.0, 1.0, 0.0])
    norm = float(np.linalg.norm(direction))
    return 0.0 if norm < 1.0e-6 else math.atan2(float(direction[1]), float(direction[0]))


def control_from_keys(
    active: dict[str, bool],
    camera_heading: float,
    lookat_movement_direction: bool,
) -> tuple[int, float, float]:
    moving = any(active[key] for key in "wasd")
    mode = 2 if moving else 0
    for key, candidate in MODE_KEYS.items():
        if active[key]:
            mode = candidate
            break
    if mode == 0:
        return 0, 0.0, 0.0

    forward = float(active["w"]) - float(active["s"])
    left = float(active["a"]) - float(active["d"])
    relative_heading = math.atan2(left, forward) if moving else 0.0
    movement = camera_heading + relative_heading
    movement = math.atan2(math.sin(movement), math.cos(movement))
    facing = movement if lookat_movement_direction else camera_heading
    return mode, movement, facing


def control_changed(
    current: tuple[int, float, float],
    previous: tuple[int, float, float],
    angle_epsilon: float = 0.05,
) -> bool:
    def angle_delta(first: float, second: float) -> float:
        return abs(math.atan2(math.sin(first - second), math.cos(first - second)))

    return (
        current[0] != previous[0]
        or angle_delta(current[1], previous[1]) > angle_epsilon
        or angle_delta(current[2], previous[2]) > angle_epsilon
    )


def take_context(plan: np.ndarray, cursor: int) -> np.ndarray:
    """Match get_context_mujoco_qpos(): current frame and the next three."""
    if len(plan) == 0:
        return np.zeros((CONTEXT_FRAMES, QPOS_DIM), dtype="<f4")
    indices = [max(0, min(cursor + offset, len(plan) - 1)) for offset in range(CONTEXT_FRAMES)]
    return np.ascontiguousarray(plan[indices], dtype="<f4")


def estimate_replan_lead_frames(
    wall_ms: float,
    period: float,
    replan_frames: int,
    configured_lead: int,
) -> int:
    """Leave enough playback frames to hide one measured planner call."""
    if configured_lead > 0:
        requested = configured_lead
    else:
        requested = math.ceil(max(0.0, wall_ms) / (period * 1000.0)) + 3
    return max(1, min(requested, max(1, replan_frames - 1)))


def latest_full_context_cursor(plan: np.ndarray) -> int:
    return max(0, len(plan) - CONTEXT_FRAMES)


@dataclass
class AsyncPlanRequest:
    future: Future
    context: np.ndarray
    control: tuple[int, float, float]
    handoff_cursor: int
    random_seed: int
    reason: str
    submitted_at: float
    result: tuple[np.ndarray, dict[str, int | float], float] | None = None
    superseded: bool = False
    superseded_reported: bool = False
    missed_reported: bool = False


def plan_with_wall_time(
    planner: CppPlanner,
    context: np.ndarray,
    control: tuple[int, float, float],
    random_seed: int,
    target_clip_name: str,
) -> tuple[np.ndarray, dict[str, int | float], float]:
    started = time.monotonic()
    plan, metadata = planner.plan(
        context,
        mode=control[0],
        movement_direction=control[1],
        facing_direction=control[2],
        random_seed=random_seed,
        target_clip_name=target_clip_name,
    )
    return plan, metadata, (time.monotonic() - started) * 1000.0


def validate_qpos(model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray) -> None:
    if model.nq != QPOS_DIM:
        raise ValueError(f"viewer XML must have nq={QPOS_DIM}, got {model.nq}")
    if qpos.ndim != 2 or qpos.shape[1] != QPOS_DIM or not np.isfinite(qpos).all():
        raise ValueError(f"invalid final C++ qpos: shape={qpos.shape}")
    quaternion_error = np.max(np.abs(np.linalg.norm(qpos[:, 3:7], axis=1) - 1.0))
    if quaternion_error > 1.0e-3:
        raise ValueError(f"invalid final C++ root quaternion: max_error={quaternion_error}")
    for frame in qpos:
        data.qpos[:] = frame
        data.qvel.fill(0.0)
        mujoco.mj_forward(model, data)
        if not np.isfinite(data.xpos).all():
            raise RuntimeError("MuJoCo FK produced non-finite positions from final C++ qpos")


def make_planner(args: argparse.Namespace) -> CppPlanner:
    return CppPlanner(
        args.planner_cli,
        args.asset_pack,
        args.model_root,
        device=args.device,
        require_musa=args.require_musa,
        warmup=args.warmup,
    )


def run_headless(args: argparse.Namespace, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    context = np.zeros((CONTEXT_FRAMES, QPOS_DIM), dtype="<f4")
    with make_planner(args) as planner:
        for segment in range(args.segments):
            plan, metadata = planner.plan(
                context,
                mode=args.mode,
                movement_direction=args.movement_direction,
                facing_direction=args.facing_direction,
                random_seed=(args.seed + segment) & 0xFFFFFFFF,
                target_clip_name=args.target_clip,
            )
            validate_qpos(model, data, plan)
            consumed = min(args.replan_frames, len(plan) - 1)
            context = take_context(plan, consumed)
            print(
                f"replan={segment + 1} input=[4,36] output={list(plan.shape)} "
                f"consumed={consumed} mode={args.mode}:{CLIP_NAMES[args.mode]} "
                f"tokens={metadata['tokens']} clip={metadata['clip']} "
                f"planner_ms={metadata['total_ms']:.2f}"
            )
    print(f"final-qpos C++ continuous-replan headless PASS: segments={args.segments}")


def run_viewer(args: argparse.Namespace, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    from mujoco.viewer import launch_passive

    planner = make_planner(args)
    keyboard = KeyboardHandler()
    executor = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="motionbricks-planner")
        if args.viewer_replan_mode == "async"
        else None
    )
    plan = np.empty((0, QPOS_DIM), dtype=np.float32)
    cursor = 0
    frames_since_replan = 0
    replan_count = 0
    played_steps = 0
    seed = args.seed
    last_planned_control = (0, 0.0, 0.0)
    period = 1.0 / (args.fps * args.speed)
    pending: AsyncPlanRequest | None = None
    async_submitted = 0
    async_discarded = 0
    async_missed = 0
    last_display_at: float | None = None
    maximum_display_gap_ms = 0.0
    late_display_gaps = 0
    try:
        initial_started = time.monotonic()
        plan, metadata = planner.plan(
            np.zeros((CONTEXT_FRAMES, QPOS_DIM), dtype="<f4"),
            mode=0,
            movement_direction=0.0,
            facing_direction=0.0,
            random_seed=seed,
        )
        initial_wall_ms = (time.monotonic() - initial_started) * 1000.0
        validate_qpos(model, data, plan)
        replan_lead = estimate_replan_lead_frames(
            initial_wall_ms,
            period,
            args.replan_frames,
            args.replan_lead_frames,
        )
        replan_count = 1
        print(
            f"replan=1 input=[4,36] output={list(plan.shape)} mode=0:idle "
            f"tokens={metadata['tokens']} clip={metadata['clip']} "
            f"planner_ms={metadata['total_ms']:.2f} wall_ms={initial_wall_ms:.2f} "
            f"viewer_mode={args.viewer_replan_mode} lead_frames={replan_lead}"
        )

        with launch_passive(model, data) as viewer:
            pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = pelvis
            viewer.cam.distance = 2.5
            viewer.cam.azimuth = 145.0
            viewer.cam.elevation = -18.0
            print(
                "viewer final C++ I/O: WASD=move, V=slow walk, Z/X/B/R/T/C/E/F/G/Q="
                "style modes, mouse=rotate camera, close window=exit"
            )
            started = time.monotonic()
            next_frame_at = started
            while viewer.is_running():
                now = time.monotonic()
                if args.max_viewer_seconds > 0.0 and now - started >= args.max_viewer_seconds:
                    break
                if args.max_steps > 0 and played_steps >= args.max_steps:
                    break
                if now < next_frame_at:
                    viewer.sync()
                    time.sleep(min(0.005, next_frame_at - now))
                    continue

                heading = camera_angle(
                    viewer.cam.lookat,
                    viewer.cam.azimuth,
                    viewer.cam.elevation,
                    viewer.cam.distance,
                )
                control = control_from_keys(
                    keyboard.snapshot(), heading, args.lookat_movement_direction
                )

                if executor is not None:
                    if pending is not None:
                        if control_changed(control, pending.control):
                            pending.superseded = True
                            if not pending.superseded_reported:
                                print(
                                    f"replan-supersede handoff_cursor={pending.handoff_cursor} "
                                    f"old_mode={pending.control[0]} new_mode={control[0]}"
                                )
                                pending.superseded_reported = True

                        if pending.result is None and pending.future.done():
                            pending.result = pending.future.result()
                            validate_qpos(model, data, pending.result[0])
                            if args.replan_lead_frames == 0:
                                replan_lead = max(
                                    replan_lead,
                                    estimate_replan_lead_frames(
                                        pending.result[2],
                                        period,
                                        args.replan_frames,
                                        0,
                                    ),
                                )

                        if pending.superseded and pending.result is not None:
                            print(
                                f"replan-discard reason=control-changed "
                                f"handoff_cursor={pending.handoff_cursor} "
                                f"wall_ms={pending.result[2]:.2f}"
                            )
                            pending = None
                            async_discarded += 1
                        elif cursor > pending.handoff_cursor:
                            if pending.result is not None:
                                print(
                                    f"replan-discard reason=missed-handoff "
                                    f"handoff_cursor={pending.handoff_cursor} cursor={cursor} "
                                    f"wall_ms={pending.result[2]:.2f}"
                                )
                                pending = None
                                async_discarded += 1
                            elif not pending.missed_reported:
                                print(
                                    f"replan-miss handoff_cursor={pending.handoff_cursor} "
                                    f"cursor={cursor}; continuing old qpos buffer"
                                )
                                pending.missed_reported = True
                                async_missed += 1
                        elif cursor == pending.handoff_cursor:
                            if pending.result is not None:
                                new_plan, new_metadata, wall_ms = pending.result
                                old_previous = plan[max(0, cursor - 1)]
                                boundary_linf = float(
                                    np.max(np.abs(new_plan[0] - old_previous))
                                )
                                natural_linf = float(
                                    np.max(np.abs(pending.context[0] - old_previous))
                                )
                                context_linf = float(
                                    np.max(np.abs(new_plan[0] - pending.context[0]))
                                )
                                plan = new_plan
                                cursor = 0
                                frames_since_replan = 0
                                last_planned_control = pending.control
                                replan_count += 1
                                print(
                                    f"replan={replan_count} handoff=exact input=[4,36] "
                                    f"output={list(plan.shape)} "
                                    f"mode={pending.control[0]}:{CLIP_NAMES[pending.control[0]]} "
                                    f"movement={pending.control[1]:+.3f} "
                                    f"facing={pending.control[2]:+.3f} "
                                    f"tokens={new_metadata['tokens']} clip={new_metadata['clip']} "
                                    f"planner_ms={new_metadata['total_ms']:.2f} "
                                    f"wall_ms={wall_ms:.2f} lead_frames={replan_lead} "
                                    f"boundary_linf={boundary_linf:.5f} "
                                    f"natural_linf={natural_linf:.5f} "
                                    f"context_linf={context_linf:.5f}"
                                )
                                pending = None
                            elif not pending.missed_reported:
                                print(
                                    f"replan-miss handoff_cursor={pending.handoff_cursor} "
                                    "result is not ready; continuing old qpos buffer"
                                )
                                pending.missed_reported = True
                                async_missed += 1

                    if pending is None:
                        latest_context = latest_full_context_cursor(plan)
                        nominal_handoff = min(args.replan_frames, latest_context)
                        changed = control_changed(control, last_planned_control)
                        change_due = args.replan_on_control_change and changed
                        interval_threshold = max(0, nominal_handoff - replan_lead)
                        interval_due = cursor >= interval_threshold
                        if change_due or interval_due:
                            if change_due:
                                handoff_cursor = min(
                                    len(plan) - 1, cursor + replan_lead
                                )
                                reason = "control"
                            elif cursor <= interval_threshold:
                                handoff_cursor = nominal_handoff
                                reason = "interval"
                            else:
                                handoff_cursor = min(
                                    len(plan) - 1, cursor + replan_lead
                                )
                                reason = "late-interval"
                            context = take_context(plan, handoff_cursor)
                            seed = (seed + 1) & 0xFFFFFFFF
                            future = executor.submit(
                                plan_with_wall_time,
                                planner,
                                context,
                                control,
                                seed,
                                args.target_clip,
                            )
                            pending = AsyncPlanRequest(
                                future=future,
                                context=context,
                                control=control,
                                handoff_cursor=handoff_cursor,
                                random_seed=seed,
                                reason=reason,
                                submitted_at=time.monotonic(),
                            )
                            async_submitted += 1
                            print(
                                f"replan-submit={async_submitted} reason={reason} "
                                f"cursor={cursor} handoff_cursor={handoff_cursor} "
                                f"lead_frames={handoff_cursor - cursor} "
                                f"mode={control[0]}:{CLIP_NAMES[control[0]]}"
                            )

                display_at = time.monotonic()
                if last_display_at is not None:
                    display_gap_ms = (display_at - last_display_at) * 1000.0
                    maximum_display_gap_ms = max(maximum_display_gap_ms, display_gap_ms)
                    if display_gap_ms > period * 1500.0:
                        late_display_gaps += 1
                last_display_at = display_at
                data.qpos[:] = plan[min(cursor, len(plan) - 1)]
                data.qvel.fill(0.0)
                mujoco.mj_forward(model, data)
                viewer.sync()
                cursor = min(cursor + 1, len(plan) - 1)
                frames_since_replan += 1
                played_steps += 1

                if executor is None:
                    interval_due = frames_since_replan >= args.replan_frames
                    change_due = args.replan_on_control_change and control_changed(
                        control, last_planned_control
                    )
                    buffer_due = cursor + CONTEXT_FRAMES >= len(plan)
                    if interval_due or change_due or buffer_due:
                        context = take_context(plan, cursor)
                        seed = (seed + 1) & 0xFFFFFFFF
                        planning_started = time.monotonic()
                        plan, metadata = planner.plan(
                            context,
                            mode=control[0],
                            movement_direction=control[1],
                            facing_direction=control[2],
                            random_seed=seed,
                            target_clip_name=args.target_clip,
                        )
                        wall_ms = (time.monotonic() - planning_started) * 1000.0
                        validate_qpos(model, data, plan)
                        cursor = 0
                        frames_since_replan = 0
                        last_planned_control = control
                        replan_count += 1
                        print(
                            f"replan={replan_count} input=[4,36] output={list(plan.shape)} "
                            f"mode={control[0]}:{CLIP_NAMES[control[0]]} "
                            f"movement={control[1]:+.3f} facing={control[2]:+.3f} "
                            f"tokens={metadata['tokens']} clip={metadata['clip']} "
                            f"planner_ms={metadata['total_ms']:.2f} wall_ms={wall_ms:.2f} "
                            "handoff=blocking"
                        )

                next_frame_at += period
                if next_frame_at < time.monotonic() - period:
                    next_frame_at = time.monotonic() + period
    finally:
        keyboard.close()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
        planner.close()
    print(
        f"final-qpos C++ viewer completed: played_steps={played_steps} "
        f"replans={replan_count} mode={args.viewer_replan_mode} "
        f"async_submitted={async_submitted} async_discarded={async_discarded} "
        f"async_missed={async_missed} max_display_gap_ms={maximum_display_gap_ms:.2f} "
        f"late_display_gaps={late_display_gaps}"
    )


def main() -> None:
    args = parse_args()
    args.planner_cli = args.planner_cli.expanduser().resolve()
    args.asset_pack = args.asset_pack.expanduser().resolve()
    args.model_root = args.model_root.expanduser().resolve()
    args.xml = args.xml.expanduser().resolve()
    if not args.xml.is_file():
        raise FileNotFoundError(f"MuJoCo scene not found: {args.xml}")
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    if args.headless:
        run_headless(args, model, data)
    else:
        _prefer_x11_glfw()
        run_viewer(args, model, data)


if __name__ == "__main__":
    main()
