import argparse
import os
import platform
import time
from pathlib import Path

import torch as t

# pyGLFW chooses its native library at import time.  Prefer the bundled X11
# build under Wayland when XWayland is available; MotionBricks' keyboard grabs
# and MuJoCo viewer teardown are both reliable on that backend.
if platform.system() == "Linux" and os.environ.get("DISPLAY"):
    os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")

import mujoco
import mujoco.viewer
import numpy as np
from motionbricks.motion_backbone.demo.utils import navigation_demo


QPOS_FPS = 30.0
G1_QPOS_SIZE = 36
G1_SDK_JOINT_COUNT = 29


def _prefer_x11_glfw_on_linux():
    """Use GLFW's X11 backend when XWayland is available.

    The Wayland/EGL backend used by some GLFW builds can segfault while the
    passive MuJoCo viewer tears down its context.  MotionBricks also relies on
    X11 for its keyboard-grab workaround, so X11 is the consistent backend for
    this demo.
    """
    if platform.system() != "Linux" or not os.environ.get("DISPLAY"):
        return
    try:
        import glfw

        glfw.init_hint(glfw.PLATFORM, glfw.PLATFORM_X11)
    except (AttributeError, ImportError):
        pass


def _disable_mujoco_keyboard_shortcuts(controller_keys='wasdrtfgeqzxcvb'):
    """Prevent MuJoCo's viewer from processing keyboard shortcuts that
    conflict with the WASD motion controller.

    On Linux/X11: uses passive key grabs to intercept keys at the X server
    level before GLFW sees them.  pynput still captures keys via XRecord.

    On macOS/Windows: not yet supported — MuJoCo shortcuts may interfere.
    """
    if platform.system() != 'Linux':
        return
    try:
        from Xlib import display as xdisplay, X
        _xdpy = xdisplay.Display()
        _root = _xdpy.screen().root

        def _find_window_by_name(win, name_substr):
            try:
                name = win.get_wm_name()
                if name and name_substr in name:
                    return win
            except Exception:
                pass
            for child in win.query_tree().children:
                r = _find_window_by_name(child, name_substr)
                if r:
                    return r
            return None

        time.sleep(0.5)
        mj_win = _find_window_by_name(_root, 'MuJoCo')
        if mj_win:
            for ch in controller_keys:
                keycode = _xdpy.keysym_to_keycode(ord(ch) - 32)
                mj_win.grab_key(keycode, X.AnyModifier,
                                False, X.GrabModeAsync, X.GrabModeAsync)
            _xdpy.sync()
    except Exception as e:
        print(f"Note: could not disable MuJoCo keyboard shortcuts: {e}")


def _validate_qpos_output_args(args):
    if args.qpos_output is None:
        return None

    output_path = Path(args.qpos_output).expanduser()
    if output_path.suffix.lower() != ".npz":
        raise ValueError("--qpos_output must use the .npz extension")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing qpos recording: {output_path}")
    if args.controller != "fixed":
        raise ValueError("--qpos_output currently requires --controller fixed")
    if args.num_runs != 1:
        raise ValueError("--qpos_output requires --num_runs 1")
    if args.max_steps <= 0:
        raise ValueError("--qpos_output requires --max_steps greater than zero")
    return output_path


def _sdk_joint_names_in_qpos_order(mj_model):
    joints = []
    for joint_id in range(mj_model.njnt):
        if mj_model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        joints.append((int(mj_model.jnt_qposadr[joint_id]), name))

    joints.sort(key=lambda item: item[0])
    qpos_addresses = [address for address, _ in joints]
    joint_names = [name for _, name in joints]
    if mj_model.nq != G1_QPOS_SIZE:
        raise ValueError(f"Expected G1 nq={G1_QPOS_SIZE}, got {mj_model.nq}")
    if len(joint_names) != G1_SDK_JOINT_COUNT or any(name is None for name in joint_names):
        raise ValueError(
            f"Expected {G1_SDK_JOINT_COUNT} named G1 joints, got {len(joint_names)}"
        )
    if qpos_addresses != list(range(7, G1_QPOS_SIZE)):
        raise ValueError(f"Unexpected G1 joint qpos addresses: {qpos_addresses}")
    return joint_names


def save_qpos_recording(output_path, qpos_frames, mj_model, controller):
    """Validate and save the qpos frames actually returned by get_next_frame()."""
    output_path = Path(output_path)
    qpos = np.asarray(qpos_frames, dtype=np.float32)
    if qpos.ndim != 2 or qpos.shape[1] != G1_QPOS_SIZE:
        raise ValueError(f"Expected qpos with shape [frames, {G1_QPOS_SIZE}], got {qpos.shape}")
    if not np.isfinite(qpos).all():
        raise ValueError("qpos recording contains non-finite values")

    quaternion_norms = np.linalg.norm(qpos[:, 3:7], axis=1)
    if not np.allclose(quaternion_norms, 1.0, atol=1e-3, rtol=0.0):
        raise ValueError(
            "qpos recording contains invalid root quaternions; "
            f"norm range is [{quaternion_norms.min()}, {quaternion_norms.max()}]"
        )

    fps = 1.0 / float(mj_model.opt.timestep)
    if not np.isclose(fps, QPOS_FPS):
        raise ValueError(f"Expected a {QPOS_FPS:g} Hz MuJoCo model, got {fps:g} Hz")
    time_s = np.arange(qpos.shape[0], dtype=np.float64) / fps
    joint_names = np.asarray(_sdk_joint_names_in_qpos_order(mj_model), dtype=np.str_)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as output_file:
        np.savez(
            output_file,
            qpos=qpos,
            time_s=time_s,
            fps=np.asarray(fps, dtype=np.float64),
            mode=np.asarray(controller.mode, dtype=np.int32),
            mode_name=np.asarray(controller.mode_name, dtype=np.str_),
            target_speed_mps=np.asarray(controller.target_speed_mps, dtype=np.float32),
            movement_heading_rad=np.asarray(controller.movement_heading, dtype=np.float32),
            facing_heading_rad=np.asarray(controller.facing_heading, dtype=np.float32),
            random_seed=np.asarray(controller.random_seed, dtype=np.uint32),
            joint_names=joint_names,
            qpos_layout=np.asarray(
                "root_xyz,root_quaternion_wxyz,29_sdk_order_joints", dtype=np.str_
            ),
        )
    print(f"Saved {qpos.shape[0]} qpos frames at {fps:g} Hz to {output_path}")


def _add_context(control_signals, context_motion_features, context_mujoco_qpos, use_qpos):
    if use_qpos:
        control_signals['context_mujoco_qpos'] = context_mujoco_qpos
    else:
        control_signals['context_motion_features'] = context_motion_features


def _force_generate_first_fixed_segment(demo_agent, args):
    """Replace the reset-time idle buffer before the first recorded frame."""
    context_motion_features = demo_agent.full_agent.get_context_motion_features()
    context_mujoco_qpos = demo_agent.full_agent.get_context_mujoco_qpos()
    initial_qpos = context_mujoco_qpos[0, 0].detach().cpu().numpy()
    demo_agent.mj_data.qpos[:] = initial_qpos

    control_signals = demo_agent.controller.generate_control_signals(
        None,
        demo_agent.mj_model,
        demo_agent.mj_data,
        visualize=False,
        control_info={"force_idle": False, "allowed_mode": None},
    )
    _add_context(control_signals, context_motion_features, context_mujoco_qpos, args.use_qpos)
    with t.inference_mode():
        demo_agent.full_agent.generate_new_frames(
            control_signals,
            demo_agent.controller.get_controller_dt() * args.generate_dt,
            force_generation=True,
        )
    mujoco.mj_forward(demo_agent.mj_model, demo_agent.mj_data)


def main(args) -> None:
    qpos_output = _validate_qpos_output_args(args)
    _prefer_x11_glfw_on_linux()
    demo_agent = navigation_demo(args)

    num_runs = 0
    while num_runs < args.num_runs:
        num_runs += 1
        print(f"Running iteration {num_runs}... / {args.num_runs}")
        random_seed = args.random_seed * (num_runs + 2333) * 2333 % (2 ** 32 - 1)
        np.random.seed(random_seed)
        t.manual_seed(random_seed)
        demo_agent.full_agent.reset()

        if args.controller == "fixed":
            _force_generate_first_fixed_segment(demo_agent, args)

        steps = 0
        recorded_qpos = [] if qpos_output is not None else None

        if args.has_viewer:
            with mujoco.viewer.launch_passive(demo_agent.mj_model, demo_agent.mj_data) as viewer:
                _disable_mujoco_keyboard_shortcuts()

                while viewer.is_running() and steps < args.max_steps:
                    force_idle = steps + 100 > args.max_steps
                    steps += 1
                    viewer.user_scn.ngeom = 0
                    step_start = time.time()
                    qpos = demo_agent.full_agent.get_next_frame()
                    if recorded_qpos is not None:
                        recorded_qpos.append(np.asarray(qpos, dtype=np.float32).copy())
                    context_motion_features = demo_agent.full_agent.get_context_motion_features()
                    context_mujoco_qpos = demo_agent.full_agent.get_context_mujoco_qpos()
                    demo_agent.mj_data.qpos[:] = qpos

                    control_signals = demo_agent.controller.generate_control_signals(
                        viewer, demo_agent.mj_model, demo_agent.mj_data, visualize=True,
                        control_info={"force_idle": force_idle,
                                      'allowed_mode': getattr(args, 'allowed_mode', None)}
                    )

                    _add_context(control_signals, context_motion_features, context_mujoco_qpos, args.use_qpos)

                    with t.inference_mode():
                        demo_agent.full_agent.generate_new_frames(
                            control_signals,
                            demo_agent.controller.get_controller_dt() * args.generate_dt
                        )

                    mujoco.mj_forward(demo_agent.mj_model, demo_agent.mj_data)
                    viewer.cam.lookat[:] = demo_agent.controller.get_prev_qpos()[:, :3].mean(axis=0)
                    viewer.sync()
                    time_until_next_step = demo_agent.mj_model.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
        else:
            while steps < args.max_steps:
                steps += 1
                force_idle = steps + 100 > args.max_steps
                qpos = demo_agent.full_agent.get_next_frame()
                if recorded_qpos is not None:
                    recorded_qpos.append(np.asarray(qpos, dtype=np.float32).copy())
                context_motion_features = demo_agent.full_agent.get_context_motion_features()
                context_mujoco_qpos = demo_agent.full_agent.get_context_mujoco_qpos()
                demo_agent.mj_data.qpos[:] = qpos

                control_signals = demo_agent.controller.generate_control_signals(
                    None, demo_agent.mj_model, demo_agent.mj_data, visualize=False,
                    control_info={"force_idle": force_idle, 'allowed_mode': getattr(args, 'allowed_mode', None)}
                )
                _add_context(control_signals, context_motion_features, context_mujoco_qpos, args.use_qpos)

                with t.inference_mode():
                    demo_agent.full_agent.generate_new_frames(
                        control_signals, demo_agent.controller.get_controller_dt() * args.generate_dt
                    )

                mujoco.mj_forward(demo_agent.mj_model, demo_agent.mj_data)

        if qpos_output is not None:
            if len(recorded_qpos) != args.max_steps:
                raise RuntimeError(
                    f"Recording stopped after {len(recorded_qpos)} frames; expected {args.max_steps}"
                )
            save_qpos_recording(qpos_output, recorded_qpos, demo_agent.mj_model, demo_agent.controller)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive demo for the G1 humanoid")

    # path configs
    parser.add_argument("--humanoid_xml", type=str, default="assets/skeletons/g1/scene_29dof.xml")
    parser.add_argument("--result_dir", type=str, default="./out")
    parser.add_argument("--data_root", type=str, default="./datasets")
    parser.add_argument("--explicit_dataset_folder", type=str, default=None)
    parser.add_argument("--reprocess_clips", type=int, default=0)

    # controller config
    parser.add_argument("--controller", type=str, default="wasd",
                        choices=["wasd", "random", "fixed"])
    parser.add_argument("--fixed_mode", type=str, default="walk")
    parser.add_argument("--fixed_target_speed_mps", type=float, default=1.5)
    parser.add_argument("--fixed_movement_heading", type=float, default=0.0,
                        help="Fixed movement heading in radians")
    parser.add_argument("--fixed_facing_heading", type=float, default=0.0,
                        help="Fixed facing heading in radians")
    parser.add_argument("--lookat_movement_direction", type=int, default=0)
    parser.add_argument("--has_viewer", type=int, default=1)
    parser.add_argument("--pre_filter_qpos", type=int, default=1)
    parser.add_argument("--source_root_realignment", type=int, default=1)
    parser.add_argument("--target_root_realignment", type=int, default=1)
    parser.add_argument("--force_canonicalization", type=int, default=1)
    parser.add_argument("--skip_ending_target_cond", type=int, default=0)
    parser.add_argument("--random_speed_scale", type=int, default=0)
    parser.add_argument("--speed_scale", type=str, default="0.8,1.2")
    parser.add_argument("--generate_dt", type=float, default=2.0)

    # run configs
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Inference device: auto (MUSA, then CUDA, then CPU), musa[:N], cuda[:N], or cpu",
    )
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument(
        "--golden_dir", type=str, default=None,
        help="Optional directory for C++ golden data (manifest.json plus raw tensor binaries)",
    )
    parser.add_argument(
        "--qpos_output", "--qpos-output", dest="qpos_output", type=str, default=None,
        help="Write fixed-controller playback qpos and metadata to a new .npz file",
    )

    # model configurations
    parser.add_argument("--use_qpos", type=int, default=1)
    parser.add_argument("--planner", type=str, default="default")
    parser.add_argument("--allowed_mode", type=str, default=None)
    parser.add_argument("--clips", type=str, default="G1")

    args = parser.parse_args()

    args.return_model_configs = True
    args.return_dataloader = True
    args.recording_dir = None
    args.EXP = args.planner
    args.speed_scale = [float(i) for i in args.speed_scale.split(",")]

    main(args)
