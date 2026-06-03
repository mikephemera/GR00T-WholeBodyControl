#!/usr/bin/env python3
"""Export MB motion sequence from fallen pose to unitree_sim2x CSV format.

Run from motionbricks/ directory:
    python scripts/export_fallen_sequence.py --num_steps 600
"""

import os
import sys
import time
import argparse

import mujoco
import mujoco.viewer
import numpy as np
import torch as t
from scipy.spatial.transform import Rotation as R

# --- Add project root to path ---
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from motionbricks.motion_backbone.demo.utils import navigation_demo
from motionbricks.motion_backbone.demo.clips import clip_holder_G1

# Body indices for unitree_sim2x CSV format.
# Index 0 in CSV = pelvis (MJCF body[1]), which serves as the root.
# The metadata body_indexes follow SMPL body-part numbering.
# We map: 1→pelvis, 4→left_hip_yaw, 10→right_hip_yaw, 18→left_shoulder_roll,
#         5→left_knee, 11→right_knee, 19→left_shoulder_yaw, 9→right_hip_roll,
#         16→torso, 22→left_wrist_pitch, 28→right_wrist_roll, 17→left_shoulder_pitch,
#         23→left_wrist_yaw, 29→right_wrist_pitch
BODY_INDICES = [1, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29]

# Fallen pose imported from unitree_sim2x/MJDATA.TXT
FALLEN_POSE = {
    "root_z": 0.23,
    "root_quat": np.array([0.71, 0.62, -0.2, -0.26]),  # w,x,y,z
    "joints": np.array(
        [
            -0.64,
            0.34,
            0.99,
            1.3,
            -0.34,
            -0.07,
            -1.5,
            -0.072,
            -0.11,
            0.95,
            -0.081,
            0.017,
            -1.4,
            0.53,
            0.43,
            -0.29,
            0.013,
            -0.045,
            0.69,
            0.41,
            0.063,
            -0.002,
            0.32,
            -1.9,
            0.65,
            -0.69,
            0.15,
            -0.076,
            -0.53,
        ]
    ),
}

# Fully flat supine pose (lie on back).
# Use the alternate supine heading and set arm joints to lie flat instead of raised.
FALLEN_POSE_2 = {
    "root_z": 0.18,
    "root_quat": np.array([0.70710678, 0.0, -0.70710678, 0.0]),  # w,x,y,z
    "joints": np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,  # left_leg
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,  # right_leg
            0.0,
            0.0,
            0.0,  # waist
            -1.5708,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,  # left_arm (flat on floor)
            -1.5708,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,  # right_arm (flat on floor)
        ]
    ),
}

FALLEN_POSES = {"1": FALLEN_POSE, "2": FALLEN_POSE_2}


def apply_fallen_pose(mj_data, full_agent, pose_dict, device="cuda"):
    """Set MuJoCo state and agent buffer to the given fallen pose."""
    qpos = mj_data.qpos.copy()
    qpos[2] = pose_dict["root_z"]
    qpos[3:7] = pose_dict["root_quat"]
    qpos[7:] = pose_dict["joints"]

    mj_data.qpos[:] = qpos
    mj_data.qvel[:] = 0.0

    # Fill internal buffer with 64 copies so the agent starts from this pose
    t_qpos = t.from_numpy(qpos).float().to(device).view(1, 1, -1)
    t_qpos = t_qpos.repeat(1, 64, 1)
    full_agent.frames["mujoco_qpos"] = t_qpos
    full_agent._current_frame_idx = 0


def compute_body_states_for_qpos(mj_model, mj_data, qpos, body_indices):
    """Run MuJoCo forward kinematics and return body positions & quaternions."""
    mj_data.qpos[:] = qpos
    mujoco.mj_forward(mj_model, mj_data)
    body_pos = mj_data.xpos[body_indices].copy()  # [14, 3]
    body_quat = mj_data.xquat[body_indices].copy()  # [14, 4] w,x,y,z
    return body_pos, body_quat


def compute_angular_velocity_from_quats(q_prev, q_next, dt):
    """Compute angular velocity in world frame from two quaternions.

    q_prev, q_next: [nb, 4]  quaternions in [w,x,y,z] format
    dt: time difference (q_next at t+dt, q_prev at t-dt, so 2*dt between them)
    Returns: [nb, 3] angular velocity in world frame
    """
    nb = q_prev.shape[0]
    ang_vel = np.zeros((nb, 3))
    for b in range(nb):
        r_prev = R.from_quat(q_prev[b, [1, 2, 3, 0]])  # scipy uses x,y,z,w
        r_next = R.from_quat(q_next[b, [1, 2, 3, 0]])
        # dR = R_next @ R_prev^T  (world-frame rotation change)
        r_diff = r_next * r_prev.inv()
        rotvec = (
            r_diff.as_rotvec()
        )  # axis * angle (in world frame? no, this is in the body's frame...)
        # Actually, as_rotvec gives the rotation vector. For small angles,
        # rotvec ≈ ω * Δt in world frame since both R are world-from-body.
        # dR = R(t+dt) * R(t-dt)^T ≈ exp([ω]_× * 2*dt) in world frame
        ang_vel[b] = rotvec / dt
    return ang_vel


def compute_velocities(recorded_qpos, body_pos_seq, body_quat_seq, fps=30):
    """Compute joint velocities and body velocities via central finite differences."""
    num_frames = len(recorded_qpos)
    dt_single = 1.0 / fps
    nb = len(BODY_INDICES)

    joint_vel = np.zeros_like(recorded_qpos[:, 7:])
    body_lin_vel = np.zeros_like(body_pos_seq)
    body_ang_vel = np.zeros((num_frames, nb, 3))

    # Forward difference for first frame
    joint_vel[0] = (recorded_qpos[1, 7:] - recorded_qpos[0, 7:]) / dt_single
    body_lin_vel[0] = (body_pos_seq[1] - body_pos_seq[0]) / dt_single
    body_ang_vel[0] = compute_angular_velocity_from_quats(
        body_quat_seq[0], body_quat_seq[1], dt_single
    )

    # Central difference for interior frames
    for i in range(1, num_frames - 1):
        joint_vel[i] = (recorded_qpos[i + 1, 7:] - recorded_qpos[i - 1, 7:]) / (2 * dt_single)
        body_lin_vel[i] = (body_pos_seq[i + 1] - body_pos_seq[i - 1]) / (2 * dt_single)
        body_ang_vel[i] = compute_angular_velocity_from_quats(
            body_quat_seq[i - 1], body_quat_seq[i + 1], 2 * dt_single
        )

    # Backward difference for last frame
    joint_vel[-1] = (recorded_qpos[-1, 7:] - recorded_qpos[-2, 7:]) / dt_single
    body_lin_vel[-1] = (body_pos_seq[-1] - body_pos_seq[-2]) / dt_single
    body_ang_vel[-1] = compute_angular_velocity_from_quats(
        body_quat_seq[-2], body_quat_seq[-1], dt_single
    )

    return joint_vel, body_lin_vel, body_ang_vel


def write_csv(filepath, data_2d, header=None):
    """Write a 2D array to CSV, optionally with a header row."""
    with open(filepath, "w") as f:
        if header:
            f.write(header + "\n")
        np.savetxt(f, data_2d, delimiter=",", fmt="%.6f")


def export_motion(
    output_dir,
    motion_name,
    recorded_qpos,
    body_pos_seq,
    body_quat_seq,
    joint_vel,
    body_lin_vel,
    body_ang_vel,
):
    """Write all CSV files and metadata for a motion sequence."""
    out_dir = os.path.join(output_dir, motion_name)
    os.makedirs(out_dir, exist_ok=True)

    num_frames, nb = len(recorded_qpos), len(BODY_INDICES)

    # joint_pos.csv: [T, 29]
    joint_header = ",".join(f"joint_{i}" for i in range(29))
    write_csv(os.path.join(out_dir, "joint_pos.csv"), recorded_qpos[:, 7:], joint_header)

    # joint_vel.csv: [T, 29]
    write_csv(os.path.join(out_dir, "joint_vel.csv"), joint_vel, joint_header)

    # body_pos.csv: [T, 14*3]
    body_pos_header = ",".join(f"body_{i}_{axis}" for i in range(nb) for axis in ["x", "y", "z"])
    write_csv(
        os.path.join(out_dir, "body_pos.csv"), body_pos_seq.reshape(num_frames, -1), body_pos_header
    )

    # body_quat.csv: [T, 14*4]
    body_quat_header = ",".join(f"body_{i}_{q}" for i in range(nb) for q in ["w", "x", "y", "z"])
    write_csv(
        os.path.join(out_dir, "body_quat.csv"),
        body_quat_seq.reshape(num_frames, -1),
        body_quat_header,
    )

    # body_lin_vel.csv: [T, 14*3]
    lin_vel_header = ",".join(f"body_{i}_{axis}" for i in range(nb) for axis in ["x", "y", "z"])
    write_csv(
        os.path.join(out_dir, "body_lin_vel.csv"),
        body_lin_vel.reshape(num_frames, -1),
        lin_vel_header,
    )

    # body_ang_vel.csv: [T, 14*3]
    ang_vel_header = ",".join(f"body_{i}_{axis}" for i in range(nb) for axis in ["x", "y", "z"])
    write_csv(
        os.path.join(out_dir, "body_ang_vel.csv"),
        body_ang_vel.reshape(num_frames, -1),
        ang_vel_header,
    )

    # info.txt
    with open(os.path.join(out_dir, "info.txt"), "w") as f:
        f.write(f"Motion Information: {motion_name}\n")
        f.write("==================================================\n\n")
        f.write(f"joint_pos:\n  Shape: ({num_frames}, 29)\n  Dtype: float32\n")
        f.write(
            f"  Range: [{recorded_qpos[:, 7:].min():.3f}, {recorded_qpos[:, 7:].max():.3f}]\n\n"
        )
        f.write(f"joint_vel:\n  Shape: ({num_frames}, 29)\n  Dtype: float32\n")
        f.write(f"  Range: [{joint_vel.min():.3f}, {joint_vel.max():.3f}]\n\n")
        f.write(f"body_pos_w:\n  Shape: ({num_frames}, {nb}, 3)\n  Dtype: float32\n")
        f.write(f"  Range: [{body_pos_seq.min():.3f}, {body_pos_seq.max():.3f}]\n\n")
        f.write(f"body_quat_w:\n  Shape: ({num_frames}, {nb}, 4)\n  Dtype: float32\n")
        f.write(f"  Range: [{body_quat_seq.min():.3f}, {body_quat_seq.max():.3f}]\n\n")
        f.write(f"body_lin_vel_w:\n  Shape: ({num_frames}, {nb}, 3)\n  Dtype: float32\n")
        f.write(f"  Range: [{body_lin_vel.min():.3f}, {body_lin_vel.max():.3f}]\n\n")
        f.write(f"body_ang_vel_w:\n  Shape: ({num_frames}, {nb}, 3)\n  Dtype: float32\n")
        f.write(f"  Range: [{body_ang_vel.min():.3f}, {body_ang_vel.max():.3f}]\n\n")
        f.write(f"_body_indexes:\n  Shape: ({nb},)\n  Dtype: int64\n")
        f.write(f"  Range: [{min(BODY_INDICES):.3f}, {max(BODY_INDICES):.3f}]\n\n")
        f.write(f"time_step_total:\n  Shape: ()\n  Dtype: int64\n")
        f.write(f"  Range: [{num_frames}, {num_frames}]\n")

    # metadata.txt
    with open(os.path.join(out_dir, "metadata.txt"), "w") as f:
        f.write(f"Metadata for: {motion_name}\n")
        f.write("==============================\n\n")
        f.write(f"Body part indexes:\n{BODY_INDICES}\n\n")
        f.write(f"Total timesteps: {num_frames}\n\n")
        f.write("Data arrays summary:\n")
        f.write(f"  joint_pos: ({num_frames}, 29) (float32)\n")
        f.write(f"  joint_vel: ({num_frames}, 29) (float32)\n")
        f.write(f"  body_pos_w: ({num_frames}, {nb}, 3) (float32)\n")
        f.write(f"  body_quat_w: ({num_frames}, {nb}, 4) (float32)\n")
        f.write(f"  body_lin_vel_w: ({num_frames}, {nb}, 3) (float32)\n")
        f.write(f"  body_ang_vel_w: ({num_frames}, {nb}, 3) (float32)\n")
        f.write(f"  _body_indexes: ({nb},) (int64)\n")
        f.write(f"  time_step_total: () (int64)\n")

    print(f"Exported {num_frames} frames to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Export MB motion sequence to unitree_sim2x CSV format."
    )

    # Path configs
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: unitree_sim2x/assets/motions/g1_29dof/csv/)",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="fallen_recovery",
        help="Name of the output motion directory",
    )

    # Run configs
    parser.add_argument(
        "--num_steps",
        type=int,
        default=600,
        help="Number of simulation steps to record (600 = 20s @ 30fps)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="walk",
        help="Motion mode for recovery (walk, idle, slow_walk, etc.)",
    )
    parser.add_argument(
        "--has_viewer", type=int, default=0, help="Launch MuJoCo viewer (1=yes, 0=no)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generate_dt", type=float, default=2.0, help="Replan interval in seconds")

    # Model configs (passed to navigation_demo)
    parser.add_argument("--clips", type=str, default="G1")
    parser.add_argument("--planner", type=str, default="default")
    parser.add_argument(
        "--pose",
        type=str,
        default="1",
        choices=["1", "2"],
        help="Which fallen pose to use (1=original, 2=flatter)",
    )

    args = parser.parse_args()
    pose_dict = FALLEN_POSES[args.pose]

    # Default output to unitree_sim2x (absolute path)
    if args.output_dir is None:
        args.output_dir = "/mnt/datafiles/Work-syncfree/unitree_sim2x/assets/motions/g1_29dof/csv"
    args.output_dir = os.path.abspath(args.output_dir)

    # Build synthetic args for navigation_demo (matching demo defaults)
    demo_args = argparse.Namespace()
    demo_args.humanoid_xml = os.path.join(
        _project_root, "assets", "skeletons", "g1", "scene_29dof.xml"
    )
    demo_args.skeleton_xml = os.path.join(_project_root, "assets", "skeletons", "g1", "g1.xml")
    demo_args.result_dir = os.path.join(_project_root, "out")
    demo_args.data_root = os.path.join(_project_root, "datasets")
    demo_args.explicit_dataset_folder = os.path.join(_project_root, "datasets", "motionbricks-G1")
    demo_args.clips_ckpt = os.path.join(_project_root, "out", "G1-clip.ckpt")
    demo_args.reprocess_clips = 0

    demo_args.controller = "random"  # avoid keyboard/X11 dependency
    demo_args.has_viewer = 0
    demo_args.use_qpos = 1
    demo_args.clips = args.clips
    demo_args.EXP = args.planner
    demo_args.planner = args.planner
    demo_args.speed_scale = [1.0, 1.0]
    demo_args.random_speed_scale = 0
    demo_args.pre_filter_qpos = 1
    # For fully supine initialization (pose 2), keep raw root orientation to avoid
    # heading re-alignment jumps that can look like planar teleporting.
    if args.pose == "2":
        demo_args.source_root_realignment = 0
        demo_args.target_root_realignment = 0
        demo_args.force_canonicalization = 0
    else:
        demo_args.source_root_realignment = 1
        demo_args.target_root_realignment = 1
        demo_args.force_canonicalization = 1
    demo_args.skip_ending_target_cond = 0
    demo_args.lookat_movement_direction = 0
    demo_args.generate_dt = args.generate_dt
    demo_args.max_steps = 100000
    demo_args.num_runs = 1
    demo_args.random_seed = args.seed
    demo_args.return_model_configs = True
    demo_args.return_dataloader = True
    demo_args.recording_dir = None
    demo_args.return_dataloader = True

    # Set random seeds
    np.random.seed(args.seed)
    t.manual_seed(args.seed)

    # --- Load models ---
    print("Loading MotionBricks models...")
    demo = navigation_demo(demo_args)

    device = "cuda"
    full_agent = demo.full_agent
    mj_model = demo.mj_model
    mj_data = demo.mj_data
    clip_holder = full_agent._clip_holder

    # Validate mode
    mode_name = args.mode
    if mode_name not in clip_holder.CLIPS:
        print(f'Error: mode "{mode_name}" not found. Available: {list(clip_holder.CLIPS.keys())}')
        sys.exit(1)
    mode_idx = list(clip_holder.CLIPS.keys()).index(mode_name)

    # Get allowed_pred_num_tokens for this mode
    clip_info = clip_holder.CLIPS[mode_name]
    min_token = full_agent._inferencer._args.get("min_tokens", 6)
    max_token = full_agent._inferencer._args.get("max_tokens", 16)
    num_token_range = max_token - min_token + 1
    if "allowed_pred_num_tokens" in clip_info:
        allowed_pred_num_tokens = t.tensor(clip_info["allowed_pred_num_tokens"]).view(1, -1)
    else:
        allowed_pred_num_tokens = t.ones(num_token_range, dtype=t.int).view(1, -1)

    # --- Apply fallen pose ---
    print(f"Applying fallen pose {args.pose}...")
    apply_fallen_pose(mj_data, full_agent, pose_dict, device)
    print(
        f'  root_z={pose_dict["root_z"]}, '
        f'root_quat=[{pose_dict["root_quat"][0]:.2f}, {pose_dict["root_quat"][1]:.2f}, '
        f'{pose_dict["root_quat"][2]:.2f}, {pose_dict["root_quat"][3]:.2f}]'
    )

    # --- Run MB inference loop ---
    recorded_qpos = []
    first_generation_done = False

    print(f"Running MB inference for {args.num_steps} steps (mode={mode_name})...")

    def _run_step():
        nonlocal first_generation_done
        qpos = full_agent.get_next_frame()
        recorded_qpos.append(qpos.copy())

        context_mujoco_qpos = full_agent.get_context_mujoco_qpos()

        control_signals = {
            "movement_direction": t.tensor([[1.0, 0.0, 0.0]]),
            "facing_direction": t.tensor([[1.0, 0.0, 0.0]]),
            "mode": t.tensor([[mode_idx]]).long(),
            "context_mujoco_qpos": context_mujoco_qpos,
            "allowed_pred_num_tokens": allowed_pred_num_tokens,
        }

        with t.no_grad():
            full_agent.generate_new_frames(
                control_signals,
                args.generate_dt,
                force_generation=(not first_generation_done),
            )

        if not first_generation_done:
            first_generation_done = True

        return qpos

    if args.has_viewer:
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            step = 0
            while viewer.is_running() and step < args.num_steps:
                step_start = time.time()
                qpos = _run_step()
                mj_data.qpos[:] = qpos
                mujoco.mj_forward(mj_model, mj_data)
                viewer.sync()

                if (step + 1) % 120 == 0:
                    root_z = recorded_qpos[-1][2]
                    print(f"  Step {step + 1}/{args.num_steps}, root_z={root_z:.3f}")

                time_until_next = mj_model.opt.timestep - (time.time() - step_start)
                if time_until_next > 0:
                    time.sleep(time_until_next)
                step += 1
    else:
        for step in range(args.num_steps):
            _run_step()
            if (step + 1) % 120 == 0:
                root_z = recorded_qpos[-1][2]
                print(f"  Step {step + 1}/{args.num_steps}, root_z={root_z:.3f}")

    recorded_qpos = np.array(recorded_qpos)
    num_frames = len(recorded_qpos)
    print(f"Recorded {num_frames} frames.")

    # --- Post-process: compute body states ---
    print("Computing body positions and quaternions via MuJoCo forward kinematics...")
    body_pos_seq = np.zeros((num_frames, len(BODY_INDICES), 3))
    body_quat_seq = np.zeros((num_frames, len(BODY_INDICES), 4))

    for i in range(num_frames):
        pos, quat = compute_body_states_for_qpos(mj_model, mj_data, recorded_qpos[i], BODY_INDICES)
        body_pos_seq[i] = pos
        body_quat_seq[i] = quat

    # Compute velocities
    print("Computing velocities via finite differences...")
    joint_vel, body_lin_vel, body_ang_vel = compute_velocities(
        recorded_qpos, body_pos_seq, body_quat_seq, fps=30
    )

    # --- Export ---
    print(f"Exporting to {args.output_dir}/{args.output_name}/ ...")
    export_motion(
        args.output_dir,
        args.output_name,
        recorded_qpos,
        body_pos_seq,
        body_quat_seq,
        joint_vel,
        body_lin_vel,
        body_ang_vel,
    )
    print("Done.")


if __name__ == "__main__":
    main()
