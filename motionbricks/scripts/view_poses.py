#!/usr/bin/env python3
"""Render fixed poses from CLAUDE_poses.md in MuJoCo with interactive browsing.

Usage:
    python scripts/view_poses.py                          # browse all poses
    python scripts/view_poses.py --pose POSE1             # view specific pose
    python scripts/view_poses.py --pose POSE1,POSE11      # view subset
    python scripts/view_poses.py --list                   # list available poses

Keyboard controls:
    N / Right Arrow     next pose
    P / Left Arrow      previous pose
    R                   reset to default qpos
    Q / Esc             quit
"""

import os
import re
import sys
import argparse
import numpy as np

import mujoco
import mujoco.viewer

# GLFW key codes
_KEY_ESC = 256
_KEY_RIGHT = 262
_KEY_LEFT = 263
_KEY_N = 78
_KEY_P = 80
_KEY_R = 82
_KEY_Q = 81


def parse_poses(filepath):
    """Parse CLAUDE_poses.md and return dict of {name: {root_z, root_quat, joints}}."""
    with open(filepath) as f:
        content = f.read()

    poses = {}
    # Find all POSE definitions — split on the pattern marker
    pattern = re.compile(r"(POSE\d+)\s*=\s*\{")
    blocks = pattern.split(content)

    # blocks[0] = text before first POSE, then alternating (name, body, name, body, ...)
    for i in range(1, len(blocks), 2):
        name = blocks[i]
        body = blocks[i + 1]

        # --- root_z ---
        m = re.search(r'"root_z"\s*:\s*([-\d.e]+)', body)
        if not m:
            print(f"Warning: could not parse root_z for {name}, skipping")
            continue
        root_z = float(m.group(1))

        # --- root_quat ---
        m = re.search(r'"root_quat"\s*:\s*np\.array\(\s*\[([^\]]+)\]\)', body)
        if not m:
            print(f"Warning: could not parse root_quat for {name}, skipping")
            continue
        root_quat = np.fromstring(m.group(1), sep=",", dtype=float)

        # --- joints ---
        # Find the joints section and pull out all float values
        joint_start = body.index('"joints"')
        joint_body = body[joint_start:]
        # Match float literals followed by a comma (handles both single and double commas)
        values = re.findall(r'([-\d.e]+)\s*,', joint_body)
        if not values:
            print(f"Warning: could not parse joints for {name}, skipping")
            continue
        joints = np.array([float(v) for v in values], dtype=float)

        if len(joints) != 29:
            print(f"Warning: {name} has {len(joints)} joints, expected 29")

        poses[name] = {"root_z": root_z, "root_quat": root_quat, "joints": joints}

    return poses


def apply_pose(mj_model, mj_data, pose, default_qpos):
    """Apply a pose dict to MuJoCo data."""
    qpos = default_qpos.copy()
    qpos[2] = pose["root_z"]
    qpos[3:7] = pose["root_quat"]  # w,x,y,z
    qpos[7:] = pose["joints"]
    mj_data.qpos[:] = qpos
    mj_data.qvel[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)


def apply_default_pose(mj_model, mj_data, default_qpos):
    """Reset to the default (standing) qpos."""
    mj_data.qpos[:] = default_qpos
    mj_data.qvel[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)


def main():
    parser = argparse.ArgumentParser(
        description="Render G1 fixed poses from CLAUDE_poses.md in MuJoCo"
    )
    parser.add_argument(
        "--poses-file",
        type=str,
        default="/mnt/datafiles/Work-syncfree/unitree_sim2x/CLAUDE_poses.md",
        help="Path to CLAUDE_poses.md",
    )
    parser.add_argument(
        "--pose",
        type=str,
        default=None,
        help="Specific pose(s) to load, comma-separated (e.g. POSE1,POSE11)",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="MuJoCo scene XML (default: motionbricks/assets/skeletons/g1/scene_29dof.xml)",
    )
    parser.add_argument("--list", action="store_true", help="List available pose names and exit")
    args = parser.parse_args()

    if not os.path.exists(args.poses_file):
        print(f"Error: poses file not found: {args.poses_file}")
        sys.exit(1)

    poses = parse_poses(args.poses_file)
    if not poses:
        print("Error: no poses parsed from file")
        sys.exit(1)

    if args.list:
        print(f"Available poses ({len(poses)}):")
        for name in sorted(poses, key=lambda n: int(n[4:])):
            p = poses[name]
            print(f"  {name}: root_z={p['root_z']:.4f}, joints=[{p['joints'].min():.3f}..{p['joints'].max():.3f}]")
        return

    # --- Filter poses ---
    if args.pose:
        requested = [n.strip() for n in args.pose.split(",")]
        missing = [n for n in requested if n not in poses]
        if missing:
            print(f"Error: poses not found: {missing}")
            print(f"Available: {list(poses.keys())}")
            sys.exit(1)
        pose_names = requested
    else:
        pose_names = sorted(poses.keys(), key=lambda n: int(n[4:]))

    print(f"Loaded {len(pose_names)} poses: {pose_names}")

    # --- Load MuJoCo model ---
    if args.scene is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.scene = os.path.join(
            script_dir, "..", "assets", "skeletons", "g1", "scene_29dof.xml"
        )

    if not os.path.exists(args.scene):
        print(f"Error: scene not found: {args.scene}")
        sys.exit(1)

    mj_model = mujoco.MjModel.from_xml_path(args.scene)
    mj_data = mujoco.MjData(mj_model)
    default_qpos = mj_data.qpos.copy()

    # --- Viewer state ---
    state = {"idx": 0, "quit": False}

    def key_callback(keycode):
        if keycode == _KEY_ESC or keycode == _KEY_Q:
            state["quit"] = True
        elif keycode == _KEY_RIGHT or keycode == _KEY_N:
            state["idx"] = (state["idx"] + 1) % len(pose_names)
        elif keycode == _KEY_LEFT or keycode == _KEY_P:
            state["idx"] = (state["idx"] - 1) % len(pose_names)
        elif keycode == _KEY_R:
            apply_default_pose(mj_model, mj_data, default_qpos)
            print(f"\r[R] Reset to default qpos (standing)")
        elif _KEY_0 <= keycode <= _KEY_9:
            digit = keycode - _KEY_0
            # Try to find a pose whose numeric suffix starts with this digit
            for j, name in enumerate(pose_names):
                if name[4:].startswith(str(digit)):
                    state["idx"] = j
                    break

    # Apply first pose
    print(f"\nViewing: {pose_names[0]}")
    apply_pose(mj_model, mj_data, poses[pose_names[0]], default_qpos)

    last_idx = 0
    with mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=key_callback, show_left_ui=False, show_right_ui=False
    ) as viewer:
        while viewer.is_running() and not state["quit"]:
            if state["idx"] != last_idx:
                name = pose_names[state["idx"]]
                p = poses[name]
                apply_pose(mj_model, mj_data, p, default_qpos)
                print(f"\r[{state['idx'] + 1}/{len(pose_names)}] {name}  "
                      f"root_z={p['root_z']:.3f}  "
                      f"quat=[{p['root_quat'][0]:.3f},{p['root_quat'][1]:.3f},"
                      f"{p['root_quat'][2]:.3f},{p['root_quat'][3]:.3f}]")
                last_idx = state["idx"]

            viewer.sync()

    print("\nDone.")


if __name__ == "__main__":
    main()
