#!/usr/bin/env python3
"""Replay a motion sequence exported by export_fallen_sequence.py in MuJoCo viewer.

Usage:
    python scripts/replay_exported_motion.py --motion_dir /path/to/csv/fallen_recovery
    python scripts/replay_exported_motion.py  # uses default export path
"""

import argparse
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np


def load_csv(filepath, skip_header=True):
    return np.loadtxt(filepath, delimiter=',', skiprows=1 if skip_header else 0)


def main():
    parser = argparse.ArgumentParser(description='Replay exported motion CSV sequence in MuJoCo.')
    parser.add_argument('--motion_dir', type=str, default=None,
                        help='Path to the exported CSV directory')
    parser.add_argument('--fps', type=int, default=30,
                        help='Replay frame rate')
    parser.add_argument('--loop', type=int, default=1,
                        help='Loop playback (1=yes, 0=no)')
    parser.add_argument('--scene', type=str,
                        default='/mnt/datafiles/Work-syncfree/unitree_sim2x/assets/robots/g1_29dof/scene_flat.xml',
                        help='MuJoCo scene XML')
    args = parser.parse_args()

    if args.motion_dir is None:
        args.motion_dir = '/mnt/datafiles/Work-syncfree/unitree_sim2x/assets/motions/g1_29dof/csv/fallen_recovery'

    if not os.path.isdir(args.motion_dir):
        print(f'Error: motion directory not found: {args.motion_dir}')
        sys.exit(1)

    # Load CSV data
    print(f'Loading motion from: {args.motion_dir}')
    joint_pos = load_csv(os.path.join(args.motion_dir, 'joint_pos.csv'))          # [T, 29]
    body_pos = load_csv(os.path.join(args.motion_dir, 'body_pos.csv'))            # [T, 14*3]
    body_quat = load_csv(os.path.join(args.motion_dir, 'body_quat.csv'))          # [T, 14*4]

    num_frames = joint_pos.shape[0]
    nb = 14  # number of tracked bodies
    body_pos = body_pos.reshape(num_frames, nb, 3)
    body_quat = body_quat.reshape(num_frames, nb, 4)

    # CSV body_0 = pelvis (root). Extract root from first body column.
    root_pos_seq = body_pos[:, 0, :]     # [T, 3]
    root_quat_seq = body_quat[:, 0, :]   # [T, 4] w,x,y,z

    print(f'  Frames: {num_frames}')
    print(f'  Joint pos range: [{joint_pos.min():.3f}, {joint_pos.max():.3f}]')
    print(f'  Root z range: [{root_pos_seq[:, 2].min():.3f}, {root_pos_seq[:, 2].max():.3f}]')

    # Load MuJoCo model
    print(f'Loading scene: {args.scene}')
    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    model.opt.timestep = 1.0 / args.fps

    # Verify joint count matches
    nb_joints_csv = joint_pos.shape[1]
    nb_joints_mj = model.nq - 7  # subtract free joint (3 pos + 4 quat)
    print(f'  CSV joints: {nb_joints_csv}, MuJoCo joints: {nb_joints_mj}')
    if nb_joints_csv != nb_joints_mj:
        print(f'  WARNING: joint count mismatch!')

    frame_idx = [0]  # use list for mutable closure in viewer callback

    print(f'\nReplaying {num_frames} frames at {args.fps} fps...')
    print('Controls: Space=pause, Right=step, Esc=quit')

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Sync the viewer loop with our desired frame rate
        viewer._run_speed = 1.0  # real-time
        next_frame_time = time.time()

        while viewer.is_running():
            i = frame_idx[0]

            # Reconstruct full qpos
            qpos = np.zeros(model.nq)
            qpos[:3] = root_pos_seq[i]          # root position
            qpos[3:7] = root_quat_seq[i]         # root quaternion [w,x,y,z]
            qpos[7:7 + nb_joints_csv] = joint_pos[i]  # joint positions

            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            viewer.sync()

            # Frame timing
            elapsed = time.time() - next_frame_time
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)
            next_frame_time += model.opt.timestep

            # Advance frame, optionally loop
            frame_idx[0] += 1
            if frame_idx[0] >= num_frames:
                if args.loop:
                    frame_idx[0] = 0
                    next_frame_time = time.time()  # reset timing on loop
                else:
                    print('Playback complete. Close viewer to exit.')
                    # Keep viewer open, freeze at last frame
                    frame_idx[0] = num_frames - 1
                    time.sleep(0.1)

        print('Viewer closed.')


if __name__ == '__main__':
    main()
