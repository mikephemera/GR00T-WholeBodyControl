"""View preprocessed mocap clips stored in the G1-clip checkpoint.

Plays each clip in the MuJoCo viewer, auto-cycling through all 15 clips.
Close the viewer window (ESC) to stop.
"""

import argparse
import numpy as np
import torch as t
import mujoco
import mujoco.viewer
import time

CLIP_NAMES = [
    "idle", "slow_walk", "walk", "hand_crawling", "walk_boxing",
    "elbow_crawling", "stealth_walk", "injured_walk", "walk_stealth",
    "walk_happy_dance", "walk_zombie", "walk_gun", "walk_scared",
    "walk_left", "walk_right",
]


def load_clips(ckpt_path: str):
    sd = t.load(ckpt_path, map_location="cpu", weights_only=True)
    qpos = sd["mujoco_qpos"]
    num_frames = sd["num_frames_per_clip"]
    return qpos, num_frames


def main():
    parser = argparse.ArgumentParser(description="View MotionBricks mocap clips")
    parser.add_argument("--ckpt", type=str, default="out/G1-clip.ckpt")
    parser.add_argument("--xml", type=str, default="assets/skeletons/g1/scene_29dof.xml")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--loops_per_clip", type=int, default=2,
                        help="How many times to loop each clip before advancing")
    args = parser.parse_args()

    qpos_all, num_frames_all = load_clips(args.ckpt)
    num_clips = len(CLIP_NAMES)

    mj_model = mujoco.MjModel.from_xml_path(args.xml)
    mj_data = mujoco.MjData(mj_model)

    # Print clip summary
    print("=" * 70)
    print("MotionBricks Clip Viewer — playing all 15 mocap clips")
    print("=" * 70)
    print(f"{'#':>3s}  {'Name':<22s}  {'Frames':>6s}  {'Duration':>8s}")
    print("-" * 70)
    for i, name in enumerate(CLIP_NAMES):
        nf = num_frames_all[i].item()
        dur = nf / args.fps
        print(f"{i+1:3d}  {name:<22s}  {nf:6d}  {dur:7.2f}s")
    print("-" * 70)
    print("Auto-cycling through clips. Close viewer window to quit.\n")

    clip_idx = 0
    loop_count = 0
    frame_idx = 0
    state = "PLAYING"  # PLAYING | TRANSITION

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            nf = num_frames_all[clip_idx].item()
            if nf <= 0:
                clip_idx = (clip_idx + 1) % num_clips
                frame_idx = 0
                loop_count = 0
                continue

            f = frame_idx % nf
            qpos = qpos_all[clip_idx, f].numpy().copy()
            mj_data.qpos[:] = qpos
            mujoco.mj_forward(mj_model, mj_data)

            # Camera follows the robot
            viewer.cam.lookat[:] = mj_data.qpos[:3].copy()

            viewer.sync()

            name = CLIP_NAMES[clip_idx]
            print(f"\r[{clip_idx+1:2d}/{num_clips}] {name:<22s}  "
                  f"frame {f:4d}/{nf:4d}  "
                  f"loop {loop_count+1}/{args.loops_per_clip}  ",
                  end="", flush=True)

            frame_idx += 1
            if frame_idx >= nf:
                frame_idx = 0
                loop_count += 1
                if loop_count >= args.loops_per_clip:
                    loop_count = 0
                    clip_idx = (clip_idx + 1) % num_clips

            # Maintain FPS
            elapsed = time.time() - step_start
            sleep_time = max(0, 1.0 / args.fps - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    print("\nDone.")


if __name__ == "__main__":
    main()
