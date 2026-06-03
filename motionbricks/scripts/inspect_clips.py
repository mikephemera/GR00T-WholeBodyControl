"""Inspect the preprocessed clip checkpoint — print shapes, ranges, and a few sample values."""

import torch as t
import argparse


CLIP_NAMES = [
    "idle", "slow_walk", "walk", "hand_crawling", "walk_boxing",
    "elbow_crawling", "stealth_walk", "injured_walk", "walk_stealth",
    "walk_happy_dance", "walk_zombie", "walk_gun", "walk_scared",
    "walk_left", "walk_right",
]


def main():
    parser = argparse.ArgumentParser(description="Inspect MotionBricks clip checkpoint")
    parser.add_argument("--ckpt", type=str, default="out/G1-clip.ckpt")
    args = parser.parse_args()

    sd = t.load(args.ckpt, map_location="cpu", weights_only=True)

    for k, v in sd.items():
        if hasattr(v, "shape"):
            print(f"{k}: shape={list(v.shape)}, dtype={v.dtype}, "
                  f"range=[{v.min().item():.4f}, {v.max().item():.4f}]")
        else:
            print(f"{k}: {type(v).__name__}")

    print()
    print("=== num_frames_per_clip ===")
    nf = sd["num_frames_per_clip"]
    for i, name in enumerate(CLIP_NAMES):
        print(f"  [{i:2d}] {name:<22s}: {nf[i].item():4d} frames")

    print()
    print("=== mujoco_qpos[0,0] (idle, frame 0) ===")
    print("  root_pos:", sd["mujoco_qpos"][0, 0, :3].tolist())
    print("  root_quat:", sd["mujoco_qpos"][0, 0, 3:7].tolist())
    print("  joints (first 10):", sd["mujoco_qpos"][0, 0, 7:17].tolist())
    print("  joints (last 10):", sd["mujoco_qpos"][0, 0, -10:].tolist())
    print("  total dof:", sd["mujoco_qpos"].shape[-1])

    print()
    print("=== motion_feature[0,0] (idle, frame 0) first 10 dims ===")
    print(" ", sd["motion_feature"][0, 0, :10].tolist())


if __name__ == "__main__":
    main()
