#!/usr/bin/env python3
"""
Generate smooth transition motion between two fixed G1 poses using MotionBricks.

This script uses MotionBricks' neural networks (Root Model + Pose Model + VQVAE Decoder)
to generate intermediate frames between two static poses defined in CLAUDE_poses.md.

The model's predict() function natively supports "keyframe constraints":
    - First 4 frames = start pose (repeated)
    - Last 4 frames = end pose (repeated)
    - Model predicts root trajectory + pose tokens for intermediate frames

Usage:
    # Generate transition, export to unitree_sim2x CSV format
    python scripts/generate_pose_to_pose.py --start POSE1 --end POSE11

    # Specify number of tokens (1 token = 4 frames ≈ 0.133s)
    python scripts/generate_pose_to_pose.py --start POSE1 --end POSE11 --num-tokens 8

    # Auto-length (let Root Model decide)
    python scripts/generate_pose_to_pose.py --start POSE1 --end POSE11 --auto-length

    # Visualize in MuJoCo viewer without exporting
    python scripts/generate_pose_to_pose.py --start POSE1 --end POSE11 --view

    # List available poses
    python scripts/generate_pose_to_pose.py --list

Run from motionbricks/ directory.
"""

import os
import sys
import re
import time
import argparse
from typing import Dict, Optional, Tuple

import numpy as np
import torch as t
import mujoco
import mujoco.viewer

# --- Add project root to path ---
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from types import SimpleNamespace
from motionbricks.exp_setup.experiment import test
from motionbricks.motion_backbone.inference.motion_inference import motion_inference
from motionbricks.helper.mujoco_helper import get_mujoco_converter
from motionbricks.motionlib.core.utils.rotations import matrix_to_cont6d

# ==============================================================================
# Pose Parsing (compatible with CLAUDE_poses.md format)
# ==============================================================================

def parse_poses(filepath: str) -> Dict[str, Dict]:
    """Parse CLAUDE_poses.md and return dict of {name: {root_z, root_quat, joints}}.

    Joint order (29-DoF, matches unitree_sim2x SDK order):
        left_leg(6):  hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
        right_leg(6): hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
        waist(3):     yaw, roll, pitch
        left_arm(7):  shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
                      wrist_roll, wrist_pitch, wrist_yaw
        right_arm(7): shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
                      wrist_roll, wrist_pitch, wrist_yaw
    """
    with open(filepath) as f:
        content = f.read()

    poses = {}
    pattern = re.compile(r"(POSE\d+)\s*=\s*\{")
    blocks = pattern.split(content)

    for i in range(1, len(blocks), 2):
        name = blocks[i]
        body = blocks[i + 1]

        # root_z
        m = re.search(r'"root_z"\s*:\s*([-\d.e]+)', body)
        if not m:
            print(f"Warning: could not parse root_z for {name}, skipping")
            continue
        root_z = float(m.group(1))

        # root_quat
        m = re.search(r'"root_quat"\s*:\s*np\.array\(\s*\[([^\]]+)\]\)', body)
        if not m:
            print(f"Warning: could not parse root_quat for {name}, skipping")
            continue
        root_quat = np.fromstring(m.group(1), sep=",", dtype=float)

        # joints
        joint_start = body.index('"joints"')
        joint_body = body[joint_start:]
        values = re.findall(r'([-\d.e]+)\s*,', joint_body)
        if not values:
            print(f"Warning: could not parse joints for {name}, skipping")
            continue
        joints = np.array([float(v) for v in values], dtype=float)

        if len(joints) != 29:
            print(f"Warning: {name} has {len(joints)} joints, expected 29")

        poses[name] = {"root_z": root_z, "root_quat": root_quat, "joints": joints}

    return poses


def pose_to_qpos(pose_dict: Dict) -> np.ndarray:
    """Convert a pose dict to MuJoCo qpos array [36].

    qpos layout: root_pos(3) + root_quat[w,x,y,z](4) + joints(29) = 36
    """
    qpos = np.zeros(36)
    qpos[2] = pose_dict["root_z"]
    qpos[3:7] = pose_dict["root_quat"]  # w,x,y,z
    qpos[7:] = pose_dict["joints"]
    return qpos


# ==============================================================================
# Quaternion helpers (MuJoCo convention: [w, x, y, z])
# ==============================================================================

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions in [w,x,y,z] format."""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of quaternion in [w,x,y,z] format."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_heading(q: np.ndarray) -> float:
    """Extract heading (Z-rotation angle in radians) from quaternion [w,x,y,z]."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def quat_from_z_rotation(angle: float) -> np.ndarray:
    """Create quaternion [w,x,y,z] representing rotation around Z axis."""
    return np.array([np.cos(angle/2), 0.0, 0.0, np.sin(angle/2)])


def align_pose_heading(qpos: np.ndarray, target_heading: float) -> np.ndarray:
    """Rotate a qpos's root quaternion so its heading matches target_heading.

    Applies a Z-rotation correction to the root quaternion and adjusts
    the root XY position accordingly (rotates it in the ground plane).

    Args:
        qpos:       qpos array [36] with quaternion in [w,x,y,z] at indices 3:7
        target_heading: desired heading in radians

    Returns:
        new qpos with aligned heading
    """
    current_heading = quat_heading(qpos[3:7])
    correction = target_heading - current_heading
    q_correction = quat_from_z_rotation(correction)

    new_qpos = qpos.copy()
    # Apply Z-rotation to root quaternion (world-frame rotation)
    new_qpos[3:7] = quat_mul(q_correction, qpos[3:7])

    # Also rotate root XY position around Z
    cos_a, sin_a = np.cos(correction), np.sin(correction)
    rx, ry = qpos[0], qpos[1]
    new_qpos[0] = cos_a * rx - sin_a * ry
    new_qpos[1] = sin_a * rx + cos_a * ry

    return new_qpos


def quat_nlerp(q1: np.ndarray, q2: np.ndarray, alpha: float) -> np.ndarray:
    """Normalized linear interpolation (NLERP) of quaternions [w,x,y,z].

    Ensures shortest path by checking dot product sign.
    """
    dot = np.dot(q1, q2)
    if dot < 0:
        q2 = -q2  # take shortest path
    result = (1 - alpha) * q1 + alpha * q2
    # Normalize
    norm = np.linalg.norm(result)
    if norm > 1e-10:
        result = result / norm
    return result


# ==============================================================================
# Core: Pose-to-Pose Generator
# ==============================================================================

class PoseToPoseGenerator:
    """Generate transition motion between two fixed G1 poses.

    Uses MotionBricks' motion_inference.predict() with start/end keyframe
    constraints. The model natively supports this: it takes 8 constraint frames
    (4 context + 4 target) and generates intermediate frames between them.
    """

    NUM_FRAMES_PER_TOKEN = 4

    def __init__(self, inferencer: motion_inference, converter, motion_rep, device: str = 'cuda'):
        self.inferencer = inferencer.eval().to(device)
        self.converter = converter.to(device)
        self.motion_rep = motion_rep
        self.global_motion_rep = motion_rep.dual_rep.global_motion_rep
        self.local_motion_rep = motion_rep.dual_rep.local_motion_rep
        self.device = device
        self.fps = motion_rep.fps

    # ------------------------------------------------------------------
    # qpos → Model Features (following full_agent._generate_inbetween_frames pattern)
    # ------------------------------------------------------------------

    def qpos_to_features(self, qpos_np: np.ndarray) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
        """Convert a single qpos [36] to model features repeated over 4 frames.

        Args:
            qpos_np: MuJoCo qpos array [36]

        Returns:
            global_root:  [1, 4, 5]  (pos_xyz + heading_cos/sin)
            local_root:   [1, 4, 4]  (rot_vel + lin_vel_xz + root_y)
            local_pose:   [1, 4, 303] (relative_joint_pos + joint_rot_6d)
        """
        # Repeat single pose to 4 frames
        qpos = t.from_numpy(qpos_np).float().to(self.device).view(1, 1, -1)
        qpos = qpos.repeat(1, self.NUM_FRAMES_PER_TOKEN, 1)  # [1, 4, 36]

        # Convert qpos → motion-space joint transforms
        global_joint_positions, global_joint_rotations = \
            self.converter.convert_mujoco_qpos_to_motion_transforms(qpos)
        # [1, 4, 34, 3], [1, 4, 34, 3, 3]  (motion space: Y-up, Z-forward)

        # --- global_root_values [1, 4, 5] ---
        root_pos = global_joint_positions[:, :, 0, :]  # pelvis
        root_angle = t.atan2(
            global_joint_rotations[:, :, 0, 0, 2],   # Y-up: X component of Z-forward
            global_joint_rotations[:, :, 0, 2, 2],   # Y-up: Z component of Z-forward
        )
        global_root = t.cat([
            root_pos,
            t.cos(root_angle)[..., None],
            t.sin(root_angle)[..., None],
        ], dim=-1)  # [1, 4, 5]

        # --- local_root_values [1, 4, 4] ---
        local_root = t.zeros(1, self.NUM_FRAMES_PER_TOKEN, 4, device=self.device)
        # rot_vel: finite difference of heading angle
        local_root[:, :self.NUM_FRAMES_PER_TOKEN - 1, 0] = \
            (((root_angle[:, 1:] - root_angle[:, :-1] + t.pi) % (2 * t.pi)) - t.pi) * self.fps
        # lin_vel_xz: finite difference of root position (XZ plane)
        local_root[:, :self.NUM_FRAMES_PER_TOKEN - 1, 1:3] = \
            (root_pos[:, 1:, [0, 2]] - root_pos[:, :-1, [0, 2]]) * self.fps
        # root_y (absolute height)
        local_root[:, :, 3] = root_pos[:, :, 1]

        # --- local_poses [1, 4, 303] ---
        # Joint positions relative to root (XZ only, Y kept absolute)
        joint_pos = global_joint_positions[:, :, 1:, :].clone()  # exclude root → [1,4,33,3]
        joint_pos[..., 0] = joint_pos[..., 0] - root_pos[:, :, 0:1]  # relative X
        joint_pos[..., 2] = joint_pos[..., 2] - root_pos[:, :, 2:3]  # relative Z

        # Joint rotations as 6D continuous representation
        joint_rot_6d = matrix_to_cont6d(global_joint_rotations)  # [1, 4, 34, 6]

        local_pose = t.cat([
            joint_pos.reshape(1, self.NUM_FRAMES_PER_TOKEN, -1),    # 33*3 = 99
            joint_rot_6d.reshape(1, self.NUM_FRAMES_PER_TOKEN, -1),  # 34*6 = 204
        ], dim=-1)  # [1, 4, 303]

        return global_root, local_root, local_pose

    # ------------------------------------------------------------------
    # Constraint Construction
    # ------------------------------------------------------------------

    def build_constraints(
        self,
        start_features: Tuple[t.Tensor, t.Tensor, t.Tensor],
        end_features: Tuple[t.Tensor, t.Tensor, t.Tensor],
        strict_end: bool = True,
    ) -> Dict:
        """Build 8-frame constraint tensors for motion_inference.predict().

        Args:
            start_features: (global_root, local_root, local_pose) for start pose
            end_features:   (global_root, local_root, local_pose) for end pose
            strict_end:     If True, fully constrain end pose.
                            If False, relax end velocity constraints for smoother arrival.

        Returns:
            dict with keys: global_root_values, has_global_root_values,
                           local_root_values, has_local_root_values,
                           local_poses, has_local_poses
        """
        s_gr, s_lr, s_lp = start_features
        e_gr, e_lr, e_lp = end_features

        device = s_gr.device

        # Concatenate: [4 start + 4 end] = 8 frames
        global_root = t.cat([s_gr, e_gr], dim=1)
        local_root = t.cat([s_lr, e_lr], dim=1)
        local_pose = t.cat([s_lp, e_lp], dim=1)

        # Default: all frames have valid data (on same device as features)
        has_global_root = t.ones(1, 8, dtype=t.bool, device=device)
        has_local_root = t.ones(1, 8, dtype=t.bool, device=device)
        has_local_pose = t.ones(1, 8, dtype=t.bool, device=device)

        # Velocity at last frame of each 4-frame block is unreliable
        # (computed from static repeated poses → zero velocity, not real dynamics)
        has_local_root[:, self.NUM_FRAMES_PER_TOKEN - 1] = False  # frame 3
        has_local_root[:, -1] = False                              # frame 7

        if not strict_end:
            # Relax end constraints → model has more freedom for smooth arrival
            # Keep only the first frame of the end block as strict constraint
            has_global_root[:, -self.NUM_FRAMES_PER_TOKEN + 1:] = False
            has_local_root[:, -self.NUM_FRAMES_PER_TOKEN:] = False
            has_local_pose[:, -self.NUM_FRAMES_PER_TOKEN + 1:] = False

        return {
            'global_root_values': global_root,
            'has_global_root_values': has_global_root,
            'local_root_values': local_root,
            'has_local_root_values': has_local_root,
            'local_poses': local_pose,
            'has_local_poses': has_local_pose,
        }

    # ------------------------------------------------------------------
    # Main Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        start_qpos: np.ndarray,
        end_qpos: np.ndarray,
        num_tokens: Optional[int] = None,
        config: Optional[Dict] = None,
        strict_end: bool = True,
        align_heading: bool = True,
    ) -> Tuple[np.ndarray, int, int]:
        """Generate transition motion between two poses.

        Args:
            start_qpos:     Start pose qpos [36]
            end_qpos:       End pose qpos [36]
            num_tokens:     Number of tokens to generate (None = model decides).
                            1 token = 4 frames ≈ 0.133s @ 30fps.
            config:         Inference config dict (see motion_inference.predict()).
            strict_end:     If True, strictly enforce end pose constraints.
            align_heading:  If True, align end pose's heading to match start pose,
                            eliminating unnecessary yaw rotation during transition.

        Returns:
            qpos_seq:       Generated qpos sequence [T, 36]
            num_frames:     Total number of frames generated
            num_tokens_out: Number of tokens predicted
        """
        if config is None:
            config = {
                'num_inference_step': 5,
                'pose_token_sampling_use_argmax': True,
                'skip_ending_target_cond': False,
                'final_root_pred_mode': 'from_root_module',
                'use_constraints_at_decoder': True,
            }

        # --- Optional heading alignment ---
        # Extract start heading and align end pose to avoid unwanted yaw rotation.
        # The end pose's body configuration is preserved; only its world-frame
        # facing direction is adjusted to match the start pose.
        if align_heading:
            h_start = quat_heading(start_qpos[3:7])
            h_end = quat_heading(end_qpos[3:7])
            aligned_end_qpos = align_pose_heading(end_qpos, h_start)
            if abs(h_end - h_start) > 0.01:
                print(f"  Heading aligned: end {np.degrees(h_end):.1f}° -> "
                      f"{np.degrees(h_start):.1f}° (delta={np.degrees(h_end - h_start):.1f}°)")
        else:
            aligned_end_qpos = end_qpos

        # Convert poses to model features
        start_feat = self.qpos_to_features(start_qpos)
        end_feat = self.qpos_to_features(aligned_end_qpos)

        # Build 8-frame constraints
        constraints = self.build_constraints(start_feat, end_feat, strict_end=strict_end)

        # Prepare num_tokens
        num_tokens_t = None
        if num_tokens is not None:
            num_tokens_t = t.full([1, 1], num_tokens, dtype=t.int, device=self.device)

        # --- Run inference ---
        with t.no_grad():
            pred_global_motions, pred_num_tokens = self.inferencer.predict(
                constraints['global_root_values'],
                constraints['has_global_root_values'],
                constraints['local_root_values'],
                constraints['has_local_root_values'],
                constraints['local_poses'],
                constraints['has_local_poses'],
                num_tokens_t,
                config=config,
            )

        # --- Convert output to qpos ---
        qpos_seq = self.converter.convert_motion_features_to_mujoco_qpos(
            pred_global_motions, self.motion_rep, is_normalized=False
        )
        # Fix quaternion order: converter outputs [x,y,z,w] -> MuJoCo expects [w,x,y,z]
        root_rot = qpos_seq[:, :, 3:7].clone()
        qpos_seq[:, :, 3:7] = root_rot[:, :, [3, 0, 1, 2]]

        num_frames_total = pred_num_tokens.item() * self.NUM_FRAMES_PER_TOKEN
        qpos_out = qpos_seq[0].cpu().numpy()

        # Optionally snap endpoints to exact target poses
        if config.get('snap_endpoints', True):
            qpos_out = self._snap_endpoints(qpos_out, start_qpos, aligned_end_qpos)

        # Optionally stabilize heading across all frames.
        # The VQVAE decoder may introduce heading variation even with aligned
        # endpoints. Counter-rotate each frame to maintain constant heading.
        if align_heading:
            qpos_out = self._stabilize_heading(qpos_out, quat_heading(start_qpos[3:7]))

        return qpos_out, num_frames_total, pred_num_tokens.item()

    def _stabilize_heading(self, qpos_seq: np.ndarray, target_heading: float) -> np.ndarray:
        """Counter-rotate each frame to maintain constant heading.

        The VQVAE decoder's root rotation may drift from the desired heading.
        This post-processing step extracts the actual heading from each frame's
        root quaternion and applies a Z-rotation correction to match target_heading.
        """
        result = qpos_seq.copy()
        for i in range(len(result)):
            current_h = quat_heading(result[i, 3:7])
            correction = target_heading - current_h
            if abs(correction) < 1e-6:
                continue
            q_corr = quat_from_z_rotation(correction)
            result[i, 3:7] = quat_mul(q_corr, result[i, 3:7])
            # Also rotate root XY position around Z
            cos_a, sin_a = np.cos(correction), np.sin(correction)
            rx, ry = result[i, 0], result[i, 1]
            result[i, 0] = cos_a * rx - sin_a * ry
            result[i, 1] = sin_a * rx + cos_a * ry
        return result

    def _snap_endpoints(self, qpos_seq, start_qpos, end_qpos, blend_frames=4):
        """Snap first and last frames to exact target poses with smooth blending.

        Frame 0 = exact start pose, frame N-1 = exact end pose.
        Frames 1..blend_frames-1 blend from exact to generated.
        Uses NLERP for quaternion components to avoid rotation artifacts.
        """
        N = len(qpos_seq)
        if N < blend_frames * 2:
            return qpos_seq  # too short for blending

        result = qpos_seq.copy()

        # --- Start endpoint ---
        result[0] = start_qpos
        for i in range(1, blend_frames):
            alpha = i / blend_frames
            # Linear blend for position and joints
            result[i, :3] = (1 - alpha) * start_qpos[:3] + alpha * qpos_seq[i, :3]
            result[i, 7:] = (1 - alpha) * start_qpos[7:] + alpha * qpos_seq[i, 7:]
            # NLERP for quaternion (indices 3:7)
            result[i, 3:7] = quat_nlerp(start_qpos[3:7], qpos_seq[i, 3:7], alpha)

        # --- End endpoint ---
        result[-1] = end_qpos
        for i in range(1, blend_frames):
            alpha = i / blend_frames
            idx = N - 1 - i
            result[idx, :3] = (1 - alpha) * qpos_seq[idx, :3] + alpha * end_qpos[:3]
            result[idx, 7:] = (1 - alpha) * qpos_seq[idx, 7:] + alpha * end_qpos[7:]
            result[idx, 3:7] = quat_nlerp(qpos_seq[idx, 3:7], end_qpos[3:7], alpha)

        return result


# ==============================================================================
# CSV Export (unitree_sim2x compatible format)
# ==============================================================================

# Body indices for unitree_sim2x CSV format (matching export_fallen_sequence.py)
BODY_INDICES = [1, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29]


def compute_body_states(mj_model, mj_data, qpos: np.ndarray):
    """Run MuJoCo FK and return body positions & quaternions for body_indices."""
    mj_data.qpos[:] = qpos
    mujoco.mj_forward(mj_model, mj_data)
    body_pos = mj_data.xpos[BODY_INDICES].copy()
    body_quat = mj_data.xquat[BODY_INDICES].copy()  # w,x,y,z
    return body_pos, body_quat


def compute_velocities(recorded_qpos, body_pos_seq, body_quat_seq, fps=30):
    """Compute joint & body velocities via central finite differences."""
    from scipy.spatial.transform import Rotation as R

    num_frames = len(recorded_qpos)
    dt = 1.0 / fps
    nb = len(BODY_INDICES)

    joint_vel = np.zeros_like(recorded_qpos[:, 7:])
    body_lin_vel = np.zeros_like(body_pos_seq)
    body_ang_vel = np.zeros((num_frames, nb, 3))

    def _ang_vel(q_prev, q_next, dt_2):
        """Angular velocity from two quaternions separated by dt_2."""
        ang = np.zeros((nb, 3))
        for b in range(nb):
            rp = R.from_quat(q_prev[b, [1, 2, 3, 0]])
            rn = R.from_quat(q_next[b, [1, 2, 3, 0]])
            ang[b] = (rn * rp.inv()).as_rotvec() / dt_2
        return ang

    # Forward difference (frame 0)
    joint_vel[0] = (recorded_qpos[1, 7:] - recorded_qpos[0, 7:]) / dt
    body_lin_vel[0] = (body_pos_seq[1] - body_pos_seq[0]) / dt
    body_ang_vel[0] = _ang_vel(body_quat_seq[0], body_quat_seq[1], dt)

    # Central difference (frames 1..N-2)
    for i in range(1, num_frames - 1):
        joint_vel[i] = (recorded_qpos[i + 1, 7:] - recorded_qpos[i - 1, 7:]) / (2 * dt)
        body_lin_vel[i] = (body_pos_seq[i + 1] - body_pos_seq[i - 1]) / (2 * dt)
        body_ang_vel[i] = _ang_vel(body_quat_seq[i - 1], body_quat_seq[i + 1], 2 * dt)

    # Backward difference (frame N-1)
    joint_vel[-1] = (recorded_qpos[-1, 7:] - recorded_qpos[-2, 7:]) / dt
    body_lin_vel[-1] = (body_pos_seq[-1] - body_pos_seq[-2]) / dt
    body_ang_vel[-1] = _ang_vel(body_quat_seq[-2], body_quat_seq[-1], dt)

    return joint_vel, body_lin_vel, body_ang_vel


def export_csv(output_dir: str, motion_name: str,
               recorded_qpos: np.ndarray,
               body_pos_seq: np.ndarray,
               body_quat_seq: np.ndarray,
               joint_vel: np.ndarray,
               body_lin_vel: np.ndarray,
               body_ang_vel: np.ndarray):
    """Write unitree_sim2x-compatible CSV files."""
    out_dir = os.path.join(output_dir, motion_name)
    os.makedirs(out_dir, exist_ok=True)

    num_frames, nb = len(recorded_qpos), len(BODY_INDICES)

    def _write_csv(filename, data_2d, header):
        with open(os.path.join(out_dir, filename), "w") as f:
            f.write(header + "\n")
            np.savetxt(f, data_2d, delimiter=",", fmt="%.6f")

    # joint_pos.csv [T, 29]
    _write_csv("joint_pos.csv", recorded_qpos[:, 7:],
               ",".join(f"joint_{i}" for i in range(29)))

    # joint_vel.csv [T, 29]
    _write_csv("joint_vel.csv", joint_vel,
               ",".join(f"joint_{i}" for i in range(29)))

    # body_pos.csv [T, 14*3]
    _write_csv("body_pos.csv", body_pos_seq.reshape(num_frames, -1),
               ",".join(f"body_{i}_{a}" for i in range(nb) for a in ["x", "y", "z"]))

    # body_quat.csv [T, 14*4]
    _write_csv("body_quat.csv", body_quat_seq.reshape(num_frames, -1),
               ",".join(f"body_{i}_{q}" for i in range(nb) for q in ["w", "x", "y", "z"]))

    # body_lin_vel.csv [T, 14*3]
    _write_csv("body_lin_vel.csv", body_lin_vel.reshape(num_frames, -1),
               ",".join(f"body_{i}_{a}" for i in range(nb) for a in ["x", "y", "z"]))

    # body_ang_vel.csv [T, 14*3]
    _write_csv("body_ang_vel.csv", body_ang_vel.reshape(num_frames, -1),
               ",".join(f"body_{i}_{a}" for i in range(nb) for a in ["x", "y", "z"]))

    # info.txt
    with open(os.path.join(out_dir, "info.txt"), "w") as f:
        f.write(f"Motion: {motion_name} (pose-to-pose via MotionBricks)\n")
        f.write(f"Frames: {num_frames}, FPS: 30, Duration: {num_frames/30:.2f}s\n")
        f.write(f"Joint pos range: [{recorded_qpos[:, 7:].min():.3f}, {recorded_qpos[:, 7:].max():.3f}]\n")

    # metadata.txt
    with open(os.path.join(out_dir, "metadata.txt"), "w") as f:
        f.write(f"Metadata for: {motion_name}\n")
        f.write(f"Body part indexes: {BODY_INDICES}\n")
        f.write(f"Total timesteps: {num_frames}\n")

    print(f"Exported {num_frames} frames → {out_dir}/")


# ==============================================================================
# Model Loading
# ==============================================================================

def load_models(args) -> Tuple:
    """Load MotionBricks models without requiring dataset.

    Returns: (inferencer, converter, motion_rep)
    """
    # Build args for experiment.test()
    test_args = SimpleNamespace()
    test_args.result_dir = args.result_dir
    test_args.data_root = args.data_root
    test_args.EXP = args.planner
    test_args.planner = args.planner
    test_args.return_model_configs = True
    test_args.return_dataloader = False  # Skip dataset loading
    test_args.explicit_dataset_folder = None

    # Load pose and root models
    models, confs = test(test_args)

    # Load checkpoint weights
    for model_name in ['pose', 'root']:
        state_dict = t.load(confs[model_name].ckpt_path)['state_dict']
        models[model_name].load_state_dict(state_dict)

    # Create motion inference engine
    inferencer = motion_inference(models, models['pose'].args)

    # Get motion representation and converter
    motion_rep = models['pose'].motion_rep
    conv = get_mujoco_converter(motion_rep, args.skeleton_xml)

    return inferencer, conv, motion_rep


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate transition motion between two G1 poses using MotionBricks"
    )

    # --- Pose selection ---
    parser.add_argument("--start", type=str, default=None,
                        help="Start pose name (e.g. POSE1)")
    parser.add_argument("--end", type=str, default=None,
                        help="End pose name (e.g. POSE11)")
    parser.add_argument("--poses-file", type=str,
                        default="/mnt/datafiles/Work-syncfree/unitree_sim2x/CLAUDE_poses.md",
                        help="Path to CLAUDE_poses.md")
    parser.add_argument("--list", action="store_true",
                        help="List available pose names and exit")

    # --- Generation control ---
    parser.add_argument("--num-tokens", type=int, default=None,
                        help="Number of tokens to generate (1 token=4 frames). None=auto.")
    parser.add_argument("--auto-length", action="store_true",
                        help="Let the Root Model decide the number of tokens")
    parser.add_argument("--num-inference-steps", type=int, default=5,
                        help="Pose Model refinement iterations (1=fast, 10=quality)")
    parser.add_argument("--strict-end", type=int, default=1,
                        help="Strictly enforce end pose constraints (1=yes, 0=relaxed)")
    parser.add_argument("--snap-endpoints", type=int, default=1,
                        help="Snap first/last frames to exact start/end poses with blending (1=yes, 0=no)")
    parser.add_argument("--align-heading", type=int, default=1,
                        help="Align end pose heading to start pose to avoid unwanted yaw rotation (1=yes, 0=no)")

    # --- Output ---
    parser.add_argument("--output-dir", type=str,
                        default="/mnt/datafiles/Work-syncfree/unitree_sim2x/assets/motions/g1_29dof/csv",
                        help="Output directory for CSV export")
    parser.add_argument("--output-name", type=str, default=None,
                        help="Output subdirectory name (default: auto-generated)")
    parser.add_argument("--view", action="store_true",
                        help="Visualize in MuJoCo viewer (no CSV export)")
    parser.add_argument("--no-export", action="store_true",
                        help="Skip CSV export, just generate and print info")

    # --- Model paths ---
    parser.add_argument("--result-dir", type=str,
                        default=os.path.join(_project_root, "out"))
    parser.add_argument("--data-root", type=str,
                        default=os.path.join(_project_root, "datasets"))
    parser.add_argument("--skeleton-xml", type=str,
                        default=os.path.join(_project_root, "assets", "skeletons", "g1", "g1.xml"))
    parser.add_argument("--scene-xml", type=str,
                        default=os.path.join(_project_root, "assets", "skeletons", "g1", "scene_29dof.xml"))
    parser.add_argument("--planner", type=str, default="default")

    args = parser.parse_args()

    # --- Parse poses ---
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
            print(f"  {name}: root_z={p['root_z']:.4f}, "
                  f"quat=[{p['root_quat'][0]:.3f},{p['root_quat'][1]:.3f},"
                  f"{p['root_quat'][2]:.3f},{p['root_quat'][3]:.3f}], "
                  f"joints range=[{p['joints'].min():.3f}..{p['joints'].max():.3f}]")
        return

    # Validate pose selection
    if not args.start or not args.end:
        print("Error: --start and --end are required (use --list to see available poses)")
        sys.exit(1)

    for pname in [args.start, args.end]:
        if pname not in poses:
            print(f"Error: pose '{pname}' not found. Available: {list(poses.keys())}")
            sys.exit(1)

    start_pose = poses[args.start]
    end_pose = poses[args.end]

    # --- Load models ---
    print("Loading MotionBricks models...")
    inferencer, converter, motion_rep = load_models(args)
    device = 'cuda'
    print("Models loaded.")

    # --- Create generator ---
    generator = PoseToPoseGenerator(inferencer, converter, motion_rep, device=device)

    # --- Convert poses to qpos ---
    start_qpos = pose_to_qpos(start_pose)
    end_qpos = pose_to_qpos(end_pose)

    print(f"\nStart pose: {args.start}")
    print(f"  root_z={start_pose['root_z']:.4f}, "
          f"quat=[{start_pose['root_quat'][0]:.3f},{start_pose['root_quat'][1]:.3f},"
          f"{start_pose['root_quat'][2]:.3f},{start_pose['root_quat'][3]:.3f}]")
    print(f"End pose:   {args.end}")
    print(f"  root_z={end_pose['root_z']:.4f}, "
          f"quat=[{end_pose['root_quat'][0]:.3f},{end_pose['root_quat'][1]:.3f},"
          f"{end_pose['root_quat'][2]:.3f},{end_pose['root_quat'][3]:.3f}]")

    # --- Generate ---
    num_tokens = None if args.auto_length else args.num_tokens
    config = {
        'num_inference_step': args.num_inference_steps,
        'pose_token_sampling_use_argmax': True,
        'skip_ending_target_cond': False,
        'final_root_pred_mode': 'from_root_module',
        'use_constraints_at_decoder': True,
        'snap_endpoints': bool(args.snap_endpoints),
    }

    print(f"\nGenerating transition (num_tokens={num_tokens}, "
          f"inference_steps={args.num_inference_steps}, "
          f"strict_end={bool(args.strict_end)})...")

    t0 = time.time()
    qpos_seq, num_frames, num_tokens_out = generator.generate(
        start_qpos, end_qpos,
        num_tokens=num_tokens,
        config=config,
        strict_end=bool(args.strict_end),
        align_heading=bool(args.align_heading),
    )
    elapsed = time.time() - t0

    print(f"Generated {num_frames} frames ({num_tokens_out} tokens) "
          f"in {elapsed:.2f}s ({num_frames/elapsed:.1f} fps)")
    print(f"Duration: {num_frames / 30:.2f}s @ 30fps")
    print(f"Root Z range: [{qpos_seq[:, 2].min():.3f}, {qpos_seq[:, 2].max():.3f}]")
    print(f"Joint range: [{qpos_seq[:, 7:].min():.3f}, {qpos_seq[:, 7:].max():.3f}]")

    # --- View or Export ---
    if args.view:
        print("\nLaunching MuJoCo viewer...")
        mj_model = mujoco.MjModel.from_xml_path(args.scene_xml)
        mj_data = mujoco.MjData(mj_model)

        state = {"frame": 0, "playing": True, "quit": False}

        def key_cb(keycode):
            if keycode == 256 or keycode == 81:  # ESC or Q
                state["quit"] = True
            elif keycode == 32:  # Space
                state["playing"] = not state["playing"]
            elif keycode == 262:  # Right arrow
                state["frame"] = min(state["frame"] + 30, num_frames - 1)
            elif keycode == 263:  # Left arrow
                state["frame"] = max(state["frame"] - 30, 0)

        with mujoco.viewer.launch_passive(
            mj_model, mj_data, key_callback=key_cb,
            show_left_ui=False, show_right_ui=False
        ) as viewer:
            last_frame = -1
            while viewer.is_running() and not state["quit"]:
                if state["frame"] != last_frame or state["playing"]:
                    if state["playing"]:
                        state["frame"] = (state["frame"] + 1) % num_frames
                    mj_data.qpos[:] = qpos_seq[state["frame"]]
                    mujoco.mj_forward(mj_model, mj_data)
                    last_frame = state["frame"]

                viewer.sync()
                if state["playing"]:
                    time.sleep(1.0 / 30.0)

        print("Viewer closed.")

    if not args.no_export and not args.view:
        # --- Export CSV ---
        output_name = args.output_name
        if output_name is None:
            output_name = f"pose_{args.start}_to_{args.end}"

        print(f"\nComputing body states via MuJoCo FK...")
        mj_model = mujoco.MjModel.from_xml_path(args.scene_xml)
        mj_data = mujoco.MjData(mj_model)

        body_pos_seq = np.zeros((num_frames, len(BODY_INDICES), 3))
        body_quat_seq = np.zeros((num_frames, len(BODY_INDICES), 4))
        for i in range(num_frames):
            pos, quat = compute_body_states(mj_model, mj_data, qpos_seq[i])
            body_pos_seq[i] = pos
            body_quat_seq[i] = quat

        print("Computing velocities...")
        joint_vel, body_lin_vel, body_ang_vel = compute_velocities(
            qpos_seq, body_pos_seq, body_quat_seq, fps=30
        )

        print(f"Exporting to {args.output_dir}/{output_name}/ ...")
        export_csv(
            args.output_dir, output_name,
            qpos_seq, body_pos_seq, body_quat_seq,
            joint_vel, body_lin_vel, body_ang_vel,
        )

    print("Done.")


if __name__ == "__main__":
    main()
