"""Attach the standalone C++ ONNX backend to MotionBricks inference.

The regular MotionBricks Python inference object remains responsible for
feature normalization, token selection and motion-representation conversion.
Only its three neural modules are replaced with proxies that speak the fixed
binary protocol implemented by ``cpp/motionbricks_ort_cli``.
"""

from __future__ import annotations

import struct
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import nn


_FRAME = struct.Struct("<IHHII")
_REQUEST_MAGIC = 0x4D424351
_RESPONSE_MAGIC = 0x4D424352
_PROTOCOL_VERSION = 1
_ROOT, _POSE, _DECODER, _SHUTDOWN = 1, 2, 3, 255


def _array(value, dtype):
    return np.ascontiguousarray(value.detach().to("cpu").numpy(), dtype=dtype)


class _CppProcess:
    def __init__(self, cli: Path, onnx_dir: Path, device_id: int = -1):
        self.process = subprocess.Popen(
            [str(cli), "--onnx-dir", str(onnx_dir), "--device-id", str(device_id)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0,
        )

    def _read_exact(self, size: int) -> bytes:
        data = bytearray()
        while len(data) < size:
            chunk = self.process.stdout.read(size - len(data))
            if not chunk:
                raise RuntimeError(
                    f"MotionBricks C++ backend exited (code {self.process.poll()})"
                )
            data.extend(chunk)
        return bytes(data)

    def run(self, operation: int, arrays: list[np.ndarray]) -> list[np.ndarray]:
        payload = b"".join(array.tobytes(order="C") for array in arrays)
        self.process.stdin.write(
            _FRAME.pack(_REQUEST_MAGIC, _PROTOCOL_VERSION, operation, len(payload), 0)
            + payload
        )
        self.process.stdin.flush()
        magic, version, response_op, payload_bytes, status = _FRAME.unpack(
            self._read_exact(_FRAME.size)
        )
        if (magic, version, response_op) != (
            _RESPONSE_MAGIC,
            _PROTOCOL_VERSION,
            operation,
        ):
            raise RuntimeError("invalid MotionBricks C++ response header")
        response = self._read_exact(payload_bytes)
        if status:
            raise RuntimeError(response.decode("utf-8", errors="replace"))
        if operation == _ROOT:
            spec = ((12, np.float32), (1, np.int64), (64 * 5, np.float32))
        elif operation == _POSE:
            spec = ((16 * 8 * 10, np.float32),)
        elif operation == _DECODER:
            spec = ((64 * 413, np.float32),)
        else:
            return []
        outputs, offset = [], 0
        for count, dtype in spec:
            size = count * np.dtype(dtype).itemsize
            if offset + size > len(response):
                raise RuntimeError("truncated MotionBricks C++ response payload")
            outputs.append(
                np.frombuffer(response, dtype=dtype, count=count, offset=offset)
                .copy()
            )
            offset += size
        if offset != len(response):
            raise RuntimeError("unexpected bytes in MotionBricks C++ response")
        return outputs

    def close(self):
        if self.process.poll() is not None:
            return
        try:
            self.process.stdin.write(
                _FRAME.pack(_REQUEST_MAGIC, _PROTOCOL_VERSION, _SHUTDOWN, 0, 0)
            )
            self.process.stdin.flush()
            self._read_exact(_FRAME.size)
        except (OSError, RuntimeError):
            self.process.kill()
        finally:
            self.process.wait(timeout=5)


class _RootProxy(nn.Module):
    IS_MODEL_TOKENIZED = False
    MASKED_NUM_TOKENS = 18

    def __init__(self, process):
        super().__init__()
        self.process = process
        self.OUT_OF_REACH_NUM_TOKENS = 17
        self.initted = torch.tensor(True)

    def get_num_frames_per_token(self):
        return 4

    def __call__(
        self,
        global_root_values,
        has_global_root_values,
        local_root_values,
        has_local_root_values,
        local_poses,
        has_local_poses,
        num_tokens,
        text_embeddings=None,
        has_text_embeddings=None,
        allowed_pred_num_tokens=None,
        config=None,
    ):
        del text_embeddings, has_text_embeddings, config
        if global_root_values.shape[0] != 1:
            raise ValueError("C++ MotionBricks backend currently supports batch size 1")
        if allowed_pred_num_tokens is None:
            allowed_pred_num_tokens = torch.ones((1, 11), dtype=torch.int64)
        out = self.process.run(
            _ROOT,
            [
                _array(global_root_values, np.float32),
                _array(has_global_root_values, np.bool_),
                _array(local_root_values, np.float32),
                _array(has_local_root_values, np.bool_),
                _array(local_poses, np.float32),
                _array(has_local_poses, np.bool_),
                _array(num_tokens, np.int64),
                _array(allowed_pred_num_tokens, np.int64),
            ],
        )
        return {
            "num_token_logits": torch.from_numpy(out[0]).reshape(1, 12).to(num_tokens.device),
            "pred_num_tokens": torch.from_numpy(out[1]).reshape(1, 1).to(num_tokens.device),
            "pred_global_root_values": torch.from_numpy(out[2]).reshape(1, 64, 5).to(global_root_values.device),
        }


class _PoseProxy(nn.Module):
    POSE_MASK_ID = 10

    def __init__(self, process):
        super().__init__()
        self.process = process
        self.initted = torch.tensor(True)

    def get_num_frames_per_token(self):
        return 4

    def get_num_heads(self):
        return (8,)

    def __call__(self, pose_tokens, local_root_values, pose_cond, has_pose_cond,
                 num_tokens, text_embeddings=None, has_text_embeddings=None):
        del text_embeddings, has_text_embeddings
        if pose_tokens.shape[0] != 1:
            raise ValueError("C++ MotionBricks backend currently supports batch size 1")
        out = self.process.run(
            _POSE,
            [
                _array(pose_tokens, np.int64),
                _array(local_root_values, np.float32),
                _array(pose_cond, np.float32),
                _array(has_pose_cond, np.bool_),
                _array(num_tokens, np.int64),
            ],
        )
        return {"pose_logits": torch.from_numpy(out[0]).reshape(1, 16, 8, 10).to(pose_tokens.device)}


class _DecoderProxy(nn.Module):
    def __init__(self, process, motion_rep):
        super().__init__()
        self.process = process
        self.motion_rep = motion_rep
        self.decoder_external_cond_feature_mode = "root_without_hip_height_without_heading"
        self.decoder_target_cond_feature_mode = "joint_positions_and_rotations_and_hip_height"

    def forward_decoder(self, tokens, target_cond, has_target_cond, external_cond,
                        use_overall_indices=False, token_mask=None):
        del use_overall_indices
        if tokens.shape[0] != 1:
            raise ValueError("C++ MotionBricks backend currently supports batch size 1")
        out = self.process.run(
            _DECODER,
            [
                _array(tokens, np.int64),
                _array(external_cond, np.float32),
                _array(target_cond, np.float32),
                _array(has_target_cond, np.bool_),
                _array(token_mask, np.bool_),
            ],
        )
        return {"recon_state": torch.from_numpy(out[0]).reshape(1, 64, 413).to(tokens.device)}


def attach_cpp_backend(inferencer, cli: str | Path, onnx_dir: str | Path,
                       device_id: int = -1):
    """Replace neural calls in an existing MotionBricks inference object."""
    process = _CppProcess(Path(cli), Path(onnx_dir), device_id)
    inferencer._root_model.backbone_net = _RootProxy(process)
    inferencer._pose_model.backbone_net = _PoseProxy(process)
    inferencer._vqvae_pose_model = _DecoderProxy(process, inferencer.local_motion_rep)
    inferencer._cpp_process = process
    # full_navigation_agent queries this proxy for constants and moves the
    # inference object to the selected torch device during construction.
    inferencer._root_model.backbone_net.OUT_OF_REACH_NUM_TOKENS = 17
    inferencer._root_model.backbone_net.to = lambda *args, **kwargs: inferencer._root_model.backbone_net
    inferencer._pose_model.backbone_net.to = lambda *args, **kwargs: inferencer._pose_model.backbone_net
    return inferencer


def close_cpp_backend(inferencer):
    process = getattr(inferencer, "_cpp_process", None)
    if process is not None:
        process.close()
