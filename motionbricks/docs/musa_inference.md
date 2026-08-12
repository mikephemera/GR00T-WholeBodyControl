# MUSA Inference

MotionBricks inference can run on a Moore Threads GPU using a vendor build of PyTorch and `torch_musa`. This adaptation covers the official G1 demo in FP32; it does not change the training, distributed, or AMP paths.

## Environment

Start with a working system installation of the vendor `torch` and `torch_musa`. Use a virtual environment with access to system packages so installing MotionBricks does not replace that runtime:

```bash
cd motionbricks
python -m virtualenv --system-site-packages ../.venv_motionbricks_musa
../.venv_motionbricks_musa/bin/python -m pip install -e . python-xlib
```

If `virtualenv` is unavailable, install it using the operating system package manager or the system Python's user package directory. The project pins MuJoCo 3.3.7 because MuJoCo 3.11 has been observed to segfault during Linux GLFW viewer teardown.

Verify the runtime before loading the checkpoints:

```bash
../.venv_motionbricks_musa/bin/python - <<'PY'
import torch
import torch_musa

assert torch.musa.is_available()
torch.musa.set_device(0)
print(torch.__version__)
print(torch.musa.get_device_name(0))
PY
```

The demo requires the four checkpoint files under `out/` and the G1 STL files under `assets/skeletons/g1/meshes/`. Restore these through Git LFS as described in the main README. When using a validated offline copy, copy only those checkpoint and mesh paths; do not overwrite the repository's source, configuration, or hyperparameter files.

## Run the official demo

`--device auto` is the default and selects MUSA, CUDA, then CPU. An explicit device is useful for validation:

```bash
# Interactive viewer
DISPLAY=:0 ../.venv_motionbricks_musa/bin/python \
  scripts/interactive_demo_g1.py --device musa:0

# Finite headless regression: exercises root, pose, and VQVAE inference
../.venv_motionbricks_musa/bin/python scripts/interactive_demo_g1.py \
  --device musa:0 --controller random --has_viewer 0 --max_steps 20
```

To produce the C++ bring-up golden trace at the same time, add
`--golden_dir ./out/golden_musa`. The trace has two layers (engineering
forward-kinematics/spring/conversion values and neural-network inputs/outputs),
with raw little-endian binaries and JSON shape/dtype manifests. See
[`docs/golden_data.md`](golden_data.md) for the file format and comparison
tolerances.

Successful startup prints a line similar to:

```text
MotionBricks inference device: musa:0 (M1000)
```

The demo also verifies that the inference model parameters are on the selected device. Check accelerator use during a run with `mthreads-gmi`.

## Troubleshooting

- **`torch.musa.is_available()` is false:** activate the environment that exposes the vendor `torch_musa` installation and confirm that the Moore Threads driver can see the GPU. Do not install upstream `torch` over the vendor build.
- **`NestedTensormusa` kernel error:** the pose and root encoders disable PyTorch's NestedTensor fast path and use the regular MUSA Transformer kernel. Confirm that the modified MotionBricks package is the one imported by the environment.
- **Viewer exits with a segmentation fault:** confirm `python -c 'import mujoco; print(mujoco.__version__)'` reports 3.3.7, then reinstall with `python -m pip install -e .` if necessary.
- **Wayland keyboard input fails:** use an XWayland display and keep the terminal focused. The demo automatically selects pyGLFW's X11 library when `DISPLAY` is set; it can also be forced with `PYGLFW_LIBRARY_VARIANT=x11`.
- **FP16/BF16 recommendation from the Transformer kernel:** this is an informational warning. The validated inference path uses FP32 intentionally.
