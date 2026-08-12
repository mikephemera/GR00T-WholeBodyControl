# C++ golden data

The demo can persist a deterministic trace for the C++ MotionBricks port. Set
`--golden_dir` when running the headless regression:

```bash
python scripts/interactive_demo_g1.py \
  --device musa:0 --controller random --has_viewer 0 --max_steps 20 \
  --random_seed 1234 --golden_dir ./out/golden_musa
```

The directory contains a root `manifest.json` and one directory per generated
inference (`records/record_000000/`, ...). Each record has a `record.json` and
raw tensor files. A tensor file is contiguous, little-endian bytes in the
listed C-order `shape`; `dtype` uses NumPy notation (`<f4` = float32,
`<i4` = int32, `|b1` = bool). No pickle or NumPy `.npy` header is used.

Two layers are recorded:

* `basic/...` is the engineering pipeline: controller input, canonicalized
  context transforms, spring-model positions/headings, target clip forward
  kinematics, motion-feature conversion and final MuJoCo qpos.
* `nn/...` is the neural contract: normalized root/pose inputs and masks,
  root token logits/trajectory, every pose sampling iteration's tokens/logits,
  VQVAE decoder inputs/reconstruction, and final predicted motion features.

Compare C++ output with the same tensor's `shape` and `dtype` first, then use a
per-element absolute/relative tolerance appropriate for the backend. Token
indices and masks should be compared exactly; FP32 tensors normally use a
small tolerance (MUSA and CPU can differ in the last bits). The `metadata`
object records the device, controller interval, mode and whether qpos was used
as input, so a trace can be replayed with matching settings.

Recording is opt-in and has no effect when `--golden_dir` is omitted. The
recorder copies tensors to CPU only at the record boundary; it does not change
the model's device or dtype.
