# C++ MotionBricks final-qpos viewer

There are two C++ entry points under `motionbricks/cpp`:

- `motionbricks_ort_cli` is the lower-level three-network bridge. Python still
  performs MotionBricks preprocessing and postprocessing on that path.
- `motionbricks_planner_cli` is the final deployment interface used by the
  viewer. Its input is four 36-D MuJoCo qpos frames plus controller values; its
  output is the final future MuJoCo qpos sequence.

The final viewer does not construct `navigation_demo`, load PyTorch
checkpoints, or call Python canonicalization, spring/clip, FK, neural
orchestration, decoder postprocessing, or qpos conversion code. Python only
collects keyboard/camera input, keeps the four-frame playback context and
displays the C++ qpos in MuJoCo.

## Build the final planner server

Run from the `GR00T-WholeBodyControl` repository root:

```bash
cmake -S motionbricks/cpp -B motionbricks/cpp/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBFM_CONTROLLER_ROOT=/home/michael/Work-syncfree/bfm_controller \
  -DORT_ROOT=/home/michael/Work-syncfree/bfm_controller/third_party/onnxruntime-linux-aarch64-musa-sdk-5.1.0
cmake --build motionbricks/cpp/build --target motionbricks_planner_cli -j4
```

The current repository owns the thin server and viewer. The final planner
implementation, release asset pack and ONNX models are compiled/read from the
explicit `bfm_controller` path above; no files in that repository are changed.

## Continuous-replanning regression

This performs three final-interface calls. Each call consumes 16 frames and
uses the current frame plus the next three frames as the following C++ input:

```bash
PYTHONPATH=motionbricks .venv/bin/python \
  motionbricks/scripts/interactive_demo_g1_cpp.py \
  --headless \
  --segments 3 \
  --replan-frames 16 \
  --mode 2 \
  --device cpu
```

Every line must report `input=[4,36]`, an output shaped `[N,36]`, finite
MuJoCo FK and a valid root quaternion. This is a final-qpos test, not just a
three-network test.

## Manual viewer comparison

CPU:

```bash
PYTHONPATH=motionbricks PYGLFW_LIBRARY_VARIANT=x11 \
  .venv/bin/python motionbricks/scripts/interactive_demo_g1_cpp.py \
  --device cpu \
  --viewer-replan-mode async \
  --replan-frames 16
```

Strict MUSA verification:

```bash
PYTHONPATH=motionbricks PYGLFW_LIBRARY_VARIANT=x11 \
  .venv/bin/python motionbricks/scripts/interactive_demo_g1_cpp.py \
  --device musa \
  --require-musa \
  --viewer-replan-mode async \
  --replan-frames 16
```

`--require-musa` must be kept when claiming a MUSA run; otherwise provider
fallback must not be reported as MUSA success.

The viewer defaults to `--viewer-replan-mode async`. It submits the next
final-qpos request ahead of its handoff frame, continues displaying the old
buffer while C++ is running, and switches plans only at the exact four-frame
context boundary. The default three warmups reduce first-use MUSA latency.
Successful continuous runs should print `handoff=exact` and finish with
`async_missed=0` and `late_display_gaps=0`.

Use `--viewer-replan-mode sync` only as a blocking diagnostic baseline. It
calls the same C++ final interface, but planner time is paid in the render
loop and therefore appears as a visible pause. `--replan-lead-frames N` can
override the automatic planning lead when testing a different machine.

Controls:

- `W/A/S/D`: camera-relative movement; plain movement selects `walk` (mode 2).
- `V`: slow walk.
- `Z/X/B/R/T/C/E/F/G/Q`: the corresponding MotionBricks style modes from
  `clip_holder_G1`.
- Mouse drag: change camera-relative movement/facing direction.
- Close the viewer window to exit.

The default 16-frame interval matches the original demo's
`controller_dt * generate_dt * 30 FPS` setting. Add
`--replan-on-control-change` only when immediate key-response testing is more
important than matching that schedule.

For an original-versus-C++ visual comparison, run the original
`interactive_demo_g1.py` and this final-qpos viewer separately with the same
camera, key sequence and 16-frame replanning interval. Check:

1. idle stability and the transition from idle to `W` walking;
2. forward/backward and `A/D` lateral behavior, especially whether the final
   C++ implementation matches `blendspace_modes_remap_from_velocity`;
3. heading changes while rotating the viewer camera;
4. style-key pose selection;
5. visible jumps at each printed `replan=N` boundary and the return to idle.

Each replan log prints the exact final-interface shapes, selected mode/clip,
predicted token count, C++/wall latency and handoff continuity metrics. The
completion line reports the largest observed display gap and any late frames.
