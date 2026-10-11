# shader-gen

A playable character defined entirely as a signed distance function. The same
`f(p)` the fragment shader sphere-traces is the field collision queries, so the
thing you see and the thing you bump into cannot drift apart.

Rust + wgpu. See [`docs/DESIGN.md`](docs/DESIGN.md) for why, and for the limits
of the approach.

## Run

```bash
cargo run --release
```

| | |
| --- | --- |
| `W A S D` | move |
| `Shift` | run |
| `← →` | turn |
| `↑ ↓` | camera pitch |
| `Space` | jump |
| `N` | next generated body |
| `Esc` | quit |

Render a frame without a display:

```bash
cargo run --release -- --headless frame.png --seed 7 --settle 420
```

## Test

```bash
cargo test --release
```

The important one is `cpu_and_gpu_evaluators_agree`. The SDF is evaluated
twice - in WGSL to draw it, in Rust to collide with it - and two
implementations of one definition drift. The test samples a few hundred points
across several generated bodies, evaluates both, and fails the build if they
disagree by more than a millimetre. Without it, a divergence shows up weeks
later as a physics bug nobody connects to a rendering change.

## Layout

| | |
| --- | --- |
| `src/sdf.rs` | bones, posing, the CPU evaluator |
| `src/body.rs` | seed to proportions to skeleton |
| `src/world.rs` | ground and boxes, CPU side |
| `src/player.rs` | collision, locomotion, walk cycle |
| `src/gpu.rs` | buffers, render pipeline, parity probe |
| `shaders/common.wgsl` | the distance functions, shared by both pipelines |
| `shaders/character.wgsl` | the renderer |
| `shaders/parity.wgsl` | the GPU half of the parity test |
