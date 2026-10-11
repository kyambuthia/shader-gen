# shader-gen design

A playable character defined entirely as a signed distance function, rendered
by sphere tracing and collided against by evaluating the same function.
Standalone, Rust + wgpu.

## Why an SDF

A signed distance function is the one representation where rendering and
collision are *the same expression*:

| need | how |
| --- | --- |
| render | sphere trace: step along the ray by `f(p)`, stop when `f(p) < eps` |
| collide | `f(p) < 0` is penetrating, `-f(p)` is depth, `grad f` is the push-out normal |
| ground query | sphere trace downward, same loop |
| normal, AO, soft shadow | gradient, or cheap resampling of the same `f` |

In a conventional pipeline the visual mesh and the collision mesh are separate
artifacts that drift apart. Here they are one expression and *cannot* disagree.
That property, not the visual style, is the reason for this project.

## Shape of the thing

The character is data, not code. A seed produces roughly twenty numbers:

```rust
struct Bone { parent: u8, offset: Vec3, len: f32, r_head: f32, r_tail: f32 }
struct CharacterDef { bones: Vec<Bone>, blend_k: f32 }
```

Round cones - capsules with a different radius at each end - are the primitive.
They taper like limbs and have an exact distance function. The union is a
polynomial smooth minimum so joints fuse rather than intersect:

```wgsl
fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}
```

## The central risk: two evaluators

The GPU raymarches the field. The CPU evaluates it for collision. That is two
implementations of the function whose entire point was being a single source of
truth, and they will drift.

The discipline that fixes it is a **parity test**: sample several thousand
points across the character's bounding volume, evaluate on both sides, fail the
build if they diverge beyond epsilon. This runs in CI. Drift becomes a red
build rather than a rendering artifact nobody connects to a physics bug three
weeks later.

Pose travels to the GPU as a uniform buffer of rigid bone transforms. Both
sides read the same `CharacterDef`.

## Collision

The detailed field is for pixels. The bones are for physics.

Collision does not sample the smooth-minned surface. Each bone's round cone is
its own collision shape: query the world at the bone endpoints, push out along
the gradient. Roughly thirty queries per frame, microseconds on CPU. Because
both come from the same `CharacterDef`, the render surface and the collision
shapes cannot disagree about where the character is.

## Known limits

These are properties of the representation, not bugs to fix later.

- **Animation is worse than skinning.** You cannot skin an SDF. Posing
  inverse-transforms the sample point into each bone's space and takes the
  smooth minimum. Joints look blobby. That is inherent. The style has to want
  it.
- **`smin` breaks the distance property.** The field stops being a true
  distance and can overestimate, so sphere tracing overshoots and speckles the
  surface. Step at 0.8-0.9x the reported distance; costs about 20% more steps.
- **Bone transforms must stay rigid.** Non-uniform scale breaks distance
  correctness. Vary the cone radii instead of the transform.
- **Cost must be bounded by a proxy.** Fullscreen at 1080p, 64 steps, 20
  primitives is about 15 G evaluations per second at 60 fps, which is hopeless.
  Rasterize a bounding box and raymarch only the fragments inside it, starting
  the ray at the box face. Per-limb bounding spheres to skip distant bones in
  the inner loop are the single biggest further win. Integrated graphics will
  still struggle.
- **It fights the rasterizer.** Writing depth from the fragment shader disables
  early-Z. Shadow maps, MSAA and sorting all need deliberate handling.

## Build order

Riskiest first. Steps 2 and 3 are where this either works or does not;
everything before is a tech demo and everything after is polish.

1. Static character, fullscreen raymarch, no animation. Proves the primitive
   set and the look.
2. CPU evaluator and the parity test in CI. Proves the shared-truth approach
   before anything depends on it.
3. Collision and locomotion: gravity, ground trace, push-out, a capsule that
   walks on an SDF world. This is "playable" and the real proof.
4. Pose and animation: bone transforms, a walk cycle.
5. Procedural generation: seed to `CharacterDef`. Deliberately last. It is the
   cheapest part once the pipeline exists, and doing it early tempts you to
   tune parameters before you can see them move.
6. Optimisation: bounding proxy, limb culling, half-resolution raymarch with
   upscale.

## Rejected alternative

Evaluating the SDF in a compute shader, running marching cubes on the GPU and
baking a mesh plus skeleton. Lower risk, integrates with any engine or physics
library, and still gives procedural generation - but it puts the mesh and the
collider back into separate artifacts, which is the problem the SDF was chosen
to solve. Kept as the fallback if the runtime cost proves unworkable.

## Status

All six phases are implemented. What each one actually turned into:

1. **Render** - sphere tracer with soft shadows and AO. The ground plane is
   intersected analytically rather than marched; tracing an infinite plane
   converges slowly at grazing angles and the distance-scaled hit epsilon then
   terminates rays at banded depths, which reads as concentric rings.
2. **Parity** - `tests/parity.rs`, run over five seeds, tolerance 1e-3 m. The
   GPU half is a *fragment* shader writing to a 512x1 target, not a compute
   pass: downlevel GL adapters report zero compute workgroups and refuse one.
   It encodes each f32 as its raw bit pattern through `Rgba8Unorm`, because
   `Rgba32Float` is not renderable everywhere either, and a unorm8 byte
   round-trips exactly so nothing is lost.
3. **Collision** - capsule fitted to the character's own resolved bounds,
   pushed out of the world along the gradient, four solver passes. Plus
   step-up: a capsule cannot climb anything taller than its radius, so a
   blocked grounded move is retried from a lifted start and dropped back onto
   whatever is underneath. Without it the character stops dead at a 0.3 m kerb.
4. **Animation** - the skeleton is authored as head/tail pairs in model space
   and converted to parent-relative rotations at construction, because nobody
   can reason about a chain of local quaternions. Posing resolves to a flat
   list of world-space cones, so the shader never learns what a bone is.
   Stride advances with distance travelled, not time, so the gait stays locked
   to the ground at any speed.
5. **Generation** - a seed picks height, bulk, leg ratio, shoulder width, head
   and arm scale; every radius and joint derives from those. Proved out last,
   as planned.
6. **Optimisation** - the character march is clipped to its bounding box by a
   slab test, so a ray that misses the body costs one test instead of ~96
   field evaluations.


## Measured

`--bench` on the only machine this has run on, a Mesa Intel HD Graphics 3000
(Sandy Bridge, 2011, GL backend, no compute):

| resolution | before | after |
| --- | --- | --- |
| 512x320 | - | 33.8 ms (29.6 fps) |
| 640x400 | 125.2 ms (8.0 fps) | 49.1 ms (20.4 fps) |
| 1024x640 | timed out | 115.7 ms (8.6 fps) |

The window renders at `--scale` (default 2), so a 1024x640 window costs the
512x320 figure. 2.5x came from three changes, in order of value:

1. **Shadow rays stopped evaluating the body everywhere.** Nineteen cones at
   every step of every shadow ray was the largest cost in the frame; most
   ground pixels cast a ray that never passes near the character and were
   paying for it anyway. Now the body is only sampled over the stretch of ray
   that crosses its bounding box.
2. **The box march is clipped to the boxes' collective bounds,** and the
   ground plane was taken out of the marched field entirely - it is
   intersected analytically and cannot shadow anything above itself. A ray
   aimed at the sky now costs one slab test instead of a full step budget.
3. **Half-resolution rendering with a bilinear upscale.** Everything else
   trims work per ray; this removes three quarters of the rays.

Occlusion got the same bounding-box treatment as shadows.

### Still not done

- **No proxy geometry.** The fragment shader still runs over the whole
  screen. With the march clipped, a missing ray is cheap, so this is worth
  much less than it looked before measuring - which is the point of measuring.
- **Only ever run on one GPU,** and an unusually weak one. The format and
  limit choices throughout are shaped by its downlevel GL backend.
- **The windowed path is written but has never been executed.** Everything
  here was verified through `--headless`, `--bench` and the test suite.
