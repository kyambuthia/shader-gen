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
