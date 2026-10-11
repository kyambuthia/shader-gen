// Shared distance functions. Concatenated ahead of both character.wgsl and
// parity.wgsl at build time, so the renderer and the parity probe evaluate
// the same text rather than two copies of it.
//
// Every function here has a twin in Rust: sd_round_cone/smin/sd_character in
// src/sdf.rs, sd_round_box/sd_world in src/world.rs. tests/parity.rs is what
// stops them drifting.

struct Globals {
    resolution: vec2<f32>,
    time: f32,
    blend_k: f32,
    cone_count: u32,
    box_count: u32,
    _pad0: u32,
    _pad1: u32,
    eye: vec4<f32>,
    look_at: vec4<f32>,
    bounds_lo: vec4<f32>,
    bounds_hi: vec4<f32>,
    world_lo: vec4<f32>,
    world_hi: vec4<f32>,
}

// xyz is the endpoint, w is that end's radius.
struct Cone {
    a: vec4<f32>,
    b: vec4<f32>,
}

// center.xyz is the box, half.xyz its extents, half.w the corner radius.
struct RoundBox {
    center: vec4<f32>,
    half: vec4<f32>,
}

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<uniform> cones: array<Cone, 64>;       // MAX_CONES
@group(0) @binding(2) var<uniform> boxes: array<RoundBox, 16>;   // MAX_BOXES

// Exact distance to a round cone: two spherical caps and the conical band
// between them, nearest wins.
fn sd_round_cone(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, r1: f32, r2: f32) -> f32 {
    let ba = b - a;
    let l2 = dot(ba, ba);
    let rr = r1 - r2;
    let a2 = l2 - rr * rr;
    let il2 = 1.0 / l2;

    let pa = p - a;
    let y = dot(pa, ba);
    let z = y - l2;
    let cross_term = pa * l2 - ba * y;
    let x2 = dot(cross_term, cross_term);
    let y2 = y * y * l2;
    let z2 = z * z * l2;

    let k = sign(rr) * rr * rr * x2;
    if (sign(z) * a2 * z2 > k) {
        return sqrt(x2 + z2) * il2 - r2;
    }
    if (sign(y) * a2 * y2 < k) {
        return sqrt(x2 + y2) * il2 - r1;
    }
    return (sqrt(x2 * a2 * il2) + y * rr) * il2 - r1;
}

// Polynomial smooth minimum. Fuses joints, at the cost of the field no longer
// being a true distance - hence the step factor wherever this is marched.
fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

fn sd_character(p: vec3<f32>) -> f32 {
    var d = 0.0;
    for (var i = 0u; i < globals.cone_count; i = i + 1u) {
        let c = cones[i];
        let s = sd_round_cone(p, c.a.xyz, c.b.xyz, c.a.w, c.b.w);
        if (i == 0u) {
            d = s;
        } else {
            d = smin(d, s, globals.blend_k);
        }
    }
    return d;
}

fn sd_round_box(p: vec3<f32>, center: vec3<f32>, half: vec3<f32>, radius: f32) -> f32 {
    let q = abs(p - center) - half;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - radius;
}

// The boxes alone. The ground plane is handled analytically by the renderer
// and cannot shadow anything above itself, so keeping it out of the marched
// field saves a step budget on every sky and distant-ground ray.
fn sd_boxes(p: vec3<f32>) -> f32 {
    var d = 1.0e9;
    for (var i = 0u; i < globals.box_count; i = i + 1u) {
        let b = boxes[i];
        d = min(d, sd_round_box(p, b.center.xyz, b.half.xyz, b.half.w));
    }
    return d;
}

// The full field, boxes and ground. This is what collision queries, and what
// src/world.rs must stay identical to.
fn sd_world(p: vec3<f32>) -> f32 {
    return min(p.y, sd_boxes(p));
}

// Slab test against an axis-aligned box. Returns the entry and exit distances
// along the ray, or exit < entry when the ray misses.
fn ray_slab(origin: vec3<f32>, inv_dir: vec3<f32>, lo_corner: vec3<f32>,
            hi_corner: vec3<f32>) -> vec2<f32> {
    let t0 = (lo_corner - origin) * inv_dir;
    let t1 = (hi_corner - origin) * inv_dir;
    let lo = min(t0, t1);
    let hi = max(t0, t1);
    return vec2<f32>(max(max(lo.x, lo.y), max(lo.z, 0.0)),
                     min(hi.x, min(hi.y, hi.z)));
}

fn ray_bounds(origin: vec3<f32>, inv_dir: vec3<f32>) -> vec2<f32> {
    return ray_slab(origin, inv_dir, globals.bounds_lo.xyz, globals.bounds_hi.xyz);
}

fn ray_world_bounds(origin: vec3<f32>, inv_dir: vec3<f32>) -> vec2<f32> {
    return ray_slab(origin, inv_dir, globals.world_lo.xyz, globals.world_hi.xyz);
}

// Distance from a point to the character's bounding box. Nineteen cones is
// by far the most expensive thing in the scene, so shadow and occlusion rays
// use this to decide whether the body is even worth evaluating.
fn bounds_distance(p: vec3<f32>) -> f32 {
    let q = max(globals.bounds_lo.xyz - p, p - globals.bounds_hi.xyz);
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}
