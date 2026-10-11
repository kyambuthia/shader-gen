// Sphere-traced character. The cone list comes from the CPU (src/sdf.rs),
// which is the single source of truth for the shape; `sd_character` here must
// stay numerically identical to `CharacterDef::distance` there.

struct Globals {
    resolution: vec2<f32>,
    time: f32,
    blend_k: f32,
    cone_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// xyz is the endpoint, w is that end's radius.
struct Cone {
    a: vec4<f32>,
    b: vec4<f32>,
}

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<uniform> cones: array<Cone, 64>;  // MAX_CONES in src/sdf.rs

// Full-screen triangle: three vertices, no vertex buffer.
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

// Exact distance to a round cone, by the standard derivation: the segment
// splits into two spherical caps and the conical band between them.
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
// being a true distance - hence the step factor in `march`.
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

fn normal_at(p: vec3<f32>) -> vec3<f32> {
    // Tetrahedron trick: four taps instead of six.
    let e = vec2<f32>(1.0, -1.0) * 0.5773 * 0.0008;
    return normalize(
        e.xyy * sd_character(p + e.xyy) + e.yyx * sd_character(p + e.yyx) +
        e.yxy * sd_character(p + e.yxy) + e.xxx * sd_character(p + e.xxx));
}

struct Hit {
    t: f32,
    hit: bool,
}

fn march(origin: vec3<f32>, dir: vec3<f32>) -> Hit {
    var t = 0.0;
    for (var i = 0; i < 128; i = i + 1) {
        let p = origin + dir * t;
        let d = sd_character(p);
        if (d < 0.0008 * t + 0.0005) {
            return Hit(t, true);
        }
        // 0.85, not 1.0: `smin` lets the field overestimate, and a full step
        // overshoots the surface and speckles the silhouette.
        t = t + d * 0.85;
        if (t > 40.0) {
            break;
        }
    }
    return Hit(t, false);
}

// Soft shadow by marching toward the light and tracking the closest approach.
fn shadow(origin: vec3<f32>, dir: vec3<f32>) -> f32 {
    var result = 1.0;
    var t = 0.02;
    for (var i = 0; i < 48; i = i + 1) {
        let d = sd_character(origin + dir * t);
        if (d < 0.0005) {
            return 0.0;
        }
        result = min(result, 10.0 * d / t);
        t = t + clamp(d * 0.85, 0.01, 0.4);
        if (t > 8.0) {
            break;
        }
    }
    return clamp(result, 0.0, 1.0);
}

fn ambient_occlusion(p: vec3<f32>, n: vec3<f32>) -> f32 {
    var occlusion = 0.0;
    var scale = 1.0;
    for (var i = 1; i <= 5; i = i + 1) {
        let h = 0.015 * f32(i) * f32(i);
        occlusion = occlusion + (h - sd_character(p + n * h)) * scale;
        scale = scale * 0.65;
    }
    return clamp(1.0 - 1.6 * occlusion, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = (2.0 * frag.xy - globals.resolution) / globals.resolution.y;

    // Slow orbit so the silhouette reads as solid from every side.
    let angle = globals.time * 0.35;
    let eye = vec3<f32>(sin(angle) * 3.4, 1.35, cos(angle) * 3.4);
    let look_at = vec3<f32>(0.0, 0.95, 0.0);

    let forward = normalize(look_at - eye);
    let right = normalize(cross(forward, vec3<f32>(0.0, 1.0, 0.0)));
    let up = cross(right, forward);
    // uv.y is flipped: framebuffer y grows downward.
    let dir = normalize(forward * 1.7 + right * uv.x - up * uv.y);

    let sky = mix(vec3<f32>(0.10, 0.12, 0.16), vec3<f32>(0.02, 0.03, 0.05),
                  clamp(dir.y, 0.0, 1.0));
    var color = sky;

    let light = normalize(vec3<f32>(0.6, 0.8, 0.35));

    // The ground is intersected analytically, not marched. Sphere tracing an
    // infinite plane converges slowly at grazing angles, and the distance-
    // scaled hit epsilon then terminates rays at banded depths, which shows up
    // as concentric rings around the camera.
    var ground_t = -1.0;
    if (dir.y < -1.0e-4) {
        ground_t = -eye.y / dir.y;
    }

    let body = march(eye, dir);
    let hit_body = body.hit && (ground_t < 0.0 || body.t < ground_t);
    let hit_ground = ground_t > 0.0 && ground_t < 40.0 && !hit_body;

    if (hit_body || hit_ground) {
        let t = select(ground_t, body.t, hit_body);
        let p = eye + dir * t;
        let n = select(vec3<f32>(0.0, 1.0, 0.0), normal_at(p), hit_body);
        let albedo = select(vec3<f32>(0.16, 0.17, 0.19),
                            vec3<f32>(0.80, 0.78, 0.74), hit_body);

        let diffuse = max(dot(n, light), 0.0) * shadow(p, light);
        let bounce = max(dot(n, vec3<f32>(0.0, -1.0, 0.0)), 0.0) * 0.12;
        // Only the character occludes; a flat plane cannot occlude itself.
        let ao = select(1.0, ambient_occlusion(p, n), hit_body);
        let ambient = (0.22 + 0.18 * n.y) * ao;

        color = albedo * (ambient + bounce + diffuse * vec3<f32>(1.0, 0.96, 0.88));
        color = mix(color, sky, clamp(t / 40.0, 0.0, 1.0));   // distance haze
    }

    color = pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));   // to sRGB
    return vec4<f32>(color, 1.0);
}
