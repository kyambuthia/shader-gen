// Renderer. Concatenated after common.wgsl.

// Full-screen triangle: three vertices, no vertex buffer.
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn normal_at(p: vec3<f32>) -> vec3<f32> {
    // Tetrahedron trick: four taps instead of six.
    let e = vec2<f32>(1.0, -1.0) * 0.5773 * 0.0008;
    return normalize(
        e.xyy * sd_character(p + e.xyy) + e.yyx * sd_character(p + e.yyx) +
        e.yxy * sd_character(p + e.yxy) + e.xxx * sd_character(p + e.xxx));
}

fn world_normal_at(p: vec3<f32>) -> vec3<f32> {
    let e = vec2<f32>(1.0, -1.0) * 0.5773 * 0.0015;
    return normalize(
        e.xyy * sd_world(p + e.xyy) + e.yyx * sd_world(p + e.yyx) +
        e.yxy * sd_world(p + e.yxy) + e.xxx * sd_world(p + e.xxx));
}

struct Hit {
    t: f32,
    hit: bool,
}

// Phase 6: the march is clipped to the character's bounding box, so a ray
// that misses the body costs one slab test instead of 96 field evaluations.
fn march_character(origin: vec3<f32>, dir: vec3<f32>) -> Hit {
    let span = ray_bounds(origin, 1.0 / dir);
    if (span.y < span.x) {
        return Hit(0.0, false);
    }
    var t = span.x;
    for (var i = 0; i < 96; i = i + 1) {
        let d = sd_character(origin + dir * t);
        if (d < 0.0006 * t + 0.0004) {
            return Hit(t, true);
        }
        // 0.85, not 1.0: `smin` lets the field overestimate, and a full step
        // overshoots the surface and speckles the silhouette.
        t = t + d * 0.85;
        if (t > span.y) {
            break;
        }
    }
    return Hit(t, false);
}

fn march_world(origin: vec3<f32>, dir: vec3<f32>, limit: f32) -> Hit {
    var t = 0.02;
    for (var i = 0; i < 128; i = i + 1) {
        let d = sd_world(origin + dir * t);
        if (d < 0.0015 * t + 0.0008) {
            return Hit(t, true);
        }
        t = t + d * 0.95;
        if (t > limit) {
            break;
        }
    }
    return Hit(t, false);
}

// Soft shadow against the character and the world, tracking closest approach.
fn shadow(origin: vec3<f32>, dir: vec3<f32>) -> f32 {
    var result = 1.0;
    var t = 0.04;
    for (var i = 0; i < 40; i = i + 1) {
        let d = min(sd_character(origin + dir * t), sd_world(origin + dir * t));
        if (d < 0.0008) {
            return 0.0;
        }
        result = min(result, 9.0 * d / t);
        t = t + clamp(d * 0.9, 0.02, 0.5);
        if (t > 9.0) {
            break;
        }
    }
    return clamp(result, 0.0, 1.0);
}

fn ambient_occlusion(p: vec3<f32>, n: vec3<f32>) -> f32 {
    var occlusion = 0.0;
    var scale = 1.0;
    for (var i = 1; i <= 5; i = i + 1) {
        let h = 0.02 * f32(i) * f32(i);
        let d = min(sd_character(p + n * h), sd_world(p + n * h));
        occlusion = occlusion + (h - d) * scale;
        scale = scale * 0.68;
    }
    return clamp(1.0 - 1.4 * occlusion, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = (2.0 * frag.xy - globals.resolution) / globals.resolution.y;

    let eye = globals.eye.xyz;
    let forward = normalize(globals.look_at.xyz - eye);
    let right = normalize(cross(forward, vec3<f32>(0.0, 1.0, 0.0)));
    let up = cross(right, forward);
    // uv.y is flipped: framebuffer y grows downward.
    let dir = normalize(forward * 1.7 + right * uv.x - up * uv.y);

    let sky = mix(vec3<f32>(0.30, 0.37, 0.47), vec3<f32>(0.06, 0.09, 0.16),
                  clamp(dir.y * 1.4, 0.0, 1.0));
    var color = sky;
    let light = normalize(vec3<f32>(0.55, 0.78, 0.30));

    // The ground plane is intersected analytically rather than marched.
    // Sphere tracing an infinite plane converges slowly at grazing angles and
    // the distance-scaled hit epsilon then terminates rays at banded depths,
    // which reads as concentric rings around the camera. Boxes are marched.
    var plane_t = 1.0e9;
    if (dir.y < -1.0e-4) {
        plane_t = -eye.y / dir.y;
        if (plane_t < 0.0) { plane_t = 1.0e9; }
    }

    let body = march_character(eye, dir);
    let body_t = select(1.0e9, body.t, body.hit);
    let solids = march_world(eye, dir, min(min(plane_t, body_t), 60.0));
    let solid_t = select(1.0e9, solids.t, solids.hit);

    let nearest = min(min(body_t, solid_t), plane_t);
    if (nearest < 1.0e8) {
        var p = eye + dir * nearest;
        var n: vec3<f32>;
        var albedo: vec3<f32>;
        var ao = 1.0;

        if (nearest == body_t) {
            n = normal_at(p);
            albedo = vec3<f32>(0.82, 0.79, 0.74);
            ao = ambient_occlusion(p, n);
        } else if (nearest == solid_t) {
            n = world_normal_at(p);
            albedo = vec3<f32>(0.33, 0.36, 0.40);
            ao = ambient_occlusion(p, n);
        } else {
            n = vec3<f32>(0.0, 1.0, 0.0);
            // A grid so motion over flat ground is readable.
            let cell = abs(fract(p.xz * 0.5) - 0.5);
            let line = smoothstep(0.0, 0.03, min(cell.x, cell.y));
            albedo = mix(vec3<f32>(0.14, 0.15, 0.17),
                         vec3<f32>(0.26, 0.28, 0.32), line);
            ao = ambient_occlusion(p, n);
        }

        let diffuse = max(dot(n, light), 0.0) * shadow(p, light);
        let bounce = max(dot(n, vec3<f32>(0.0, -1.0, 0.0)), 0.0) * 0.10;
        let ambient = (0.15 + 0.14 * n.y) * ao;

        color = albedo * (ambient + bounce + diffuse * vec3<f32>(1.35, 1.28, 1.14));
        // Haze only bites in the far distance, so the arena keeps its contrast.
        color = mix(color, sky, pow(clamp(nearest / 70.0, 0.0, 1.0), 2.2));
    }

    color = pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));   // to sRGB
    return vec4<f32>(color, 1.0);
}
