// Phase 2: the GPU half of the parity test. Concatenated after common.wgsl.
//
// Evaluates the same sd_character and sd_world the renderer uses, at supplied
// sample points, and writes the results out for the CPU to compare against.
//
// A fragment shader rather than a compute pass, because downlevel GL adapters
// report zero compute workgroups and would refuse one. The float is written
// as its raw bit pattern through Rgba8Unorm rather than to a float target,
// because Rgba32Float is not renderable everywhere either - and unorm8
// round-trips a byte exactly, so the comparison stays lossless.

@group(0) @binding(3) var<uniform> samples: array<vec4<f32>, 256>;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn encode_f32(v: f32) -> vec4<f32> {
    let bits = bitcast<u32>(v);
    return vec4<f32>(
        f32(bits & 0xffu) / 255.0,
        f32((bits >> 8u) & 0xffu) / 255.0,
        f32((bits >> 16u) & 0xffu) / 255.0,
        f32((bits >> 24u) & 0xffu) / 255.0);
}

// The target is 512x1: two pixels per sample, character then world.
@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = u32(frag.x);
    let p = samples[pixel >> 1u].xyz;
    if ((pixel & 1u) == 0u) {
        return encode_f32(sd_character(p));
    }
    return encode_f32(sd_world(p));
}
