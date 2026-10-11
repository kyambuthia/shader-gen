// Upscales the offscreen raymarch target to the swapchain.
//
// Sphere tracing cost is per fragment, so rendering at a fraction of the
// window and stretching the result is the cheapest real saving available -
// and on a 2011 integrated GPU it is the difference between a slideshow and
// something you can steer. Bilinear filtering hides most of the softness.

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;

struct Varying {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> Varying {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var out: Varying;
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    // Framebuffer y grows downward, clip-space y grows upward.
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return out;
}

@fragment
fn fs_main(in: Varying) -> @location(0) vec4<f32> {
    return textureSample(source, source_sampler, in.uv);
}
