//! Phase 2: the CPU and GPU evaluators must agree.
//!
//! The SDF is the single source of truth for both the surface and the
//! collider, but it is evaluated twice - once in WGSL to draw it, once in
//! Rust to collide with it. Two implementations of one definition drift, and
//! the symptom is a physics bug nobody connects to a rendering change weeks
//! later. This turns that drift into a failing build.

use glam::Vec3;
use shader_gen::gpu::{adapter_and_device, ParityProbe, Scene, PARITY_SAMPLES};
use shader_gen::{body, sdf, world::World};

/// Deterministic sample points, so a failure is reproducible.
fn sample_points(lo: Vec3, hi: Vec3, count: usize) -> Vec<Vec3> {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u32 << 24) as f32
    };
    // Inflated past the bounds so the far field is covered too, not just the
    // surface: the renderer steps through empty space and must agree there.
    let pad = (hi - lo) * 0.6;
    let (lo, hi) = (lo - pad, hi + pad);
    (0..count)
        .map(|_| {
            Vec3::new(
                lo.x + (hi.x - lo.x) * next(),
                lo.y + (hi.y - lo.y) * next(),
                lo.z + (hi.z - lo.z) * next(),
            )
        })
        .collect()
}

#[test]
fn cpu_and_gpu_evaluators_agree() {
    let instance = wgpu::Instance::default();
    let (_adapter, device, queue) = pollster::block_on(adapter_and_device(&instance, None));

    let world = World::arena();
    let scene = Scene::new(&device, &queue, &world);
    let probe = ParityProbe::new(&device, &scene);

    // Several seeds, since generation varies proportions and a bug in the
    // bone maths may only show up at some of them.
    for seed in [0u64, 1, 7, 42, 1337] {
        let skeleton = body::from_seed(seed);
        let mut pose = sdf::Pose::rest(&skeleton);
        pose.root = Vec3::new(0.3, 0.0, -0.7);
        pose.yaw = 0.9;

        let mut cones = Vec::new();
        skeleton.resolve(&pose, &mut cones);
        scene.upload_cones(&queue, &cones);

        let (lo, hi) = sdf::bounds(&cones, skeleton.blend_k);
        queue.write_buffer(
            &scene.globals,
            0,
            bytemuck::bytes_of(&shader_gen::gpu::Globals {
                resolution: [1.0, 1.0],
                time: 0.0,
                blend_k: skeleton.blend_k,
                cone_count: cones.len() as u32,
                box_count: world.count(),
                bounds_lo: [lo.x, lo.y, lo.z, 0.0],
                bounds_hi: [hi.x, hi.y, hi.z, 0.0],
                ..Default::default()
            }),
        );

        let points = sample_points(lo, hi, PARITY_SAMPLES);
        let gpu = probe.evaluate(&device, &queue, &points);

        let mut worst_character = 0.0f32;
        let mut worst_world = 0.0f32;
        let mut worst_point = Vec3::ZERO;
        for (point, (gpu_character, gpu_world)) in points.iter().zip(&gpu) {
            let cpu_character = sdf::distance(&cones, skeleton.blend_k, *point);
            let cpu_world = world.distance(*point);
            let dc = (cpu_character - gpu_character).abs();
            let dw = (cpu_world - gpu_world).abs();
            if dc > worst_character {
                worst_character = dc;
                worst_point = *point;
            }
            worst_world = worst_world.max(dw);
        }

        // f32 across two compilers and two instruction selections will not be
        // bit-identical. A millimetre on a 1.8 m character is far tighter than
        // anything that could hide a real divergence, which shows up as
        // centimetres or worse.
        const TOLERANCE: f32 = 1.0e-3;
        assert!(
            worst_character < TOLERANCE,
            "seed {seed}: character evaluators diverge by {worst_character} at {worst_point}"
        );
        assert!(
            worst_world < TOLERANCE,
            "seed {seed}: world evaluators diverge by {worst_world}"
        );
    }
}

/// The collider must be derived from the character, not authored beside it.
#[test]
fn capsule_tracks_the_generated_body() {
    for seed in [0u64, 3, 9, 100] {
        let skeleton = body::from_seed(seed);
        let player = shader_gen::player::Player::new(&skeleton, Vec3::ZERO);

        let mut cones = Vec::new();
        skeleton.resolve(&sdf::Pose::rest(&skeleton), &mut cones);
        let (lo, hi) = sdf::bounds(&cones, skeleton.blend_k);

        assert!(
            (player.height - (hi.y - lo.y)).abs() < 1.0e-4,
            "seed {seed}: capsule height {} does not match body {}",
            player.height,
            hi.y - lo.y
        );
        assert!(
            player.radius > 0.0 && player.radius < player.height * 0.5,
            "seed {seed}: capsule radius {} is not usable",
            player.radius
        );
    }
}

/// Gravity, ground contact and push-out, with no renderer involved.
#[test]
fn character_falls_and_lands() {
    let world = World::arena();
    let skeleton = body::from_seed(0);
    let mut player = shader_gen::player::Player::new(&skeleton, Vec3::new(0.0, 3.0, 0.0));

    for _ in 0..600 {
        player.update(Default::default(), &world, 1.0 / 120.0);
    }

    assert!(player.grounded, "character never landed");
    assert!(
        player.position.y.abs() < 0.05,
        "came to rest at y = {} instead of the ground",
        player.position.y
    );

    // Walk into the wall at x in [0, 5], z = -3, and stop at its face rather
    // than passing through it.
    let mut player = shader_gen::player::Player::new(&skeleton, Vec3::new(2.5, 0.0, 0.0));
    player.yaw = std::f32::consts::PI; // face -z
    let input = shader_gen::player::Input { forward: 1.0, run: true, ..Default::default() };
    for _ in 0..480 {
        player.update(input, &world, 1.0 / 120.0);
    }
    assert!(
        player.position.z > -2.9,
        "walked through the wall to z = {}",
        player.position.z
    );
}
