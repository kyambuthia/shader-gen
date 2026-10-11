//! Playable SDF character. See docs/DESIGN.md.
//!
//!   WASD      move          Shift  run
//!   Arrows    turn / pitch  Space  jump
//!   N         next body     Esc    quit
//!
//! `--headless <path.png> [--seed N] [--settle N]` renders one frame after N
//! simulation steps and exits, which is how this is checked without a display.

use std::sync::Arc;

use glam::Vec3;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey, SmolStr};
use winit::window::{Window, WindowId};

use shader_gen::gpu::{adapter_and_device, Globals, Renderer, Scene};
use shader_gen::player::{Input, Player};
use shader_gen::{body, sdf, world::World};

/// Everything that is not wgpu: the world, the body, and where it is.
struct Sim {
    world: World,
    skeleton: sdf::Skeleton,
    player: Player,
    cones: Vec<sdf::Cone>,
    seed: u64,
    camera_yaw: f32,
    camera_pitch: f32,
    eye: Vec3,
}

impl Sim {
    fn new(seed: u64) -> Self {
        let world = World::arena();
        let skeleton = body::from_seed(seed);
        let player = Player::new(&skeleton, Vec3::new(0.0, 1.5, 0.0));
        Self {
            world,
            skeleton,
            player,
            cones: Vec::new(),
            seed,
            camera_yaw: 0.0,
            camera_pitch: 0.22,
            eye: Vec3::new(0.0, 2.0, -4.0),
        }
    }

    fn respawn(&mut self, seed: u64) {
        let position = self.player.position + Vec3::Y * 0.5;
        self.skeleton = body::from_seed(seed);
        let yaw = self.player.yaw;
        self.player = Player::new(&self.skeleton, position);
        self.player.yaw = yaw;
        self.seed = seed;
    }

    fn step(&mut self, input: Input, dt: f32) {
        self.player.update(input, &self.world, dt);
        let pose = self.player.pose(&self.skeleton);
        self.skeleton.resolve(&pose, &mut self.cones);

        // Chase camera, eased so it trails rather than snapping.
        self.camera_yaw = self.player.yaw;
        let height = self.player.height;
        let back = Vec3::new(self.camera_yaw.sin(), 0.0, self.camera_yaw.cos());
        let focus = self.player.position + Vec3::Y * height * 0.62;
        let desired = focus - back * (height * 1.9) + Vec3::Y * (height * self.camera_pitch);
        let lerp = (dt * 9.0).min(1.0);
        self.eye += (desired - self.eye) * lerp;
        // Never let the camera sink below the surface it is looking over.
        let floor = self.world.distance(self.eye);
        if floor < 0.25 {
            self.eye += self.world.normal(self.eye) * (0.25 - floor);
        }
    }

    fn focus(&self) -> Vec3 {
        self.player.position + Vec3::Y * self.player.height * 0.58
    }

    fn globals(&self, width: u32, height: u32, time: f32) -> Globals {
        let (lo, hi) = sdf::bounds(&self.cones, self.skeleton.blend_k);
        let eye = self.eye;
        let focus = self.focus();
        Globals {
            resolution: [width as f32, height as f32],
            time,
            blend_k: self.skeleton.blend_k,
            cone_count: self.cones.len() as u32,
            box_count: self.world.count(),
            eye: [eye.x, eye.y, eye.z, 0.0],
            look_at: [focus.x, focus.y, focus.z, 0.0],
            bounds_lo: [lo.x, lo.y, lo.z, 0.0],
            bounds_hi: [hi.x, hi.y, hi.z, 0.0],
            ..Default::default()
        }
    }
}

#[derive(Default)]
struct Keys {
    forward: bool,
    back: bool,
    left: bool,
    right: bool,
    turn_left: bool,
    turn_right: bool,
    pitch_up: bool,
    pitch_down: bool,
    jump: bool,
    run: bool,
}

impl Keys {
    fn input(&self) -> Input {
        Input {
            forward: f32::from(self.forward) - f32::from(self.back),
            strafe: f32::from(self.right) - f32::from(self.left),
            turn: f32::from(self.turn_right) - f32::from(self.turn_left),
            jump: self.jump,
            run: self.run,
        }
    }
}

fn headless(path: &str, seed: u64, settle: u32) {
    const WIDTH: u32 = 1024;
    const HEIGHT: u32 = 640;

    let instance = wgpu::Instance::default();
    let (_adapter, device, queue) = pollster::block_on(adapter_and_device(&instance, None));

    let mut sim = Sim::new(seed);
    // Settle under gravity, then walk forward so the pose is mid-stride
    // rather than a rest T-pose.
    for i in 0..settle {
        let input = if i > settle / 2 {
            Input { forward: 1.0, ..Default::default() }
        } else {
            Input::default()
        };
        sim.step(input, 1.0 / 120.0);
    }
    // Snap the camera, which would otherwise still be easing toward the body.
    for _ in 0..30 {
        sim.step(Input { forward: 1.0, ..Default::default() }, 1.0 / 120.0);
    }

    let format = wgpu::TextureFormat::Rgba8Unorm;
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("headless target"),
        size: wgpu::Extent3d { width: WIDTH, height: HEIGHT, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());

    let scene = Scene::new(&device, &queue, &sim.world);
    scene.upload_cones(&queue, &sim.cones);
    queue.write_buffer(
        &scene.globals,
        0,
        bytemuck::bytes_of(&sim.globals(WIDTH, HEIGHT, 0.0)),
    );
    let renderer = Renderer::new(&device, &scene, format);

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    renderer.draw(&mut encoder, &view);

    let unpadded = WIDTH * 4;
    let padded = unpadded.div_ceil(256) * 256;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (padded * HEIGHT) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(HEIGHT),
            },
        },
        wgpu::Extent3d { width: WIDTH, height: HEIGHT, depth_or_array_layers: 1 },
    );
    queue.submit([encoder.finish()]);

    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
    device
        .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
        .unwrap();
    let data = slice.get_mapped_range().expect("map range");
    let mut rgba = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);
    for row in 0..HEIGHT {
        let start = (row * padded) as usize;
        rgba.extend_from_slice(&data[start..start + unpadded as usize]);
    }
    drop(data);
    staging.unmap();

    write_png(path, WIDTH, HEIGHT, &rgba);
    let speed = Vec3::new(sim.player.velocity.x, 0.0, sim.player.velocity.z).length();
    let thigh_l = sim.cones[sdf::joint::THIGH_L];
    let thigh_r = sim.cones[sdf::joint::THIGH_R];
    println!(
        "wrote {path} ({WIDTH}x{HEIGHT}) seed {seed}, {} cones\n\
           standing {}  pos {:.2?}  speed {speed:.2}  stride {:.2}\n\
           thigh z-splay  L {:+.3}  R {:+.3}",
        sim.cones.len(),
        sim.player.grounded,
        sim.player.position,
        sim.player.stride,
        thigh_l.tail().z - thigh_l.head().z,
        thigh_r.tail().z - thigh_r.head().z,
    );
}

/// Minimal PNG writer: one stored deflate block, so debug output costs no
/// image or compression dependency.
fn write_png(path: &str, width: u32, height: u32, rgba: &[u8]) {
    fn crc32(data: &[u8]) -> u32 {
        let mut crc = 0xffff_ffffu32;
        for &byte in data {
            crc ^= byte as u32;
            for _ in 0..8 {
                crc = if crc & 1 != 0 { (crc >> 1) ^ 0xedb8_8320 } else { crc >> 1 };
            }
        }
        !crc
    }
    fn chunk(out: &mut Vec<u8>, tag: &[u8; 4], body: &[u8]) {
        out.extend_from_slice(&(body.len() as u32).to_be_bytes());
        let mut tagged = tag.to_vec();
        tagged.extend_from_slice(body);
        out.extend_from_slice(&tagged);
        out.extend_from_slice(&crc32(&tagged).to_be_bytes());
    }

    let mut raw = Vec::with_capacity(((width * 4 + 1) * height) as usize);
    for row in 0..height {
        raw.push(0); // filter: none
        let start = (row * width * 4) as usize;
        raw.extend_from_slice(&rgba[start..start + (width * 4) as usize]);
    }

    let mut zlib = vec![0x78, 0x01];
    for (i, block) in raw.chunks(65535).enumerate() {
        zlib.push(u8::from((i + 1) * 65535 >= raw.len()));
        zlib.extend_from_slice(&(block.len() as u16).to_le_bytes());
        zlib.extend_from_slice(&(!(block.len() as u16)).to_le_bytes());
        zlib.extend_from_slice(block);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in &raw {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    zlib.extend_from_slice(&((b << 16) | a).to_be_bytes());

    let mut png = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]); // 8-bit RGBA
    chunk(&mut png, b"IHDR", &ihdr);
    chunk(&mut png, b"IDAT", &zlib);
    chunk(&mut png, b"IEND", &[]);
    std::fs::write(path, png).expect("write png");
}

struct App {
    window: Option<Arc<Window>>,
    surface: Option<wgpu::Surface<'static>>,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    scene: Option<Scene>,
    renderer: Option<Renderer>,
    config: Option<wgpu::SurfaceConfiguration>,
    sim: Sim,
    keys: Keys,
    last: std::time::Instant,
    start: std::time::Instant,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("shader-gen")
                        .with_inner_size(winit::dpi::LogicalSize::new(1024, 640)),
                )
                .expect("create window"),
        );

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(window.clone()).expect("create surface");
        let (adapter, device, queue) =
            pollster::block_on(adapter_and_device(&instance, Some(&surface)));

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: Default::default(),
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let scene = Scene::new(&device, &queue, &self.sim.world);
        let renderer = Renderer::new(&device, &scene, format);

        self.scene = Some(scene);
        self.renderer = Some(renderer);
        self.device = Some(device);
        self.queue = Some(queue);
        self.surface = Some(surface);
        self.config = Some(config);
        self.window = Some(window);
        self.last = std::time::Instant::now();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
                return;
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let down = event.state.is_pressed();
                match event.logical_key.as_ref() {
                    Key::Named(NamedKey::Escape) => event_loop.exit(),
                    Key::Named(NamedKey::Space) => self.keys.jump = down,
                    Key::Named(NamedKey::Shift) => self.keys.run = down,
                    Key::Named(NamedKey::ArrowLeft) => self.keys.turn_left = down,
                    Key::Named(NamedKey::ArrowRight) => self.keys.turn_right = down,
                    Key::Named(NamedKey::ArrowUp) => self.keys.pitch_up = down,
                    Key::Named(NamedKey::ArrowDown) => self.keys.pitch_down = down,
                    Key::Character(c) => match c {
                        "w" | "W" => self.keys.forward = down,
                        "s" | "S" => self.keys.back = down,
                        "a" | "A" => self.keys.left = down,
                        "d" | "D" => self.keys.right = down,
                        "n" | "N" => {
                            if down {
                                let seed = self.sim.seed.wrapping_add(1);
                                self.sim.respawn(seed);
                            }
                        }
                        _ => {}
                    },
                    _ => {}
                }
                let _ = SmolStr::new_inline("");
                return;
            }
            WindowEvent::Resized(size) => {
                if let (Some(config), Some(device), Some(surface)) =
                    (self.config.as_mut(), self.device.as_ref(), self.surface.as_ref())
                {
                    config.width = size.width.max(1);
                    config.height = size.height.max(1);
                    surface.configure(device, config);
                }
                return;
            }
            WindowEvent::RedrawRequested => {}
            _ => return,
        }

        let (Some(window), Some(surface), Some(device), Some(queue), Some(scene), Some(renderer), Some(config)) = (
            self.window.as_ref(),
            self.surface.as_ref(),
            self.device.as_ref(),
            self.queue.as_ref(),
            self.scene.as_ref(),
            self.renderer.as_ref(),
            self.config.as_ref(),
        ) else {
            return;
        };

        // Fixed step, so collision behaves the same at any frame rate.
        let now = std::time::Instant::now();
        let elapsed = (now - self.last).as_secs_f32().min(0.1);
        self.last = now;
        const STEP: f32 = 1.0 / 120.0;
        let mut remaining = elapsed;
        let input = self.keys.input();
        self.sim.camera_pitch =
            (self.sim.camera_pitch + (f32::from(self.keys.pitch_up) - f32::from(self.keys.pitch_down)) * elapsed)
                .clamp(-0.1, 1.4);
        while remaining > 0.0 {
            self.sim.step(input, STEP.min(remaining));
            remaining -= STEP;
        }

        scene.upload_cones(queue, &self.sim.cones);
        queue.write_buffer(
            &scene.globals,
            0,
            bytemuck::bytes_of(&self.sim.globals(
                config.width,
                config.height,
                self.start.elapsed().as_secs_f32(),
            )),
        );

        use wgpu::CurrentSurfaceTexture as Current;
        // Suboptimal still presents; the other states have no texture to draw
        // into, so skip the frame and let the next one retry.
        let frame = match surface.get_current_texture() {
            Current::Success(frame) | Current::Suboptimal(frame) => frame,
            _ => {
                window.request_redraw();
                return;
            }
        };
        let view = frame.texture.create_view(&Default::default());
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        renderer.draw(&mut encoder, &view);
        queue.submit([encoder.finish()]);
        drop(frame); // wgpu 30 presents on drop; there is no present()
        window.request_redraw();
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let value = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse::<u64>().ok())
    };
    let seed = value("--seed").unwrap_or(0);

    if let Some(index) = args.iter().position(|a| a == "--headless") {
        let path = args.get(index + 1).map(String::as_str).unwrap_or("frame.png");
        headless(path, seed, value("--settle").unwrap_or(240) as u32);
        return;
    }

    let event_loop = EventLoop::new().expect("create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop
        .run_app(&mut App {
            window: None,
            surface: None,
            device: None,
            queue: None,
            scene: None,
            renderer: None,
            config: None,
            sim: Sim::new(seed),
            keys: Keys::default(),
            last: std::time::Instant::now(),
            start: std::time::Instant::now(),
        })
        .expect("run");
}
