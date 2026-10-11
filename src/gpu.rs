//! wgpu plumbing: the shared uniforms, the renderer, and the parity probe.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use crate::sdf::{Cone, MAX_CONES};
use crate::world::{RoundBox, World, MAX_BOXES};

/// Mirrors `Globals` in shaders/common.wgsl. 96 bytes; the explicit scalar
/// padding keeps Rust and WGSL agreeing, which a `vec3` would not since it
/// would force the next field to a 16-byte boundary.
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
pub struct Globals {
    pub resolution: [f32; 2],
    pub time: f32,
    pub blend_k: f32,
    pub cone_count: u32,
    pub box_count: u32,
    pub _pad0: u32,
    pub _pad1: u32,
    pub eye: [f32; 4],
    pub look_at: [f32; 4],
    pub bounds_lo: [f32; 4],
    pub bounds_hi: [f32; 4],
}

const COMMON: &str = include_str!("../shaders/common.wgsl");
const CHARACTER: &str = include_str!("../shaders/character.wgsl");
const PARITY: &str = include_str!("../shaders/parity.wgsl");

pub const PARITY_SAMPLES: usize = 256;

pub async fn adapter_and_device(
    instance: &wgpu::Instance,
    surface: Option<&wgpu::Surface<'static>>,
) -> (wgpu::Adapter, wgpu::Device, wgpu::Queue) {
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: surface,
            apply_limit_buckets: false,
        })
        .await
        .expect("no suitable GPU adapter");
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            // Take whatever the adapter offers. Sphere tracing is entirely a
            // fragment-stage job, so demanding `downlevel_defaults` only
            // rejects GL adapters over compute limits we never use.
            required_limits: adapter.limits(),
            experimental_features: Default::default(),
            memory_hints: Default::default(),
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("no suitable GPU device");
    (adapter, device, queue)
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_buffer(device: &wgpu::Device, label: &str, size: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// The scene buffers shared by every pipeline: globals, the resolved cone
/// list, and the world's boxes.
pub struct Scene {
    pub globals: wgpu::Buffer,
    pub cones: wgpu::Buffer,
    pub boxes: wgpu::Buffer,
}

impl Scene {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, world: &World) -> Self {
        // Uniform arrays rather than storage buffers: downlevel GL adapters
        // allow zero storage buffers in the fragment stage, and the whole
        // scene is a few kilobytes.
        let scene = Self {
            globals: uniform_buffer(device, "globals", std::mem::size_of::<Globals>()),
            cones: uniform_buffer(device, "cones", std::mem::size_of::<Cone>() * MAX_CONES),
            boxes: uniform_buffer(device, "boxes", std::mem::size_of::<RoundBox>() * MAX_BOXES),
        };
        queue.write_buffer(&scene.boxes, 0, bytemuck::cast_slice(&world.boxes));
        scene
    }

    pub fn upload_cones(&self, queue: &wgpu::Queue, cones: &[Cone]) {
        let mut padded = [Cone::default(); MAX_CONES];
        padded[..cones.len()].copy_from_slice(cones);
        queue.write_buffer(&self.cones, 0, bytemuck::cast_slice(&padded));
    }
}

pub struct Renderer {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
}

impl Renderer {
    pub fn new(device: &wgpu::Device, scene: &Scene, format: wgpu::TextureFormat) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scene"),
            entries: &[uniform_entry(0), uniform_entry(1), uniform_entry(2)],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: scene.globals.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: scene.cones.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: scene.boxes.as_entire_binding() },
            ],
        });
        let pipeline = build_pipeline(device, &layout, &format!("{COMMON}\n{CHARACTER}"), format);
        Self { pipeline, bind_group }
    }

    pub fn draw(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("character"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn build_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    source: &str,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sdf"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("sdf"),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("sdf"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(format.into())],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// Evaluates the shader's distance functions at supplied points and reads the
/// results back, so the CPU evaluator can be checked against the real thing
/// rather than against a second transcription of it.
pub struct ParityProbe {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    samples: wgpu::Buffer,
    target: wgpu::Texture,
    readback: wgpu::Buffer,
}

// Rgba8Unorm, not a float format: Rgba32Float is not renderable on every
// adapter, and a unorm8 byte round-trips exactly, so writing the float bit
// pattern through it loses nothing.
const PARITY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// Two pixels per sample (character, world), four bytes each.
const PARITY_PIXELS: u32 = PARITY_SAMPLES as u32 * 2;
const PARITY_ROW: u32 = PARITY_PIXELS * 4; // 2048, already 256-aligned

impl ParityProbe {
    pub fn new(device: &wgpu::Device, scene: &Scene) -> Self {
        let samples = uniform_buffer(device, "samples", PARITY_SAMPLES * 16);
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("parity"),
            entries: &[uniform_entry(0), uniform_entry(1), uniform_entry(2), uniform_entry(3)],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("parity"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: scene.globals.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: scene.cones.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: scene.boxes.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: samples.as_entire_binding() },
            ],
        });
        let pipeline =
            build_pipeline(device, &layout, &format!("{COMMON}\n{PARITY}"), PARITY_FORMAT);

        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("parity target"),
            size: wgpu::Extent3d {
                width: PARITY_PIXELS,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: PARITY_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("parity readback"),
            size: PARITY_ROW as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self { pipeline, bind_group, samples, target, readback }
    }

    /// Returns `(character_distance, world_distance)` per point, as the GPU
    /// computed them.
    pub fn evaluate(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        points: &[Vec3],
    ) -> Vec<(f32, f32)> {
        assert!(points.len() <= PARITY_SAMPLES);
        let mut packed = [[0.0f32; 4]; PARITY_SAMPLES];
        for (slot, p) in packed.iter_mut().zip(points) {
            *slot = [p.x, p.y, p.z, 0.0];
        }
        queue.write_buffer(&self.samples, 0, bytemuck::cast_slice(&packed));

        let view = self.target.create_view(&Default::default());
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("parity"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(PARITY_ROW),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: PARITY_PIXELS,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);

        let slice = self.readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
        device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .unwrap();
        let data = slice.get_mapped_range().expect("map range");
        let decode = |pixel: usize| {
            let b = &data[pixel * 4..pixel * 4 + 4];
            f32::from_bits(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        };
        let out = (0..points.len())
            .map(|i| (decode(i * 2), decode(i * 2 + 1)))
            .collect();
        drop(data);
        self.readback.unmap();
        out
    }
}
