//! The world the character walks on: a ground plane and a few rounded boxes.
//!
//! Kept deliberately small. Every primitive here has to exist twice - once in
//! Rust for collision, once in WGSL for rendering - so the cost of each one is
//! paid in both places and in the parity test that keeps them honest.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

pub const MAX_BOXES: usize = 16;

/// `center.xyz` and `half.xyz` are the box; `half.w` is the corner radius,
/// already subtracted from the half-extents so the box keeps its nominal size.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct RoundBox {
    pub center: [f32; 4],
    pub half: [f32; 4],
}

impl RoundBox {
    pub fn new(center: Vec3, half: Vec3, radius: f32) -> Self {
        let h = (half - Vec3::splat(radius)).max(Vec3::splat(0.001));
        Self {
            center: [center.x, center.y, center.z, 0.0],
            half: [h.x, h.y, h.z, radius],
        }
    }
}

pub struct World {
    pub boxes: Vec<RoundBox>,
}

/// Exact distance to an axis-aligned rounded box.
pub fn sd_round_box(p: Vec3, center: Vec3, half: Vec3, radius: f32) -> f32 {
    let q = (p - center).abs() - half;
    q.max(Vec3::ZERO).length() + q.max_element().min(0.0) - radius
}

impl World {
    /// A few platforms and steps, enough to prove ground contact, walking up
    /// onto something, and being pushed out of a wall.
    pub fn arena() -> Self {
        let mut boxes = vec![
            RoundBox::new(Vec3::new(0.0, 0.15, 4.0), Vec3::new(1.6, 0.15, 1.2), 0.06),
            RoundBox::new(Vec3::new(0.0, 0.45, 6.0), Vec3::new(1.6, 0.15, 1.2), 0.06),
            RoundBox::new(Vec3::new(0.0, 0.75, 8.0), Vec3::new(1.6, 0.15, 1.2), 0.06),
            RoundBox::new(Vec3::new(-4.5, 0.60, 2.0), Vec3::new(1.2, 0.60, 1.2), 0.10),
            RoundBox::new(Vec3::new(4.5, 1.00, 3.0), Vec3::new(1.0, 1.00, 1.0), 0.10),
            // A wall to walk into.
            RoundBox::new(Vec3::new(2.5, 0.9, -3.0), Vec3::new(2.5, 0.9, 0.25), 0.05),
        ];
        boxes.resize(MAX_BOXES, RoundBox::default());
        Self { boxes }
    }

    /// Number of live boxes; the tail of the buffer is zeroed padding.
    pub fn count(&self) -> u32 {
        self.boxes
            .iter()
            .take_while(|b| b.half[0] > 0.0 || b.half[1] > 0.0)
            .count() as u32
    }

    /// Signed distance to the world. Must stay numerically identical to
    /// `sd_world` in `shaders/common.wgsl`.
    pub fn distance(&self, p: Vec3) -> f32 {
        let mut d = p.y; // ground plane
        for b in self.boxes.iter().take(self.count() as usize) {
            let center = Vec3::new(b.center[0], b.center[1], b.center[2]);
            let half = Vec3::new(b.half[0], b.half[1], b.half[2]);
            d = d.min(sd_round_box(p, center, half, b.half[3]));
        }
        d
    }

    pub fn normal(&self, p: Vec3) -> Vec3 {
        const E: f32 = 1.0e-3;
        Vec3::new(
            self.distance(p + Vec3::X * E) - self.distance(p - Vec3::X * E),
            self.distance(p + Vec3::Y * E) - self.distance(p - Vec3::Y * E),
            self.distance(p + Vec3::Z * E) - self.distance(p - Vec3::Z * E),
        )
        .normalize_or_zero()
    }
}

impl World {
    /// Collective bounds of the boxes, used to clip the renderer's march.
    /// The ground plane is excluded: it is infinite, and the renderer
    /// intersects it analytically rather than marching it.
    pub fn box_bounds(&self) -> (Vec3, Vec3) {
        let mut lo = Vec3::splat(f32::MAX);
        let mut hi = Vec3::splat(f32::MIN);
        for b in self.boxes.iter().take(self.count() as usize) {
            let center = Vec3::new(b.center[0], b.center[1], b.center[2]);
            let half = Vec3::new(b.half[0], b.half[1], b.half[2]) + Vec3::splat(b.half[3]);
            lo = lo.min(center - half);
            hi = hi.max(center + half);
        }
        if lo.x > hi.x {
            return (Vec3::ZERO, Vec3::ZERO);
        }
        (lo, hi)
    }
}
