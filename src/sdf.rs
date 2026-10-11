//! The character definition and its CPU evaluator.
//!
//! This module is the single source of truth for the character's shape. The
//! same `Cone` list is uploaded to the GPU and sphere-traced by
//! `shaders/character.wgsl`; the evaluator here is what collision will query.
//! The two implementations must agree, which is what `tests/parity.rs` exists
//! to enforce - see `docs/DESIGN.md`.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

/// A round cone: a segment from `a` to `b` with a radius at each end. Exact
/// distance function, tapers like a limb, and cheap to evaluate.
///
/// Packed as two `vec4`s so the GPU layout needs no padding: `xyz` is the
/// endpoint and `w` is that end's radius.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Cone {
    pub a: [f32; 4],
    pub b: [f32; 4],
}

impl Cone {
    pub fn new(a: Vec3, r_a: f32, b: Vec3, r_b: f32) -> Self {
        Self {
            a: [a.x, a.y, a.z, r_a],
            b: [b.x, b.y, b.z, r_b],
        }
    }

    pub fn head(&self) -> Vec3 {
        Vec3::new(self.a[0], self.a[1], self.a[2])
    }

    pub fn tail(&self) -> Vec3 {
        Vec3::new(self.b[0], self.b[1], self.b[2])
    }
}

/// Upper bound on cones, so the GPU buffer can be a fixed allocation.
pub const MAX_CONES: usize = 64;

pub struct CharacterDef {
    pub cones: Vec<Cone>,
    /// Smooth-minimum radius. Larger fuses joints more and costs more
    /// raymarch steps, because `smin` degrades the distance bound.
    pub blend_k: f32,
}

/// Exact distance to a round cone. Adapted from the standard derivation:
/// the segment is split into the two spherical caps and the conical band
/// between them, and the nearest of the three is returned.
fn sd_round_cone(p: Vec3, a: Vec3, b: Vec3, r1: f32, r2: f32) -> f32 {
    let ba = b - a;
    let l2 = ba.dot(ba);
    let rr = r1 - r2;
    let a2 = l2 - rr * rr;
    let il2 = 1.0 / l2;

    let pa = p - a;
    let y = pa.dot(ba);
    let z = y - l2;
    let x2 = (pa * l2 - ba * y).length_squared();
    let y2 = y * y * l2;
    let z2 = z * z * l2;

    let k = rr.signum() * rr * rr * x2;
    if z.signum() * a2 * z2 > k {
        return (x2 + z2).sqrt() * il2 - r2;
    }
    if y.signum() * a2 * y2 < k {
        return (x2 + y2).sqrt() * il2 - r1;
    }
    ((x2 * a2 * il2).sqrt() + y * rr) * il2 - r1
}

/// Polynomial smooth minimum. Fuses joints instead of leaving a crease, at
/// the cost of the field no longer being a true distance.
fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
    (b * (1.0 - h) + a * h) - k * h * (1.0 - h)
}

impl CharacterDef {
    /// Signed distance to the character. Negative inside.
    ///
    /// Must stay numerically identical to `sd_character` in
    /// `shaders/character.wgsl`.
    pub fn distance(&self, p: Vec3) -> f32 {
        let mut d = f32::MAX;
        for cone in &self.cones {
            let c = sd_round_cone(p, cone.head(), cone.tail(), cone.a[3], cone.b[3]);
            d = if d == f32::MAX {
                c
            } else {
                smin(d, c, self.blend_k)
            };
        }
        d
    }

    /// Surface normal by central differences on the field.
    pub fn normal(&self, p: Vec3) -> Vec3 {
        const E: f32 = 1.0e-3;
        Vec3::new(
            self.distance(p + Vec3::X * E) - self.distance(p - Vec3::X * E),
            self.distance(p + Vec3::Y * E) - self.distance(p - Vec3::Y * E),
            self.distance(p + Vec3::Z * E) - self.distance(p - Vec3::Z * E),
        )
        .normalize_or_zero()
    }

    /// Axis-aligned bounds, inflated by the blend radius since `smin` pushes
    /// the surface outward where limbs meet.
    pub fn bounds(&self) -> (Vec3, Vec3) {
        let mut lo = Vec3::splat(f32::MAX);
        let mut hi = Vec3::splat(f32::MIN);
        for cone in &self.cones {
            for (end, r) in [(cone.head(), cone.a[3]), (cone.tail(), cone.b[3])] {
                lo = lo.min(end - Vec3::splat(r));
                hi = hi.max(end + Vec3::splat(r));
            }
        }
        let pad = Vec3::splat(self.blend_k);
        (lo - pad, hi + pad)
    }
}

/// A 1.8 m humanoid standing at the origin, feet on y = 0, facing +Z.
///
/// Step 1 holds the pose as fixed world-space segments. The parent/offset
/// hierarchy from the design arrives with animation in step 4; posing a flat
/// list is exactly the thing that does not work, so the structure changes
/// rather than being bolted onto this.
pub fn humanoid() -> CharacterDef {
    let mut cones = vec![
        // spine, pelvis to crown
        Cone::new(Vec3::new(0.0, 0.88, 0.0), 0.145, Vec3::new(0.0, 0.98, 0.0), 0.135),
        Cone::new(Vec3::new(0.0, 0.95, 0.0), 0.130, Vec3::new(0.0, 1.10, 0.0), 0.150),
        Cone::new(Vec3::new(0.0, 1.10, 0.0), 0.165, Vec3::new(0.0, 1.45, 0.0), 0.145),
        Cone::new(Vec3::new(0.0, 1.45, 0.0), 0.055, Vec3::new(0.0, 1.58, 0.0), 0.060),
        Cone::new(Vec3::new(0.0, 1.58, 0.0), 0.105, Vec3::new(0.0, 1.74, 0.0), 0.095),
    ];

    // Limbs, built on the +x side and mirrored.
    let half: Vec<Cone> = vec![
        // The arm hangs clear of the chest: at r = 0.165 the torso surface
        // reaches x = 0.165, so an upper arm any further inboard than this is
        // swallowed by it once `blend_k` is applied.
        Cone::new(Vec3::new(0.02, 1.43, 0.0), 0.085, Vec3::new(0.21, 1.40, 0.0), 0.068),
        Cone::new(Vec3::new(0.21, 1.40, 0.0), 0.068, Vec3::new(0.26, 1.12, 0.0), 0.052),
        Cone::new(Vec3::new(0.26, 1.12, 0.0), 0.052, Vec3::new(0.29, 0.86, 0.0), 0.040),
        Cone::new(Vec3::new(0.29, 0.86, 0.0), 0.043, Vec3::new(0.30, 0.76, 0.0), 0.033),
        Cone::new(Vec3::new(0.09, 0.90, 0.0), 0.100, Vec3::new(0.10, 0.50, 0.0), 0.075),
        Cone::new(Vec3::new(0.10, 0.50, 0.0), 0.072, Vec3::new(0.10, 0.08, 0.0), 0.050),
        Cone::new(Vec3::new(0.10, 0.05, -0.02), 0.055, Vec3::new(0.10, 0.04, 0.14), 0.045),
    ];
    for cone in &half {
        cones.push(*cone);
        let (mut a, mut b) = (cone.a, cone.b);
        a[0] = -a[0];
        b[0] = -b[0];
        cones.push(Cone { a, b });
    }

    assert!(cones.len() <= MAX_CONES, "character exceeds MAX_CONES");
    CharacterDef {
        cones,
        blend_k: 0.035,
    }
}
