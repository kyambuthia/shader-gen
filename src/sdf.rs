//! The character: a bone hierarchy, its resolved cones, and the CPU evaluator.
//!
//! This module is the single source of truth for the character's shape. Posing
//! happens here, on the CPU, and resolves the skeleton into a flat list of
//! world-space cones. That list is what the GPU receives and what collision
//! queries, so the shader never needs to know about bones and the two sides
//! cannot disagree about where the surface is. `tests/parity.rs` enforces it.

use bytemuck::{Pod, Zeroable};
use glam::{Quat, Vec3};

/// A round cone: a segment with a radius at each end. Exact distance function,
/// tapers like a limb, cheap to evaluate.
///
/// Packed as two `vec4`s so the GPU layout needs no padding: `xyz` is the
/// endpoint, `w` is that end's radius.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
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

/// Upper bound on cones, so the GPU buffer is a fixed allocation.
pub const MAX_CONES: usize = 64;

/// How a bone is authored: head and tail in model space, which is far easier
/// to reason about than a chain of local rotations. `Skeleton::new` converts
/// to the parent-relative form the pose maths needs.
pub struct BoneSpec {
    pub parent: i32,
    pub head: Vec3,
    pub tail: Vec3,
    pub r_head: f32,
    pub r_tail: f32,
}

pub struct Bone {
    pub parent: i32,
    /// Offset from the parent's head, in the parent's rest frame.
    pub offset: Vec3,
    /// Rest rotation relative to the parent's rest frame.
    pub rest: Quat,
    pub length: f32,
    pub r_head: f32,
    pub r_tail: f32,
}

pub struct Skeleton {
    pub bones: Vec<Bone>,
    /// Smooth-minimum radius. Larger fuses joints more and costs more raymarch
    /// steps, because `smin` degrades the distance bound.
    pub blend_k: f32,
}

/// Bone indices into the humanoid skeleton, for the walk cycle to drive.
pub mod joint {
    pub const PELVIS: usize = 0;
    pub const SPINE: usize = 1;
    pub const CHEST: usize = 2;
    pub const NECK: usize = 3;
    pub const HEAD: usize = 4;
    pub const UPPER_ARM_L: usize = 6;
    pub const FOREARM_L: usize = 7;
    pub const UPPER_ARM_R: usize = 10;
    pub const FOREARM_R: usize = 11;
    pub const THIGH_L: usize = 13;
    pub const SHIN_L: usize = 14;
    pub const FOOT_L: usize = 15;
    pub const THIGH_R: usize = 16;
    pub const SHIN_R: usize = 17;
    pub const FOOT_R: usize = 18;
}

/// Per-bone animation rotation, applied on top of the rest pose.
#[derive(Clone)]
pub struct Pose {
    pub root: Vec3,
    pub yaw: f32,
    pub local: Vec<Quat>,
}

impl Pose {
    pub fn rest(skeleton: &Skeleton) -> Self {
        Self {
            root: Vec3::ZERO,
            yaw: 0.0,
            local: vec![Quat::IDENTITY; skeleton.bones.len()],
        }
    }
}

fn rotation_to(dir: Vec3) -> Quat {
    Quat::from_rotation_arc(Vec3::Y, dir.normalize_or_zero())
}

impl Skeleton {
    pub fn new(specs: &[BoneSpec], blend_k: f32) -> Self {
        // Rest orientation of each bone in model space, then converted to the
        // parent-relative pair (offset, rest) that posing composes.
        let mut world_rest: Vec<Quat> = Vec::with_capacity(specs.len());
        let mut bones = Vec::with_capacity(specs.len());

        for spec in specs {
            let axis = spec.tail - spec.head;
            let rotation = rotation_to(axis);
            let (offset, rest) = if spec.parent < 0 {
                (spec.head, rotation)
            } else {
                let parent = &specs[spec.parent as usize];
                let inverse = world_rest[spec.parent as usize].inverse();
                (inverse * (spec.head - parent.head), inverse * rotation)
            };
            world_rest.push(rotation);
            bones.push(Bone {
                parent: spec.parent,
                offset,
                rest,
                length: axis.length(),
                r_head: spec.r_head,
                r_tail: spec.r_tail,
            });
        }

        assert!(bones.len() <= MAX_CONES, "skeleton exceeds MAX_CONES");
        Self { bones, blend_k }
    }

    /// Walk the hierarchy and write one world-space cone per bone.
    pub fn resolve(&self, pose: &Pose, out: &mut Vec<Cone>) {
        out.clear();
        let yaw = Quat::from_rotation_y(pose.yaw);
        let mut heads: Vec<Vec3> = Vec::with_capacity(self.bones.len());
        let mut rotations: Vec<Quat> = Vec::with_capacity(self.bones.len());

        for (index, bone) in self.bones.iter().enumerate() {
            let (parent_head, parent_rotation) = if bone.parent < 0 {
                (pose.root, yaw)
            } else {
                let p = bone.parent as usize;
                (heads[p], rotations[p])
            };

            let head = parent_head + parent_rotation * bone.offset;
            let rotation = parent_rotation * bone.rest * pose.local[index];
            let tail = head + rotation * (Vec3::Y * bone.length);

            heads.push(head);
            rotations.push(rotation);
            out.push(Cone::new(head, bone.r_head, tail, bone.r_tail));
        }
    }
}

/// Exact distance to a round cone. The segment splits into two spherical caps
/// and the conical band between them; the nearest of the three wins.
pub fn sd_round_cone(p: Vec3, a: Vec3, b: Vec3, r1: f32, r2: f32) -> f32 {
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

/// Polynomial smooth minimum. Fuses joints instead of leaving a crease, at the
/// cost of the field no longer being a true distance.
pub fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
    (b * (1.0 - h) + a * h) - k * h * (1.0 - h)
}

/// Signed distance to a resolved cone list. Negative inside.
///
/// Must stay numerically identical to `sd_character` in
/// `shaders/common.wgsl`, which `tests/parity.rs` checks.
pub fn distance(cones: &[Cone], blend_k: f32, p: Vec3) -> f32 {
    let mut d = 0.0;
    for (index, cone) in cones.iter().enumerate() {
        let c = sd_round_cone(p, cone.head(), cone.tail(), cone.a[3], cone.b[3]);
        d = if index == 0 { c } else { smin(d, c, blend_k) };
    }
    d
}

pub fn normal(cones: &[Cone], blend_k: f32, p: Vec3) -> Vec3 {
    const E: f32 = 1.0e-3;
    Vec3::new(
        distance(cones, blend_k, p + Vec3::X * E) - distance(cones, blend_k, p - Vec3::X * E),
        distance(cones, blend_k, p + Vec3::Y * E) - distance(cones, blend_k, p - Vec3::Y * E),
        distance(cones, blend_k, p + Vec3::Z * E) - distance(cones, blend_k, p - Vec3::Z * E),
    )
    .normalize_or_zero()
}

/// Axis-aligned bounds of a resolved cone list, inflated by the blend radius
/// since `smin` pushes the surface outward where limbs meet.
pub fn bounds(cones: &[Cone], blend_k: f32) -> (Vec3, Vec3) {
    let mut lo = Vec3::splat(f32::MAX);
    let mut hi = Vec3::splat(f32::MIN);
    for cone in cones {
        for (end, r) in [(cone.head(), cone.a[3]), (cone.tail(), cone.b[3])] {
            lo = lo.min(end - Vec3::splat(r));
            hi = hi.max(end + Vec3::splat(r));
        }
    }
    let pad = Vec3::splat(blend_k);
    (lo - pad, hi + pad)
}
