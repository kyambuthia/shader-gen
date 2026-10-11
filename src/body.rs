//! Phase 5: a seed becomes a character.
//!
//! Generation is deliberately the last phase of docs/DESIGN.md. It is the
//! cheapest part once the pipeline exists - the character is about a dozen
//! numbers, not a mesh - and doing it earlier tempts you to tune proportions
//! before you can see them move.

use glam::Vec3;

use crate::sdf::{BoneSpec, Skeleton};

/// Deterministic, self-contained, and good enough to decorrelate a handful of
/// proportions from a seed. Not a general-purpose RNG.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15).wrapping_add(1))
    }

    fn next(&mut self) -> f32 {
        // SplitMix64.
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u32 << 24) as f32
    }

    /// Centred on the midpoint, so seed 0 is not systematically extreme.
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next()
    }
}

/// The handful of numbers that define a body. Everything else is derived.
#[derive(Clone, Copy, Debug)]
pub struct Proportions {
    pub height: f32,
    /// Scales every radius: thin to heavy.
    pub bulk: f32,
    /// Share of standing height taken by the legs.
    pub leg_ratio: f32,
    pub shoulder_width: f32,
    pub head_scale: f32,
    pub arm_scale: f32,
}

impl Default for Proportions {
    fn default() -> Self {
        Self {
            height: 1.80,
            bulk: 1.0,
            leg_ratio: 0.50,
            shoulder_width: 1.0,
            head_scale: 1.0,
            arm_scale: 1.0,
        }
    }
}

impl Proportions {
    pub fn from_seed(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        Self {
            height: rng.range(1.45, 2.05),
            bulk: rng.range(0.78, 1.35),
            leg_ratio: rng.range(0.45, 0.55),
            shoulder_width: rng.range(0.82, 1.25),
            head_scale: rng.range(0.88, 1.15),
            arm_scale: rng.range(0.90, 1.12),
        }
    }
}

/// Build a skeleton in model space, feet on y = 0, facing +Z.
///
/// Authored as head/tail pairs because that is what a person can reason
/// about; `Skeleton::new` converts to the parent-relative rotations posing
/// needs.
pub fn humanoid(p: Proportions) -> Skeleton {
    let h = p.height;
    let hip = h * p.leg_ratio; // hip height off the ground
    let r = |v: f32| v * h * p.bulk; // radius as a fraction of height

    let shoulder_y = hip + (h - hip) * 0.62;
    let shoulder_x = h * 0.105 * p.shoulder_width;
    let hip_x = h * 0.055;

    // Arms hang clear of the chest: the torso surface reaches
    // chest_radius, so an upper arm any further inboard is swallowed by it
    // once blend_k is applied.
    let chest_r = r(0.092);
    let arm_x = shoulder_x.max(chest_r / (h * p.bulk) * h * p.bulk + r(0.040)) + r(0.010);
    let arm_len = (shoulder_y - hip * 0.52) * p.arm_scale;

    let v = |x: f32, y: f32, z: f32| Vec3::new(x, y, z);
    let mut specs = vec![
        // 0 pelvis
        BoneSpec { parent: -1, head: v(0.0, hip - h * 0.012, 0.0), tail: v(0.0, hip + h * 0.045, 0.0), r_head: r(0.080), r_tail: r(0.075) },
        // 1 spine
        BoneSpec { parent: 0, head: v(0.0, hip + h * 0.045, 0.0), tail: v(0.0, hip + h * 0.145, 0.0), r_head: r(0.072), r_tail: r(0.086) },
        // 2 chest
        BoneSpec { parent: 1, head: v(0.0, hip + h * 0.145, 0.0), tail: v(0.0, shoulder_y + h * 0.015, 0.0), r_head: chest_r, r_tail: r(0.080) },
        // 3 neck
        BoneSpec { parent: 2, head: v(0.0, shoulder_y + h * 0.015, 0.0), tail: v(0.0, shoulder_y + h * 0.075, 0.0), r_head: r(0.030), r_tail: r(0.033) },
        // 4 head
        BoneSpec { parent: 3, head: v(0.0, shoulder_y + h * 0.075, 0.0), tail: v(0.0, h - h * 0.012, 0.0), r_head: r(0.058) * p.head_scale, r_tail: r(0.053) * p.head_scale },
    ];

    for side in [1.0f32, -1.0] {
        let parent_chest = 2;
        let clavicle = specs.len() as i32;
        specs.push(BoneSpec {
            parent: parent_chest,
            head: v(side * h * 0.012, shoulder_y, 0.0),
            tail: v(side * arm_x, shoulder_y - h * 0.012, 0.0),
            r_head: r(0.047),
            r_tail: r(0.038),
        });
        specs.push(BoneSpec {
            parent: clavicle,
            head: v(side * arm_x, shoulder_y - h * 0.012, 0.0),
            tail: v(side * (arm_x + h * 0.018), shoulder_y - arm_len * 0.52, 0.0),
            r_head: r(0.038),
            r_tail: r(0.029),
        });
        specs.push(BoneSpec {
            parent: clavicle + 1,
            head: v(side * (arm_x + h * 0.018), shoulder_y - arm_len * 0.52, 0.0),
            tail: v(side * (arm_x + h * 0.032), shoulder_y - arm_len * 0.95, 0.0),
            r_head: r(0.029),
            r_tail: r(0.022),
        });
        specs.push(BoneSpec {
            parent: clavicle + 2,
            head: v(side * (arm_x + h * 0.032), shoulder_y - arm_len * 0.95, 0.0),
            tail: v(side * (arm_x + h * 0.036), shoulder_y - arm_len * 1.05, 0.0),
            r_head: r(0.024),
            r_tail: r(0.018),
        });
    }

    for side in [1.0f32, -1.0] {
        let thigh = specs.len() as i32;
        specs.push(BoneSpec {
            parent: 0,
            head: v(side * hip_x, hip, 0.0),
            tail: v(side * hip_x, hip * 0.46, 0.0),
            r_head: r(0.056),
            r_tail: r(0.042),
        });
        specs.push(BoneSpec {
            parent: thigh,
            head: v(side * hip_x, hip * 0.46, 0.0),
            tail: v(side * hip_x, h * 0.042, 0.0),
            r_head: r(0.040),
            r_tail: r(0.028),
        });
        specs.push(BoneSpec {
            parent: thigh + 1,
            head: v(side * hip_x, h * 0.028, -h * 0.012),
            tail: v(side * hip_x, h * 0.022, h * 0.075),
            r_head: r(0.031),
            r_tail: r(0.025),
        });
    }

    Skeleton::new(&specs, r(0.020))
}

pub fn from_seed(seed: u64) -> Skeleton {
    humanoid(Proportions::from_seed(seed))
}
