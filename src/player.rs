//! Phases 3 and 4: collision, locomotion, and the walk cycle.
//!
//! Collision never samples the smooth-minned surface. The capsule comes from
//! the character's own resolved bounds, so the thing that collides and the
//! thing that is drawn are derived from one definition and cannot disagree
//! about where the character is.

use glam::{Quat, Vec3};

use crate::sdf::{self, joint, Pose, Skeleton};
use crate::world::World;

#[derive(Clone, Copy, Default)]
pub struct Input {
    pub forward: f32, // -1..1
    pub strafe: f32,  // -1..1
    pub turn: f32,    // -1..1, radians per second scale
    pub jump: bool,
    pub run: bool,
}

pub struct Player {
    pub position: Vec3, // feet
    pub velocity: Vec3,
    pub yaw: f32,
    pub grounded: bool,
    pub radius: f32,
    pub height: f32,
    /// Advances with distance travelled, not time, so the stride stays
    /// locked to the ground however fast the character moves.
    pub stride: f32,
}

const GRAVITY: f32 = -18.0;
const WALK_SPEED: f32 = 2.6;
const RUN_SPEED: f32 = 5.4;
const ACCEL: f32 = 14.0;
const JUMP_SPEED: f32 = 6.2;
const TURN_RATE: f32 = 2.8;
/// Push-out iterations. Two is enough for a convex corner; more only matters
/// where several surfaces meet.
const SOLVER_PASSES: usize = 4;
/// Tallest ledge the character can walk onto without jumping.
const MAX_STEP: f32 = 0.45;

impl Player {
    /// Fit the collision capsule to the character's rest pose.
    pub fn new(skeleton: &Skeleton, position: Vec3) -> Self {
        let mut cones = Vec::new();
        skeleton.resolve(&Pose::rest(skeleton), &mut cones);
        let (lo, hi) = sdf::bounds(&cones, skeleton.blend_k);
        let height = hi.y - lo.y;
        // Narrower than the shoulders on purpose: a capsule as wide as the
        // arms snags on every doorway, and the arms are not load bearing.
        let radius = ((hi.x - lo.x) * 0.30).clamp(0.12, height * 0.22);
        Self {
            position,
            velocity: Vec3::ZERO,
            yaw: 0.0,
            grounded: false,
            radius,
            height,
            stride: 0.0,
        }
    }

    pub fn forward(&self) -> Vec3 {
        Vec3::new(self.yaw.sin(), 0.0, self.yaw.cos())
    }

    fn right(&self) -> Vec3 {
        Vec3::new(self.yaw.cos(), 0.0, -self.yaw.sin())
    }

    /// Points down the capsule axis, from the lower cap centre to the upper.
    fn axis_samples(&self, position: Vec3) -> [Vec3; 4] {
        let bottom = position.y + self.radius;
        let top = position.y + self.height - self.radius;
        let span = (top - bottom).max(0.0);
        [
            Vec3::new(position.x, bottom, position.z),
            Vec3::new(position.x, bottom + span * 0.34, position.z),
            Vec3::new(position.x, bottom + span * 0.67, position.z),
            Vec3::new(position.x, top, position.z),
        ]
    }

    pub fn update(&mut self, input: Input, world: &World, dt: f32) {
        self.yaw += input.turn * TURN_RATE * dt;

        let speed = if input.run { RUN_SPEED } else { WALK_SPEED };
        let mut wish = self.forward() * input.forward + self.right() * input.strafe;
        if wish.length_squared() > 1.0 {
            wish = wish.normalize();
        }
        let wish = wish * speed;

        // Horizontal control, full authority on the ground and partial in the
        // air so a jump cannot be steered like a walk.
        let authority = if self.grounded { 1.0 } else { 0.35 };
        let horizontal = Vec3::new(self.velocity.x, 0.0, self.velocity.z);
        let blended = horizontal + (wish - horizontal) * (ACCEL * authority * dt).min(1.0);
        self.velocity.x = blended.x;
        self.velocity.z = blended.z;

        if input.jump && self.grounded {
            self.velocity.y = JUMP_SPEED;
            self.grounded = false;
        }
        self.velocity.y += GRAVITY * dt;

        let was_grounded = self.grounded;
        let start = self.position;
        let planar_step = (self.velocity * dt).with_y(0.0);

        self.position += self.velocity * dt;
        self.resolve(world);

        // Step-up. A capsule cannot climb anything taller than its own
        // radius: the lower cap hits the riser and the solver pushes it
        // straight back, so a walking character simply stops dead at a kerb.
        // When a grounded move loses most of its horizontal progress, retry it
        // from a lifted start and drop back onto whatever is under the result.
        let wanted = planar_step.length();
        let gained = (self.position - start).with_y(0.0).length();
        if was_grounded && wanted > 1.0e-5 && gained < wanted * 0.7 {
            let blocked = (self.position, self.velocity);
            self.position = start + Vec3::Y * MAX_STEP + planar_step;
            self.resolve(world);

            let foot = self.position + Vec3::Y * self.radius;
            let clearance = world.distance(foot) - self.radius;
            let landed = (-0.02..=MAX_STEP + 0.05).contains(&clearance);
            let climbed = (self.position - start).with_y(0.0).length();
            if landed && climbed > gained {
                self.position.y -= clearance;
            } else {
                (self.position, self.velocity) = blocked;
            }
        }

        self.ground_check(world);

        let planar_speed = Vec3::new(self.velocity.x, 0.0, self.velocity.z).length();
        self.stride += (self.position - start).with_y(0.0).length()
            / (self.height * 0.42).max(0.1);
        if planar_speed < 0.15 && self.grounded {
            // Ease back to a neutral stance rather than freezing mid-step.
            // Keyed on speed, not on distance moved: a character pressed
            // against a wall is still trying to walk and should keep its gait.
            let target = (self.stride / std::f32::consts::TAU).round() * std::f32::consts::TAU;
            self.stride += (target - self.stride) * (8.0 * dt).min(1.0);
        }
    }

    /// Push the capsule out of anything it is inside, along the world
    /// gradient. `-d` is the penetration depth and the gradient is the
    /// normal, both from the same field the renderer draws.
    fn resolve(&mut self, world: &World) {
        for _ in 0..SOLVER_PASSES {
            let mut correction = Vec3::ZERO;
            let mut hits = 0.0f32;
            for sample in self.axis_samples(self.position) {
                let d = world.distance(sample);
                if d < self.radius {
                    correction += world.normal(sample) * (self.radius - d);
                    hits += 1.0;
                }
            }
            if hits == 0.0 {
                break;
            }
            let push = correction / hits;
            self.position += push;

            // Kill the velocity component driving into the surface, or the
            // character jitters against a wall it keeps re-entering.
            if push.length_squared() > 1.0e-12 {
                let n = push.normalize();
                let into = self.velocity.dot(n);
                if into < 0.0 {
                    self.velocity -= n * into;
                }
            }
        }
    }

    fn ground_check(&mut self, world: &World) {
        let foot = Vec3::new(self.position.x, self.position.y + self.radius, self.position.z);
        let clearance = world.distance(foot) - self.radius;
        let standing = clearance < 0.06 && self.velocity.y <= 0.01;
        if standing {
            self.position.y -= clearance;
            if self.velocity.y < 0.0 {
                self.velocity.y = 0.0;
            }
        }
        self.grounded = standing;
    }

    /// Phase 4. A walk cycle is two counter-phased legs, opposed arms, and a
    /// small vertical bob at twice the stride frequency. Amplitude scales
    /// with speed so a standing character is still.
    pub fn pose(&self, skeleton: &Skeleton) -> Pose {
        let mut pose = Pose::rest(skeleton);
        pose.root = self.position;
        pose.yaw = self.yaw;

        let planar = Vec3::new(self.velocity.x, 0.0, self.velocity.z).length();
        let gait = (planar / RUN_SPEED).clamp(0.0, 1.0);
        let swing = gait * 0.85;
        let phase = self.stride;

        let pitch = |angle: f32| Quat::from_rotation_x(angle);

        if self.grounded {
            let (l, r) = (phase.sin(), (phase + std::f32::consts::PI).sin());

            pose.local[joint::THIGH_L] = pitch(l * swing * 0.55);
            pose.local[joint::THIGH_R] = pitch(r * swing * 0.55);
            // Knees only bend one way, hence the max(0.0).
            pose.local[joint::SHIN_L] = pitch(-(l * 0.5 - 0.35).max(0.0) * swing * 1.9);
            pose.local[joint::SHIN_R] = pitch(-(r * 0.5 - 0.35).max(0.0) * swing * 1.9);
            pose.local[joint::FOOT_L] = pitch(-l * swing * 0.25);
            pose.local[joint::FOOT_R] = pitch(-r * swing * 0.25);

            pose.local[joint::UPPER_ARM_L] = pitch(r * swing * 0.45);
            pose.local[joint::UPPER_ARM_R] = pitch(l * swing * 0.45);
            pose.local[joint::FOREARM_L] = pitch(-(r * 0.5 + 0.5) * swing * 0.5);
            pose.local[joint::FOREARM_R] = pitch(-(l * 0.5 + 0.5) * swing * 0.5);

            pose.local[joint::SPINE] = Quat::from_rotation_y(-l * swing * 0.12);
            pose.local[joint::CHEST] = Quat::from_rotation_y(l * swing * 0.18);
            pose.root.y += (phase * 2.0).sin() * swing * self.height * 0.012;
        } else {
            // Tuck in the air so a jump does not look like a standing glide.
            let tuck = (self.velocity.y * -0.06).clamp(-0.3, 0.6);
            pose.local[joint::THIGH_L] = pitch(0.45 + tuck);
            pose.local[joint::THIGH_R] = pitch(0.30 + tuck);
            pose.local[joint::SHIN_L] = pitch(-0.85 - tuck);
            pose.local[joint::SHIN_R] = pitch(-0.55 - tuck);
            pose.local[joint::UPPER_ARM_L] = pitch(-0.5);
            pose.local[joint::UPPER_ARM_R] = pitch(-0.5);
            pose.local[joint::FOREARM_L] = pitch(-0.6);
            pose.local[joint::FOREARM_R] = pitch(-0.6);
        }

        pose
    }
}
