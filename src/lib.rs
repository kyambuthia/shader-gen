//! A playable character defined entirely as a signed distance function.
//!
//! See docs/DESIGN.md. The short version: one function `f(p)` is both the
//! surface the renderer sphere-traces and the field collision queries, so the
//! visual and the collider cannot drift apart - provided the CPU and GPU
//! evaluators agree, which is what tests/parity.rs enforces.

pub mod body;
pub mod gpu;
pub mod player;
pub mod sdf;
pub mod world;
