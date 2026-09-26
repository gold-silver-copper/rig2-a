//! The core of rig2: tasks, models, content and the transport contract.
//!
//! Every model call in rig2 is a `Model` of some task. This crate has
//! no provider, transport or async runtime dependency, and builds for
//! `wasm32-unknown-unknown`.

mod send;

pub use send::{BoxFuture, BoxStream, MaybeSend, MaybeSync};
