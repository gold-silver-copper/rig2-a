//! Wrappers: models that add behaviour to another model of the same task.
//!
//! Each wrapper is generic over `M: Model<T>` and implements `Model<T>` (and
//! `StreamingModel<T>` where that makes sense), so they compose in any order
//! and work for every task, including tasks defined outside this crate.
//!
//! ```
//! # use std::sync::Arc;
//! # use std::time::Duration;
//! use rig2_core::{Model, Retry, Traced};
//! # use rig2_core::{completion::Completion, ModelInfo};
//! # fn wrap(model: Arc<dyn Model<Completion>>) -> impl Model<Completion> {
//! let wrapped = Traced::new(Retry::new(model).with_attempts(3).with_base_delay(Duration::from_millis(200)));
//! # wrapped }
//! ```

mod cached;
mod fallback;
mod rate_limited;
mod record;
mod retry;
mod traced;

pub use cached::Cached;
pub use fallback::Fallback;
pub use rate_limited::RateLimited;
pub use record::{Recorded, RecordedCall, RecordedOutcome, Recorder, Recording, Replay};
pub use retry::Retry;
pub use traced::Traced;

use std::time::Duration;

/// Wait for `delay` without blocking a thread, on native and wasm32 alike.
pub(crate) async fn sleep(delay: Duration) {
    futures_timer::Delay::new(delay).await;
}

#[cfg(test)]
mod tests;
