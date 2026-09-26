//! The core of rig2: tasks, models, content, streams and the transport
//! contract.
//!
//! Every model call in rig2 is a [`Model<T>`] of some [`Task`] `T`:
//! completion, embedding, image generation, vision, or a task you define.
//! Everything generic is written once over those two traits: the wrappers
//! ([`Traced`], [`Retry`], [`RateLimited`], [`Fallback`], [`Cached`],
//! [`Recorded`] and [`Replay`]), structured output ([`extract`]), and, in
//! other crates, agents and the Bevy integration.
//!
//! This crate has no provider, transport or async runtime dependency, and
//! builds for `wasm32-unknown-unknown`.
//!
//! ```
//! use std::sync::Arc;
//! use rig2_core::completion::{Completion, CompletionResponse, Finish, respond};
//! use rig2_core::content::{AssistantContent, Text};
//! use rig2_core::{BoxFuture, Model, ModelInfo, Result};
//! use rig2_core::catalog::ModelCard;
//!
//! /// A model that always says hello.
//! struct Hello(ModelInfo);
//!
//! impl Model<Completion> for Hello {
//!     fn info(&self) -> &ModelInfo { &self.0 }
//!     fn capabilities(&self) -> ModelCard { ModelCard::unknown("hello") }
//!     fn invoke(&self, _request: rig2_core::completion::CompletionRequest) -> BoxFuture<'static, Result<CompletionResponse>> {
//!         Box::pin(async {
//!             respond(|out| {
//!                 out.part(AssistantContent::Text(Text::new("hello")));
//!                 Ok(Finish::default())
//!             })
//!         })
//!     }
//! }
//!
//! let model: Arc<dyn Model<Completion>> = Arc::new(Hello(ModelInfo::new("demo", "hello")));
//! let reply = futures::executor::block_on(model.run("hi")).unwrap();
//! assert_eq!(reply.text(), "hello");
//! ```

mod base64_bytes;
mod error;
mod send;
mod task;

pub mod catalog;
pub mod completion;
pub mod content;
pub mod extract;
pub mod http;
pub mod spec;
pub mod store;
pub mod tasks;
pub mod vision;
pub mod wrap;

pub use error::{Error, ErrorKind, Result};
pub use send::{BoxFuture, BoxStream, MaybeSend, MaybeSync};
pub use task::{Model, ModelInfo, StreamingModel, StreamingTask, Task};
pub use wrap::{
    Cached, Fallback, RateLimited, Recorded, Recorder, Recording, Replay, Retry, Traced,
};
