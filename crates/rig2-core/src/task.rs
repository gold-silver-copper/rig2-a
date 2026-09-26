//! Tasks and the models that perform them.
//!
//! A [`Task`] names a kind of model call and fixes its input and output
//! types. A [`Model<T>`] performs task `T`. Everything generic in rig2
//! (tracing, retries, recording, agents, Bevy) is written once over these two
//! traits, so a task defined in another crate gets all of it for free.
//!
//! ```
//! use rig2_core::{Model, Task};
//!
//! /// Counts the words in a string.
//! struct WordCount;
//!
//! impl Task for WordCount {
//!     const NAME: &'static str = "word_count";
//!     type Input = String;
//!     type Output = usize;
//!     type Capabilities = ();
//! }
//! # let _ = std::marker::PhantomData::<WordCount>;
//! ```

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{BoxFuture, BoxStream, MaybeSend, MaybeSync, Result};

/// A kind of model call: its name, input, output and capabilities.
pub trait Task: 'static {
    /// The operation name, used in spans, recordings and the registry.
    const NAME: &'static str;
    /// What a call takes.
    type Input: MaybeSend + 'static;
    /// What a call returns.
    type Output: MaybeSend + 'static;
    /// What a model of this task declares it can do.
    type Capabilities: Clone + MaybeSend + MaybeSync + 'static;

    /// Record task-specific span fields (token usage, counts) from an output.
    ///
    /// Only fields [`Traced`](crate::Traced) declares can be recorded; see its
    /// documentation for the list.
    fn record(_span: &tracing::Span, _output: &Self::Output) {}
}

/// A task whose results can arrive incrementally.
pub trait StreamingTask: Task {
    /// One increment of a streamed result.
    type Event: MaybeSend + 'static;

    /// Record span fields from a streamed event. Called for every event; the
    /// task records from the one that carries totals (for completion, the
    /// finish).
    fn record_event(_span: &tracing::Span, _event: &Self::Event) {}
}

/// Which provider and model a [`Model`] calls.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelInfo {
    /// The provider name, for example `openai`.
    pub provider: String,
    /// The provider's model id, for example `gpt-5-mini`.
    pub model: String,
}

impl ModelInfo {
    /// A provider and model id.
    pub fn new(provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
        }
    }
}

/// A model that performs task `T`.
///
/// Implementers write [`invoke`](Model::invoke). Callers use
/// [`run`](Model::run), which converts its argument into `T::Input`. The
/// trait is object-safe: `Arc<dyn Model<T>>` is the erased model, and it
/// implements `Model<T>` itself.
///
/// The returned future is `'static`, so a call can be spawned. An
/// implementation clones what it needs (usually one `Arc`) into the future,
/// and boxes the future once.
pub trait Model<T: Task>: MaybeSend + MaybeSync {
    /// The provider and model id.
    fn info(&self) -> &ModelInfo;

    /// What this model declares it can do.
    fn capabilities(&self) -> T::Capabilities;

    /// Perform one call.
    fn invoke(&self, input: T::Input) -> BoxFuture<'static, Result<T::Output>>;

    /// Perform one call with any input that converts into `T::Input`.
    fn run(&self, input: impl Into<T::Input>) -> BoxFuture<'static, Result<T::Output>>
    where
        Self: Sized,
    {
        self.invoke(input.into())
    }
}

/// A model whose results can be streamed.
pub trait StreamingModel<T: StreamingTask>: Model<T> {
    /// Open a stream of results.
    ///
    /// The outer future resolves once the stream is open (for HTTP, once the
    /// response headers arrive); errors after that arrive in the stream, and
    /// end it.
    fn invoke_stream(
        &self,
        input: T::Input,
    ) -> BoxFuture<'static, Result<BoxStream<'static, Result<T::Event>>>>;

    /// Open a stream with any input that converts into `T::Input`.
    fn stream(
        &self,
        input: impl Into<T::Input>,
    ) -> BoxFuture<'static, Result<BoxStream<'static, Result<T::Event>>>>
    where
        Self: Sized,
    {
        self.invoke_stream(input.into())
    }
}

impl<T: Task, M: Model<T> + ?Sized> Model<T> for Arc<M> {
    fn info(&self) -> &ModelInfo {
        (**self).info()
    }

    fn capabilities(&self) -> T::Capabilities {
        (**self).capabilities()
    }

    fn invoke(&self, input: T::Input) -> BoxFuture<'static, Result<T::Output>> {
        (**self).invoke(input)
    }
}

impl<T: StreamingTask, M: StreamingModel<T> + ?Sized> StreamingModel<T> for Arc<M> {
    fn invoke_stream(
        &self,
        input: T::Input,
    ) -> BoxFuture<'static, Result<BoxStream<'static, Result<T::Event>>>> {
        (**self).invoke_stream(input)
    }
}

#[cfg(test)]
mod tests;
