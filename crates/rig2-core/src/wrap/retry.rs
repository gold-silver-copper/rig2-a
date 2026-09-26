use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use super::sleep;
use crate::{
    BoxFuture, BoxStream, Error, MaybeSync, Model, ModelInfo, Result, StreamingModel,
    StreamingTask, Task,
};

/// Retries calls that fail with a retryable error.
///
/// A retryable error is one where [`Error::is_retryable`] holds: transport
/// failures, timeouts, rate limits and provider outages. Between attempts it
/// waits for the provider's `retry_after` when given, and otherwise for an
/// exponential backoff from the base delay, capped at the maximum delay.
/// Streams are retried only while opening; once events flow, an error ends
/// the stream.
#[derive(Debug, Clone)]
pub struct Retry<M> {
    inner: Arc<M>,
    policy: Policy,
}

#[derive(Debug, Clone, Copy)]
struct Policy {
    attempts: u32,
    base_delay: Duration,
    max_delay: Duration,
}

impl Policy {
    fn delay(self, attempt: u32, error: &Error) -> Duration {
        let backoff = self
            .base_delay
            .saturating_mul(2_u32.saturating_pow(attempt.saturating_sub(1)));
        error.retry_after().unwrap_or(backoff).min(self.max_delay)
    }

    async fn run<F: Future<Output = Result<O>>, O>(self, mut call: impl FnMut() -> F) -> Result<O> {
        let mut attempt = 1;
        loop {
            match call().await {
                Err(error) if error.is_retryable() && attempt < self.attempts => {
                    let delay = self.delay(attempt, &error);
                    tracing::debug!(attempt, ?delay, %error, "retrying");
                    sleep(delay).await;
                    attempt += 1;
                }
                result => return result,
            }
        }
    }
}

impl<M> Retry<M> {
    /// Up to 3 attempts in total, backing off from 500 ms, capped at 30 s.
    pub fn new(inner: M) -> Self {
        Self {
            inner: Arc::new(inner),
            policy: Policy {
                attempts: 3,
                base_delay: Duration::from_millis(500),
                max_delay: Duration::from_secs(30),
            },
        }
    }

    /// Set the total number of attempts, at least 1.
    pub fn with_attempts(mut self, attempts: u32) -> Self {
        self.policy.attempts = attempts.max(1);
        self
    }

    /// Set the first backoff delay.
    pub fn with_base_delay(mut self, delay: Duration) -> Self {
        self.policy.base_delay = delay;
        self
    }

    /// Cap every delay, including a provider's `retry_after`.
    pub fn with_max_delay(mut self, delay: Duration) -> Self {
        self.policy.max_delay = delay;
        self
    }
}

impl<T, M> Model<T> for Retry<M>
where
    T: Task,
    T::Input: Clone + MaybeSync,
    M: Model<T> + 'static,
{
    fn info(&self) -> &ModelInfo {
        self.inner.info()
    }

    fn capabilities(&self) -> T::Capabilities {
        self.inner.capabilities()
    }

    fn invoke(&self, input: T::Input) -> BoxFuture<'static, Result<T::Output>> {
        let inner = Arc::clone(&self.inner);
        Box::pin(self.policy.run(move || inner.invoke(input.clone())))
    }
}

impl<T, M> StreamingModel<T> for Retry<M>
where
    T: StreamingTask,
    T::Input: Clone + MaybeSync,
    M: StreamingModel<T> + 'static,
{
    fn invoke_stream(
        &self,
        input: T::Input,
    ) -> BoxFuture<'static, Result<BoxStream<'static, Result<T::Event>>>> {
        let inner = Arc::clone(&self.inner);
        Box::pin(self.policy.run(move || inner.invoke_stream(input.clone())))
    }
}
