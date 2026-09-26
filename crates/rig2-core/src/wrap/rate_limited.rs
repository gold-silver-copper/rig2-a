use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use web_time::Instant;

use super::sleep;
use crate::{BoxFuture, BoxStream, Model, ModelInfo, Result, StreamingModel, StreamingTask, Task};

/// Limits how many calls start per interval and how many run at once.
///
/// Calls beyond the rate wait for a slot instead of failing. Clones share
/// the same limits, so one `RateLimited` can guard several call sites.
#[derive(Debug, Clone)]
pub struct RateLimited<M> {
    inner: Arc<M>,
    limits: Arc<Limits>,
}

#[derive(Debug)]
struct Limits {
    bucket: Mutex<Bucket>,
    capacity: f64,
    refill_per_sec: f64,
    concurrency: async_lock::Semaphore,
}

#[derive(Debug)]
struct Bucket {
    tokens: f64,
    refilled: Instant,
}

impl Limits {
    /// Take one token, or say how long until one is available.
    fn take(&self) -> Option<Duration> {
        let mut bucket = self.bucket.lock().unwrap_or_else(PoisonError::into_inner);
        let now = Instant::now();
        let elapsed = now.duration_since(bucket.refilled).as_secs_f64();
        bucket.tokens = elapsed
            .mul_add(self.refill_per_sec, bucket.tokens)
            .min(self.capacity);
        bucket.refilled = now;
        if bucket.tokens >= 1.0 {
            bucket.tokens -= 1.0;
            None
        } else {
            Some(Duration::from_secs_f64(
                (1.0 - bucket.tokens) / self.refill_per_sec,
            ))
        }
    }

    async fn acquire(&self) -> async_lock::SemaphoreGuard<'_> {
        let guard = self.concurrency.acquire().await;
        while let Some(wait) = self.take() {
            sleep(wait).await;
        }
        guard
    }
}

impl<M> RateLimited<M> {
    /// Allow `requests` call starts per `interval`, with at most
    /// `max_concurrent` calls in flight.
    pub fn new(inner: M, requests: u32, interval: Duration, max_concurrent: usize) -> Self {
        let capacity = f64::from(requests.max(1));
        let refill_per_sec = capacity / interval.as_secs_f64().max(f64::EPSILON);
        Self {
            inner: Arc::new(inner),
            limits: Arc::new(Limits {
                bucket: Mutex::new(Bucket {
                    tokens: capacity,
                    refilled: Instant::now(),
                }),
                capacity,
                refill_per_sec,
                concurrency: async_lock::Semaphore::new(max_concurrent.max(1)),
            }),
        }
    }
}

impl<T: Task, M: Model<T> + 'static> Model<T> for RateLimited<M> {
    fn info(&self) -> &ModelInfo {
        self.inner.info()
    }

    fn capabilities(&self) -> T::Capabilities {
        self.inner.capabilities()
    }

    fn invoke(&self, input: T::Input) -> BoxFuture<'static, Result<T::Output>> {
        let (inner, limits) = (Arc::clone(&self.inner), Arc::clone(&self.limits));
        Box::pin(async move {
            let _slot = limits.acquire().await;
            inner.invoke(input).await
        })
    }
}

impl<T: StreamingTask, M: StreamingModel<T> + 'static> StreamingModel<T> for RateLimited<M> {
    /// The concurrency slot is held until the stream is open.
    fn invoke_stream(
        &self,
        input: T::Input,
    ) -> BoxFuture<'static, Result<BoxStream<'static, Result<T::Event>>>> {
        let (inner, limits) = (Arc::clone(&self.inner), Arc::clone(&self.limits));
        Box::pin(async move {
            let _slot = limits.acquire().await;
            inner.invoke_stream(input).await
        })
    }
}
