use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, PoisonError};

use serde::Serialize;

use crate::{
    BoxFuture, BoxStream, MaybeSend, Model, ModelInfo, Result, StreamingModel, StreamingTask, Task,
};

/// Remembers outputs by input, and answers repeated inputs without a call.
///
/// Inputs are keyed by their JSON form. The oldest entry is evicted once the
/// cache holds `capacity` outputs. Errors are not cached. Streams pass
/// through uncached, because a stream is consumed as it is read.
#[derive(Debug, Clone)]
pub struct Cached<M, O> {
    inner: Arc<M>,
    cache: Arc<Mutex<Entries<O>>>,
}

#[derive(Debug)]
struct Entries<O> {
    capacity: usize,
    values: HashMap<String, O>,
    order: VecDeque<String>,
}

impl<O: Clone> Entries<O> {
    fn get(&self, key: &str) -> Option<O> {
        self.values.get(key).cloned()
    }

    fn insert(&mut self, key: String, value: O) {
        if self.values.insert(key.clone(), value).is_none() {
            self.order.push_back(key);
        }
        while self.order.len() > self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.values.remove(&oldest);
            }
        }
    }
}

impl<M, O> Cached<M, O> {
    /// Cache up to `capacity` outputs of `inner`.
    pub fn new(inner: M, capacity: usize) -> Self {
        Self {
            inner: Arc::new(inner),
            cache: Arc::new(Mutex::new(Entries {
                capacity: capacity.max(1),
                values: HashMap::new(),
                order: VecDeque::new(),
            })),
        }
    }
}

impl<T, M> Model<T> for Cached<M, T::Output>
where
    T: Task,
    T::Input: Serialize,
    T::Output: Clone + MaybeSend,
    M: Model<T> + 'static,
{
    fn info(&self) -> &ModelInfo {
        self.inner.info()
    }

    fn capabilities(&self) -> T::Capabilities {
        self.inner.capabilities()
    }

    fn invoke(&self, input: T::Input) -> BoxFuture<'static, Result<T::Output>> {
        let key = match serde_json::to_string(&input) {
            Ok(key) => key,
            Err(error) => return Box::pin(std::future::ready(Err(error.into()))),
        };
        if let Some(hit) = self
            .cache
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(&key)
        {
            return Box::pin(std::future::ready(Ok(hit)));
        }
        let (call, cache) = (self.inner.invoke(input), Arc::clone(&self.cache));
        Box::pin(async move {
            let output = call.await?;
            cache
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .insert(key, output.clone());
            Ok(output)
        })
    }
}

impl<T, M> StreamingModel<T> for Cached<M, T::Output>
where
    T: StreamingTask,
    T::Input: Serialize,
    T::Output: Clone + MaybeSend,
    M: StreamingModel<T> + 'static,
{
    fn invoke_stream(
        &self,
        input: T::Input,
    ) -> BoxFuture<'static, Result<BoxStream<'static, Result<T::Event>>>> {
        self.inner.invoke_stream(input)
    }
}
