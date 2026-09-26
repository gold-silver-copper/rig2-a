use std::sync::Arc;

use crate::{
    BoxFuture, BoxStream, ErrorKind, MaybeSend, MaybeSync, Model, ModelInfo, Result,
    StreamingModel, StreamingTask, Task,
};

/// Tries models in order until one succeeds.
///
/// Any error except [`ErrorKind::Cancelled`] moves on to the next model; the
/// last model's error is returned. Streams fall back only while opening.
/// Info and capabilities are the first model's.
#[derive(Debug)]
pub struct Fallback<M> {
    first: Arc<M>,
    rest: Arc<[M]>,
}

impl<M> Clone for Fallback<M> {
    fn clone(&self) -> Self {
        Self {
            first: Arc::clone(&self.first),
            rest: Arc::clone(&self.rest),
        }
    }
}

impl<M> Fallback<M> {
    /// Try `first`, then each of `rest` in order.
    pub fn new(first: M, rest: impl IntoIterator<Item = M>) -> Self {
        Self {
            first: Arc::new(first),
            rest: rest.into_iter().collect(),
        }
    }
}

async fn first_success<M: MaybeSend + MaybeSync, O: MaybeSend>(
    first: Arc<M>,
    rest: Arc<[M]>,
    call: impl Fn(&M) -> BoxFuture<'static, Result<O>> + MaybeSend,
) -> Result<O> {
    let mut outcome = call(&first).await;
    for model in rest.iter() {
        match outcome {
            Err(error) if error.kind() != ErrorKind::Cancelled => {
                tracing::debug!(%error, "falling back");
                outcome = call(model).await;
            }
            done => return done,
        }
    }
    outcome
}

impl<T, M> Model<T> for Fallback<M>
where
    T: Task,
    T::Input: Clone,
    M: Model<T> + 'static,
{
    fn info(&self) -> &ModelInfo {
        self.first.info()
    }

    fn capabilities(&self) -> T::Capabilities {
        self.first.capabilities()
    }

    fn invoke(&self, input: T::Input) -> BoxFuture<'static, Result<T::Output>> {
        let (first, rest) = (Arc::clone(&self.first), Arc::clone(&self.rest));
        Box::pin(first_success(first, rest, move |model: &M| {
            model.invoke(input.clone())
        }))
    }
}

impl<T, M> StreamingModel<T> for Fallback<M>
where
    T: StreamingTask,
    T::Input: Clone,
    M: StreamingModel<T> + 'static,
{
    fn invoke_stream(
        &self,
        input: T::Input,
    ) -> BoxFuture<'static, Result<BoxStream<'static, Result<T::Event>>>> {
        let (first, rest) = (Arc::clone(&self.first), Arc::clone(&self.rest));
        Box::pin(first_success(first, rest, move |model: &M| {
            model.invoke_stream(input.clone())
        }))
    }
}
