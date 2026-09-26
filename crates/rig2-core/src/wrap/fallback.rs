use std::sync::Arc;

use crate::{
    BoxFuture, BoxStream, Error, ErrorKind, Model, ModelInfo, Result, StreamingModel,
    StreamingTask, Task,
};

/// Tries models in order until one succeeds.
///
/// Any error except [`ErrorKind::Cancelled`] moves on to the next model; the
/// last model's error is returned. Streams fall back only while opening.
/// Info and capabilities are the first model's.
#[derive(Debug, Clone)]
pub struct Fallback<M> {
    models: Arc<Vec<M>>,
}

impl<M> Fallback<M> {
    /// Try `models` in order. Fails with [`ErrorKind::Config`] when empty.
    pub fn new(models: Vec<M>) -> Result<Self> {
        if models.is_empty() {
            return Err(Error::new(
                ErrorKind::Config,
                "a fallback needs at least one model",
            ));
        }
        Ok(Self {
            models: Arc::new(models),
        })
    }

    fn first(&self) -> &M {
        // `new` guarantees at least one model.
        &self.models[0]
    }
}

async fn first_success<M, O, F>(models: Arc<Vec<M>>, call: impl Fn(&M) -> F) -> Result<O>
where
    F: std::future::Future<Output = Result<O>>,
{
    let mut last = Error::new(ErrorKind::Config, "a fallback needs at least one model");
    for model in models.iter() {
        match call(model).await {
            Ok(output) => return Ok(output),
            Err(error) if error.kind() == ErrorKind::Cancelled => return Err(error),
            Err(error) => {
                tracing::debug!(%error, "falling back");
                last = error;
            }
        }
    }
    Err(last)
}

impl<T, M> Model<T> for Fallback<M>
where
    T: Task,
    T::Input: Clone,
    M: Model<T> + 'static,
{
    fn info(&self) -> &ModelInfo {
        self.first().info()
    }

    fn capabilities(&self) -> T::Capabilities {
        self.first().capabilities()
    }

    fn invoke(&self, input: T::Input) -> BoxFuture<'static, Result<T::Output>> {
        let models = Arc::clone(&self.models);
        Box::pin(first_success(models, move |model: &M| {
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
        let models = Arc::clone(&self.models);
        Box::pin(first_success(models, move |model: &M| {
            model.invoke_stream(input.clone())
        }))
    }
}
