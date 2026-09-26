use futures::StreamExt;
use tracing::Instrument;
use tracing::field::Empty;

use crate::{BoxFuture, BoxStream, Model, ModelInfo, Result, StreamingModel, StreamingTask, Task};

/// Opens one `tracing` span per call, following the OpenTelemetry GenAI
/// semantic conventions.
///
/// The span is named `gen_ai` with `otel.name` set to `"{operation} {model}"`,
/// and declares these fields, which [`Task::record`] and
/// [`StreamingTask::record_event`] may fill: `gen_ai.operation.name`,
/// `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.response.model`,
/// `gen_ai.response.id`, `gen_ai.response.finish_reasons`,
/// `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`,
/// `rig2.cost_usd` and `error.type`.
///
/// Nothing is required of the subscriber; without one the span is free.
#[derive(Debug, Clone)]
pub struct Traced<M> {
    inner: M,
}

impl<M> Traced<M> {
    /// Trace every call to `inner`.
    pub fn new(inner: M) -> Self {
        Self { inner }
    }
}

fn span<T: Task>(info: &ModelInfo) -> tracing::Span {
    tracing::info_span!(
        "gen_ai",
        otel.name = %format_args!("{} {}", T::NAME, info.model),
        otel.kind = "client",
        gen_ai.operation.name = T::NAME,
        gen_ai.provider.name = %info.provider,
        gen_ai.request.model = %info.model,
        gen_ai.response.model = Empty,
        gen_ai.response.id = Empty,
        gen_ai.response.finish_reasons = Empty,
        gen_ai.usage.input_tokens = Empty,
        gen_ai.usage.output_tokens = Empty,
        rig2.cost_usd = Empty,
        error.type = Empty,
    )
}

fn record_error(span: &tracing::Span, error: &crate::Error) {
    span.record("error.type", tracing::field::debug(error.kind()));
}

impl<T: Task, M: Model<T>> Model<T> for Traced<M> {
    fn info(&self) -> &ModelInfo {
        self.inner.info()
    }

    fn capabilities(&self) -> T::Capabilities {
        self.inner.capabilities()
    }

    fn invoke(&self, input: T::Input) -> BoxFuture<'static, Result<T::Output>> {
        let span = span::<T>(self.inner.info());
        let call = span.in_scope(|| self.inner.invoke(input));
        Box::pin(async move {
            let result = call.instrument(span.clone()).await;
            match &result {
                Ok(output) => T::record(&span, output),
                Err(error) => record_error(&span, error),
            }
            result
        })
    }
}

impl<T: StreamingTask, M: StreamingModel<T>> StreamingModel<T> for Traced<M> {
    fn invoke_stream(
        &self,
        input: T::Input,
    ) -> BoxFuture<'static, Result<BoxStream<'static, Result<T::Event>>>> {
        let span = span::<T>(self.inner.info());
        let open = span.in_scope(|| self.inner.invoke_stream(input));
        Box::pin(async move {
            let stream = match open.instrument(span.clone()).await {
                Ok(stream) => stream,
                Err(error) => {
                    record_error(&span, &error);
                    return Err(error);
                }
            };
            let traced = stream.map(move |item| {
                let _entered = span.enter();
                match &item {
                    Ok(event) => T::record_event(&span, event),
                    Err(error) => record_error(&span, error),
                }
                item
            });
            Ok(Box::pin(traced) as BoxStream<'static, Result<T::Event>>)
        })
    }
}
