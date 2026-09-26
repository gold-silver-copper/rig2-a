use std::sync::{Arc, Mutex, PoisonError};

use futures::StreamExt;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::{
    BoxFuture, BoxStream, Error, ErrorKind, Model, ModelInfo, Result, StreamingModel,
    StreamingTask, Task,
};

/// Recorded calls: plain data, to save as JSON and serve with [`Replay`].
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Recording {
    /// The calls, in the order they finished.
    pub calls: Vec<RecordedCall>,
}

/// One recorded call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecordedCall {
    /// The task's name.
    pub task: String,
    /// The model that answered.
    pub model: ModelInfo,
    /// The input, as JSON.
    pub input: serde_json::Value,
    /// What came back.
    pub outcome: RecordedOutcome,
}

/// What a recorded call returned.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordedOutcome {
    /// A unary call's output, as JSON.
    Output(serde_json::Value),
    /// A stream's events, as JSON, and the error that ended it, if any.
    Stream {
        /// The events, in order.
        events: Vec<serde_json::Value>,
        /// The error that ended the stream, if any.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<Error>,
    },
    /// The call failed.
    Error(Error),
}

/// A shared handle to a [`Recording`] that [`Recorded`] models append to.
#[derive(Debug, Clone, Default)]
pub struct Recorder(Arc<Mutex<Recording>>);

impl Recorder {
    /// A copy of everything recorded so far.
    pub fn snapshot(&self) -> Recording {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn push(&self, call: RecordedCall) {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .calls
            .push(call);
    }
}

/// Records every input, output and stream event of a model.
///
/// Works for any task whose input, output and events serialize, including
/// local models. Serve the recording back with [`Replay`].
#[derive(Debug, Clone)]
pub struct Recorded<M> {
    inner: M,
    recorder: Recorder,
}

impl<M> Recorded<M> {
    /// Record `inner`'s calls into `recorder`.
    pub fn new(inner: M, recorder: Recorder) -> Self {
        Self { inner, recorder }
    }
}

impl<T, M> Model<T> for Recorded<M>
where
    T: Task,
    T::Input: Serialize,
    T::Output: Serialize,
    M: Model<T>,
{
    fn info(&self) -> &ModelInfo {
        self.inner.info()
    }

    fn capabilities(&self) -> T::Capabilities {
        self.inner.capabilities()
    }

    fn invoke(&self, input: T::Input) -> BoxFuture<'static, Result<T::Output>> {
        let (task, model, recorder) = (
            T::NAME.to_owned(),
            self.inner.info().clone(),
            self.recorder.clone(),
        );
        let input_json = match serde_json::to_value(&input) {
            Ok(json) => json,
            Err(error) => return Box::pin(std::future::ready(Err(error.into()))),
        };
        let call = self.inner.invoke(input);
        Box::pin(async move {
            let result = call.await;
            let outcome = match &result {
                Ok(output) => RecordedOutcome::Output(serde_json::to_value(output)?),
                Err(error) => RecordedOutcome::Error(error.clone()),
            };
            recorder.push(RecordedCall {
                task,
                model,
                input: input_json,
                outcome,
            });
            result
        })
    }
}

impl<T, M> StreamingModel<T> for Recorded<M>
where
    T: StreamingTask,
    T::Input: Serialize,
    T::Output: Serialize,
    T::Event: Serialize,
    M: StreamingModel<T>,
{
    fn invoke_stream(
        &self,
        input: T::Input,
    ) -> BoxFuture<'static, Result<BoxStream<'static, Result<T::Event>>>> {
        let (task, model, recorder) = (
            T::NAME.to_owned(),
            self.inner.info().clone(),
            self.recorder.clone(),
        );
        let input_json = match serde_json::to_value(&input) {
            Ok(json) => json,
            Err(error) => return Box::pin(std::future::ready(Err(error.into()))),
        };
        let open = self.inner.invoke_stream(input);
        Box::pin(async move {
            let stream = match open.await {
                Ok(stream) => stream,
                Err(error) => {
                    let outcome = RecordedOutcome::Error(error.clone());
                    recorder.push(RecordedCall {
                        task,
                        model,
                        input: input_json,
                        outcome,
                    });
                    return Err(error);
                }
            };
            let state = Tape {
                call: Some((task, model, input_json, recorder)),
                events: Vec::new(),
            };
            let taped =
                futures::stream::unfold((stream, state), |(mut stream, mut tape)| async move {
                    match stream.next().await {
                        Some(Ok(event)) => {
                            match serde_json::to_value(&event) {
                                Ok(json) => tape.events.push(json),
                                Err(error) => {
                                    tape.close(Some(error.into()));
                                }
                            }
                            Some((Ok(event), (stream, tape)))
                        }
                        Some(Err(error)) => {
                            tape.close(Some(error.clone()));
                            Some((Err(error), (stream, tape)))
                        }
                        None => {
                            tape.close(None);
                            None
                        }
                    }
                });
            Ok(Box::pin(taped) as BoxStream<'static, Result<T::Event>>)
        })
    }
}

struct Tape {
    call: Option<(String, ModelInfo, serde_json::Value, Recorder)>,
    events: Vec<serde_json::Value>,
}

impl Tape {
    fn close(&mut self, error: Option<Error>) {
        if let Some((task, model, input, recorder)) = self.call.take() {
            let events = std::mem::take(&mut self.events);
            recorder.push(RecordedCall {
                task,
                model,
                input,
                outcome: RecordedOutcome::Stream { events, error },
            });
        }
    }
}

/// Serves a [`Recording`] as a model: the durable, deterministic stand-in
/// for any recorded model.
///
/// A call is answered by the first unused recorded call of the same task
/// with an equal input. When none matches, the call fails with
/// [`ErrorKind::NotFound`].
#[derive(Debug, Clone)]
pub struct Replay {
    info: ModelInfo,
    calls: Arc<Mutex<Vec<Option<RecordedCall>>>>,
}

impl Replay {
    /// Serve `recording`, reporting `info` as the model.
    pub fn new(info: ModelInfo, recording: Recording) -> Self {
        Self {
            info,
            calls: Arc::new(Mutex::new(recording.calls.into_iter().map(Some).collect())),
        }
    }

    fn take(&self, task: &str, input: &serde_json::Value) -> Result<RecordedOutcome> {
        let mut calls = self.calls.lock().unwrap_or_else(PoisonError::into_inner);
        calls
            .iter_mut()
            .find(|slot| {
                slot.as_ref()
                    .is_some_and(|c| c.task == task && &c.input == input)
            })
            .and_then(Option::take)
            .map(|call| call.outcome)
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::NotFound,
                    format!("no recorded `{task}` call matches this input"),
                )
            })
    }

    /// How many recorded calls have not been served.
    pub fn remaining(&self) -> usize {
        self.calls
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .filter(|c| c.is_some())
            .count()
    }
}

impl<T> Model<T> for Replay
where
    T: Task,
    T::Input: Serialize,
    T::Output: DeserializeOwned,
    T::Capabilities: Default,
{
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn capabilities(&self) -> T::Capabilities {
        T::Capabilities::default()
    }

    fn invoke(&self, input: T::Input) -> BoxFuture<'static, Result<T::Output>> {
        let outcome = serde_json::to_value(&input)
            .map_err(Error::from)
            .and_then(|input| self.take(T::NAME, &input));
        Box::pin(std::future::ready(outcome.and_then(
            |outcome| match outcome {
                RecordedOutcome::Output(output) => Ok(serde_json::from_value(output)?),
                RecordedOutcome::Error(error) => Err(error),
                RecordedOutcome::Stream { .. } => Err(Error::new(
                    ErrorKind::NotFound,
                    "the recorded call was a stream, not a unary call",
                )),
            },
        )))
    }
}

impl<T> StreamingModel<T> for Replay
where
    T: StreamingTask,
    T::Input: Serialize,
    T::Output: DeserializeOwned,
    T::Event: DeserializeOwned,
    T::Capabilities: Default,
{
    fn invoke_stream(
        &self,
        input: T::Input,
    ) -> BoxFuture<'static, Result<BoxStream<'static, Result<T::Event>>>> {
        let outcome = serde_json::to_value(&input)
            .map_err(Error::from)
            .and_then(|input| self.take(T::NAME, &input));
        Box::pin(std::future::ready(outcome.and_then(
            |outcome| match outcome {
                RecordedOutcome::Stream { events, error } => {
                    let items: Vec<Result<T::Event>> = events
                        .into_iter()
                        .map(|e| serde_json::from_value(e).map_err(Error::from))
                        .chain(error.map(Err))
                        .collect();
                    Ok(Box::pin(futures::stream::iter(items))
                        as BoxStream<'static, Result<T::Event>>)
                }
                RecordedOutcome::Error(error) => Err(error),
                RecordedOutcome::Output(_) => Err(Error::new(
                    ErrorKind::NotFound,
                    "the recorded call was unary, not a stream",
                )),
            },
        )))
    }
}
