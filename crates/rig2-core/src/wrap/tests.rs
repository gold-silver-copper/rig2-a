use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::StreamExt;

use super::*;
use crate::{
    BoxFuture, BoxStream, Error, ErrorKind, Model, ModelInfo, Result, StreamingModel,
    StreamingTask, Task,
};

struct Echo;

impl Task for Echo {
    const NAME: &'static str = "echo";
    type Input = String;
    type Output = String;
    type Capabilities = ();
}

impl StreamingTask for Echo {
    type Event = String;
}

/// Answers from a script of results, counting calls.
struct Scripted {
    info: ModelInfo,
    script: Mutex<VecDeque<Result<String>>>,
    calls: Arc<AtomicUsize>,
}

impl Scripted {
    fn new(script: Vec<Result<String>>) -> (Self, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let model = Self {
            info: ModelInfo::new("test", "scripted"),
            script: Mutex::new(script.into()),
            calls: calls.clone(),
        };
        (model, calls)
    }
}

impl Model<Echo> for Scripted {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn capabilities(&self) {}

    fn invoke(&self, input: String) -> BoxFuture<'static, Result<String>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let next = self
            .script
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| Ok(input));
        Box::pin(async move { next })
    }
}

impl StreamingModel<Echo> for Scripted {
    fn invoke_stream(
        &self,
        input: String,
    ) -> BoxFuture<'static, Result<BoxStream<'static, Result<String>>>> {
        let words: Vec<Result<String>> = input.split(' ').map(|w| Ok(w.to_owned())).collect();
        Box::pin(async move {
            Ok(Box::pin(futures::stream::iter(words)) as BoxStream<'static, Result<String>>)
        })
    }
}

fn unavailable() -> Error {
    Error::new(ErrorKind::Unavailable, "overloaded")
}

#[tokio::test]
async fn retry_retries_retryable_errors_and_stops_at_the_first_success() {
    let (model, calls) = Scripted::new(vec![
        Err(unavailable()),
        Err(unavailable()),
        Ok("done".into()),
    ]);
    let model = Retry::new(model)
        .with_attempts(5)
        .with_base_delay(Duration::from_millis(1));
    assert_eq!(model.run("x").await.unwrap(), "done");
    assert_eq!(calls.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn retry_does_not_retry_a_client_error() {
    let (model, calls) = Scripted::new(vec![Err(Error::new(ErrorKind::InvalidRequest, "bad"))]);
    let model = Retry::new(model).with_base_delay(Duration::from_millis(1));
    assert_eq!(
        model.run("x").await.unwrap_err().kind(),
        ErrorKind::InvalidRequest
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn retry_honours_retry_after_up_to_the_cap() {
    let (model, _) = Scripted::new(vec![Err(
        unavailable().with_retry_after(Duration::from_secs(60))
    )]);
    let model = Retry::new(model).with_max_delay(Duration::from_millis(5));
    let started = std::time::Instant::now();
    model.run("x").await.unwrap();
    assert!(started.elapsed() < Duration::from_secs(5));
}

#[tokio::test]
async fn fallback_moves_on_until_a_model_succeeds() {
    let (first, first_calls) = Scripted::new(vec![Err(Error::new(ErrorKind::NotFound, "gone"))]);
    let (second, _) = Scripted::new(vec![Ok("second".into())]);
    let model = Fallback::new(vec![
        Arc::new(first) as Arc<dyn Model<Echo>>,
        Arc::new(second),
    ])
    .unwrap();
    assert_eq!(model.run("x").await.unwrap(), "second");
    assert_eq!(first_calls.load(Ordering::SeqCst), 1);
    assert!(Fallback::<Arc<dyn Model<Echo>>>::new(vec![]).is_err());
}

#[tokio::test]
async fn cached_answers_a_repeated_input_without_a_call_and_evicts_the_oldest() {
    let (model, calls) = Scripted::new(vec![]);
    let model: Cached<_, String> = Cached::new(model, 1);
    assert_eq!(model.run("a").await.unwrap(), "a");
    assert_eq!(model.run("a").await.unwrap(), "a");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    model.run("b").await.unwrap();
    model.run("a").await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn rate_limited_spaces_calls_beyond_the_rate() {
    let (model, _) = Scripted::new(vec![]);
    let model = Arc::new(RateLimited::new(model, 1, Duration::from_millis(50), 4));
    let started = std::time::Instant::now();
    let calls: Vec<_> = (0..3).map(|i| model.run(i.to_string())).collect();
    futures::future::join_all(calls).await;
    assert!(
        started.elapsed() >= Duration::from_millis(90),
        "{:?}",
        started.elapsed()
    );
}

#[tokio::test]
async fn a_recording_replays_unary_and_streamed_calls_without_the_model() {
    let recorder = Recorder::default();
    let (model, _) = Scripted::new(vec![
        Ok("one".into()),
        Err(Error::new(ErrorKind::RateLimited, "later")),
    ]);
    let recorded = Recorded::new(Traced::new(model), recorder.clone());
    assert_eq!(recorded.run("a").await.unwrap(), "one");
    assert_eq!(
        recorded.run("b").await.unwrap_err().kind(),
        ErrorKind::RateLimited
    );
    let words: Vec<String> = recorded
        .stream("x y")
        .await
        .unwrap()
        .map(Result::unwrap)
        .collect()
        .await;
    assert_eq!(words, ["x", "y"]);

    let json = serde_json::to_string(&recorder.snapshot()).unwrap();
    let replay = Replay::new(
        ModelInfo::new("test", "scripted"),
        serde_json::from_str(&json).unwrap(),
    );
    assert_eq!(Model::<Echo>::run(&replay, "a").await.unwrap(), "one");
    assert_eq!(
        Model::<Echo>::run(&replay, "b").await.unwrap_err().kind(),
        ErrorKind::RateLimited
    );
    let replayed: Vec<String> = StreamingModel::<Echo>::stream(&replay, "x y")
        .await
        .unwrap()
        .map(Result::unwrap)
        .collect()
        .await;
    assert_eq!(replayed, ["x", "y"]);
    assert_eq!(replay.remaining(), 0);
    assert_eq!(
        Model::<Echo>::run(&replay, "a").await.unwrap_err().kind(),
        ErrorKind::NotFound
    );
}

#[tokio::test]
async fn wrappers_compose_on_an_erased_model() {
    let (model, _) = Scripted::new(vec![Err(unavailable())]);
    let erased: Arc<dyn Model<Echo>> = Arc::new(model);
    let wrapped = Traced::new(
        Retry::new(RateLimited::new(erased, 10, Duration::from_secs(1), 2))
            .with_base_delay(Duration::from_millis(1)),
    );
    assert_eq!(wrapped.run("ok").await.unwrap(), "ok");
}
