//! The provider conformance suite: one set of completion cases every
//! provider runs, each against its own cassette.
//!
//! A provider test describes how to build its model ([`Suite::build`]) and
//! which models to use per case ([`Models`]), then calls [`run`]. Each case
//! records its cassette under the suite's fixture directory when live
//! (`RIG2_LIVE=1` and the cassette is missing), and replays it otherwise.
//! A case with no cassette and no live run is [`Outcome::NotTested`].

use std::path::PathBuf;
use std::sync::Arc;

use futures::StreamExt;
use rig2_core::completion::{
    Completion, CompletionRequest, Delta, ReasoningEffort, StreamEvent, ToolDefinition,
    check_canonical,
};
use rig2_core::content::{Image, Message, UserContent};
use rig2_core::extract::Extractor;
use rig2_core::http::SharedClient;
use rig2_core::{ErrorKind, Model, StreamingModel};
use serde::Deserialize;

use crate::cassette::{Cassette, live};

/// A 256x256 solid red PNG.
pub const RED_PNG: &[u8] = include_bytes!("red.png");

/// A completion case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Case {
    /// A plain unary reply.
    PlainText,
    /// A streamed reply, checked to be canonical.
    Streaming,
    /// One tool call.
    ToolCall,
    /// Several tool calls in one reply.
    ParallelTools,
    /// A streamed tool call whose arguments arrive as deltas.
    StreamedArguments,
    /// A reply extracted into a typed value.
    StructuredOutput,
    /// An image in the input.
    ImageInput,
    /// Reasoning tokens or reasoning content.
    Reasoning,
    /// Input and output token counts.
    Usage,
    /// A wrong key is an authentication error.
    BadKey,
    /// A wrong model is a not-found or invalid-request error.
    BadModel,
}

impl Case {
    /// Every case, in report order.
    pub const ALL: [Self; 11] = [
        Self::PlainText,
        Self::Streaming,
        Self::ToolCall,
        Self::ParallelTools,
        Self::StreamedArguments,
        Self::StructuredOutput,
        Self::ImageInput,
        Self::Reasoning,
        Self::Usage,
        Self::BadKey,
        Self::BadModel,
    ];

    /// The case's cassette file name.
    pub fn slug(self) -> &'static str {
        match self {
            Self::PlainText => "plain_text",
            Self::Streaming => "streaming",
            Self::ToolCall => "tool_call",
            Self::ParallelTools => "parallel_tools",
            Self::StreamedArguments => "streamed_arguments",
            Self::StructuredOutput => "structured_output",
            Self::ImageInput => "image_input",
            Self::Reasoning => "reasoning",
            Self::Usage => "usage",
            Self::BadKey => "bad_key",
            Self::BadModel => "bad_model",
        }
    }
}

/// How a case went.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// It passed.
    Pass,
    /// It failed, and why.
    Fail(String),
    /// The provider has no model for it.
    NotSupported,
    /// It was not run, and why.
    NotTested(String),
}

impl Outcome {
    /// The report cell.
    pub fn cell(&self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Fail(_) => "fail",
            Self::NotSupported => "not supported",
            Self::NotTested(_) => "not tested",
        }
    }
}

/// What a provider's builder receives.
#[derive(Clone)]
pub struct Setup {
    /// The HTTP client: a cassette.
    pub http: SharedClient,
    /// The key: the real one when recording, a placeholder when replaying,
    /// and a wrong one for [`Case::BadKey`].
    pub api_key: String,
    /// The model id.
    pub model: String,
}

impl std::fmt::Debug for Setup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Setup")
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

/// Which model each case uses; `None` means the provider cannot do it.
#[derive(Debug, Clone, Default)]
pub struct Models {
    /// Plain text, streaming, usage and the error cases.
    pub text: String,
    /// Tool calls.
    pub tools: Option<String>,
    /// Parallel tool calls.
    pub parallel_tools: Option<String>,
    /// Structured output.
    pub structured: Option<String>,
    /// Image input.
    pub vision: Option<String>,
    /// Reasoning.
    pub reasoning: Option<String>,
}

/// Builds a provider's completion model from a [`Setup`].
pub type Build = dyn Fn(Setup) -> Arc<dyn StreamingModel<Completion>> + Send + Sync;

/// A provider's conformance run.
pub struct Suite {
    /// The directory the case cassettes live in.
    pub fixtures: PathBuf,
    /// The environment variable holding the live key; `None` for local
    /// providers.
    pub key_var: Option<&'static str>,
    /// How to build the model.
    pub build: Box<Build>,
    /// Which model each case uses.
    pub models: Models,
}

fn weather_tool() -> ToolDefinition {
    ToolDefinition {
        name: "get_weather".into(),
        description: "Get the current weather in a city.".into(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": { "city": { "type": "string", "description": "The city name." } },
            "required": ["city"],
            "additionalProperties": false,
        }),
    }
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct Capital {
    city: String,
    country: String,
}

fn check(ok: bool, what: impl Into<String>) -> Outcome {
    if ok {
        Outcome::Pass
    } else {
        Outcome::Fail(what.into())
    }
}

async fn case(model: Arc<dyn StreamingModel<Completion>>, case: Case) -> Outcome {
    let short = |r: CompletionRequest| r.with_max_tokens(512);
    match case {
        Case::PlainText => match model
            .invoke(short("Reply with exactly the word: pong".into()))
            .await
        {
            Ok(r) => check(
                r.text().to_lowercase().contains("pong"),
                format!("unexpected text: {}", r.text()),
            ),
            Err(e) => Outcome::Fail(e.to_string()),
        },
        Case::Usage => match model.invoke(short("Say hi.".into())).await {
            Ok(r) => check(
                r.usage.input_tokens > 0 && r.usage.output_tokens > 0,
                format!("usage: {:?}", r.usage),
            ),
            Err(e) => Outcome::Fail(e.to_string()),
        },
        Case::Streaming => {
            let events: Vec<StreamEvent> = match model
                .invoke_stream(short("Count from 1 to 5.".into()))
                .await
            {
                Ok(stream) => match stream.collect::<Vec<_>>().await.into_iter().collect() {
                    Ok(events) => events,
                    Err(e) => return Outcome::Fail(e.to_string()),
                },
                Err(e) => return Outcome::Fail(e.to_string()),
            };
            if let Err(e) = check_canonical(&events) {
                return Outcome::Fail(e.to_string());
            }
            let deltas = events
                .iter()
                .filter(|e| {
                    matches!(
                        e,
                        StreamEvent::Delta {
                            delta: Delta::Text(_),
                            ..
                        }
                    )
                })
                .count();
            check(deltas >= 1, "no text deltas")
        }
        Case::ToolCall => {
            let request = short(
                CompletionRequest::from("What is the weather in Paris? Use the tool.")
                    .with_tool(weather_tool()),
            );
            match model.invoke(request).await {
                Ok(r) => {
                    let calls = r.tool_calls();
                    check(
                        calls.first().is_some_and(|c| {
                            c.name == "get_weather"
                                && c.arguments
                                    .get("city")
                                    .and_then(serde_json::Value::as_str)
                                    .is_some()
                        }),
                        format!("tool calls: {calls:?}"),
                    )
                }
                Err(e) => Outcome::Fail(e.to_string()),
            }
        }
        Case::ParallelTools => {
            let request = short(
                CompletionRequest::from("Get the weather in Paris and in Tokyo. Call the tool once per city, in the same reply.")
                    .with_tool(weather_tool()),
            );
            match model.invoke(request).await {
                Ok(r) => check(
                    r.tool_calls().len() >= 2,
                    format!("{} tool calls", r.tool_calls().len()),
                ),
                Err(e) => Outcome::Fail(e.to_string()),
            }
        }
        Case::StreamedArguments => {
            let request = short(
                CompletionRequest::from("What is the weather in Paris? Use the tool.")
                    .with_tool(weather_tool()),
            );
            let events: Vec<StreamEvent> = match model.invoke_stream(request).await {
                Ok(stream) => match stream.collect::<Vec<_>>().await.into_iter().collect() {
                    Ok(events) => events,
                    Err(e) => return Outcome::Fail(e.to_string()),
                },
                Err(e) => return Outcome::Fail(e.to_string()),
            };
            if let Err(e) = check_canonical(&events) {
                return Outcome::Fail(e.to_string());
            }
            let arguments = events.iter().any(|e| {
                matches!(
                    e,
                    StreamEvent::Delta {
                        delta: Delta::Arguments(_),
                        ..
                    }
                )
            });
            check(arguments, "no argument deltas")
        }
        Case::StructuredOutput => {
            let request = short(
                "What is the capital of France? Answer with the city and its country.".into(),
            );
            match Extractor::<Capital, _>::new(Arc::clone(&model))
                .invoke(request)
                .await
            {
                Ok(capital) => check(
                    capital.city.contains("Paris") && !capital.country.is_empty(),
                    format!("extracted {capital:?}"),
                ),
                Err(e) => Outcome::Fail(e.to_string()),
            }
        }
        Case::ImageInput => {
            let message = Message::User {
                content: vec![
                    UserContent::Text(rig2_core::content::Text::new(
                        "What single color fills this image? One word.",
                    )),
                    UserContent::Image(Image::from_bytes(RED_PNG, "image/png")),
                ],
            };
            match model.invoke(short(message.into())).await {
                Ok(r) => check(
                    r.text().to_lowercase().contains("red"),
                    format!("unexpected text: {}", r.text()),
                ),
                Err(e) => Outcome::Fail(e.to_string()),
            }
        }
        Case::Reasoning => {
            let request = CompletionRequest::from(
                "Is 391 a prime number? Think it through, then answer yes or no.",
            )
            .with_reasoning(ReasoningEffort::Low)
            .with_max_tokens(4096);
            match model.invoke(request).await {
                Ok(r) => check(
                    r.usage.reasoning_tokens > 0 || !r.reasoning().is_empty(),
                    format!("no reasoning (usage {:?})", r.usage),
                ),
                Err(e) => Outcome::Fail(e.to_string()),
            }
        }
        Case::BadKey => match model.invoke(short("hi".into())).await {
            Ok(_) => Outcome::Fail("a wrong key succeeded".into()),
            Err(e) => check(
                matches!(e.kind(), ErrorKind::Auth | ErrorKind::PermissionDenied),
                format!("expected an auth error, got {:?}: {e}", e.kind()),
            ),
        },
        Case::BadModel => match model.invoke(short("hi".into())).await {
            Ok(_) => Outcome::Fail("a wrong model succeeded".into()),
            Err(e) => check(
                matches!(e.kind(), ErrorKind::NotFound | ErrorKind::InvalidRequest),
                format!("expected not-found, got {:?}: {e}", e.kind()),
            ),
        },
    }
}

/// Run every case and report each outcome.
pub async fn run(suite: &Suite) -> Vec<(Case, Outcome)> {
    let mut results = Vec::new();
    for case_id in Case::ALL {
        let model = match case_id {
            Case::ToolCall | Case::StreamedArguments => suite.models.tools.clone(),
            Case::ParallelTools => suite.models.parallel_tools.clone(),
            Case::StructuredOutput => suite.models.structured.clone(),
            Case::ImageInput => suite.models.vision.clone(),
            Case::Reasoning => suite.models.reasoning.clone(),
            Case::BadModel => Some("rig2-no-such-model".to_owned()),
            Case::PlainText | Case::Streaming | Case::Usage | Case::BadKey => {
                Some(suite.models.text.clone())
            }
        };
        let Some(model) = model else {
            results.push((case_id, Outcome::NotSupported));
            continue;
        };
        let cassette = Cassette::open(
            suite
                .fixtures
                .join(format!("{}.cassette.json", case_id.slug())),
        );
        if !cassette.is_available() {
            results.push((
                case_id,
                Outcome::NotTested("no cassette; record with RIG2_LIVE=1".into()),
            ));
            continue;
        }
        let api_key = if case_id == Case::BadKey {
            "rig2-invalid-key-000000000000".to_owned()
        } else if cassette.is_recording() {
            suite
                .key_var
                .and_then(|v| std::env::var(v).ok())
                .unwrap_or_default()
        } else {
            "rig2-replay-key".to_owned()
        };
        let setup = Setup {
            http: Arc::new(cassette.clone()),
            api_key,
            model,
        };
        let outcome = case((suite.build)(setup), case_id).await;
        if let Err(error) = cassette.finish() {
            results.push((
                case_id,
                Outcome::Fail(format!("could not save the cassette: {error}")),
            ));
            continue;
        }
        results.push((case_id, outcome));
    }
    if live() {
        let table: serde_json::Map<String, serde_json::Value> = results
            .iter()
            .map(|(case, outcome)| {
                (
                    case.slug().to_owned(),
                    serde_json::Value::String(outcome.cell().to_owned()),
                )
            })
            .collect();
        let json = serde_json::to_string_pretty(&table).unwrap_or_default();
        if let Err(error) = std::fs::write(suite.fixtures.join("results.json"), json + "\n") {
            results.push((
                Case::PlainText,
                Outcome::Fail(format!("could not save the results: {error}")),
            ));
        }
    }
    results
}

/// Fail with every failed case. Cases not tested are allowed only when not
/// live, so a live run proves every supported case.
pub fn assert_conformant(results: &[(Case, Outcome)]) {
    let failed: Vec<String> = results
        .iter()
        .filter_map(|(case, outcome)| match outcome {
            Outcome::Fail(why) => Some(format!("{case:?}: {why}")),
            Outcome::NotTested(why) if live() => Some(format!("{case:?}: not tested: {why}")),
            _ => None,
        })
        .collect();
    assert!(
        failed.is_empty(),
        "conformance failures:\n{}",
        failed.join("\n")
    );
}

/// The report row: one cell per case, in [`Case::ALL`] order.
pub fn row(results: &[(Case, Outcome)]) -> Vec<&'static str> {
    Case::ALL
        .iter()
        .map(|case| {
            results
                .iter()
                .find(|(c, _)| c == case)
                .map_or("not tested", |(_, o)| o.cell())
        })
        .collect()
}
