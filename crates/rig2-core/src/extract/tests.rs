use std::collections::VecDeque;
use std::sync::Mutex;

use super::*;
use crate::catalog::{Claim, Evidence};
use crate::completion::{Finish, respond};
use crate::content::{AssistantContent, Text, ToolCall};

#[derive(Debug, serde::Deserialize, schemars::JsonSchema, PartialEq)]
struct City {
    name: String,
}

/// Replies from a script, keeping the requests it saw.
struct Scripted {
    info: ModelInfo,
    card: ModelCard,
    replies: Mutex<VecDeque<AssistantContent>>,
    seen: std::sync::Arc<Mutex<Vec<CompletionRequest>>>,
}

impl Scripted {
    fn new(structured: bool, replies: Vec<AssistantContent>) -> Self {
        let mut card = ModelCard::unknown("m");
        card.capabilities.insert(
            Capability::StructuredOutput,
            Claim {
                supported: structured,
                source: Evidence::Claimed,
                at: None,
            },
        );
        Self {
            info: ModelInfo::new("t", "m"),
            card,
            replies: Mutex::new(replies.into()),
            seen: std::sync::Arc::default(),
        }
    }
}

impl Model<Completion> for Scripted {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn capabilities(&self) -> ModelCard {
        self.card.clone()
    }

    fn invoke(&self, request: CompletionRequest) -> BoxFuture<'static, Result<CompletionResponse>> {
        self.seen.lock().unwrap().push(request);
        let reply = self.replies.lock().unwrap().pop_front().unwrap();
        Box::pin(async move {
            respond(|out| {
                out.part(reply);
                Ok(Finish::default())
            })
        })
    }
}

#[tokio::test]
async fn native_mode_sends_the_schema_and_repairs_a_bad_reply() {
    let model = Scripted::new(
        true,
        vec![
            AssistantContent::Text(Text::new("{\"nom\": 1}")),
            AssistantContent::Text(Text::new("{\"name\": \"Paris\"}")),
        ],
    );
    let seen = model.seen.clone();
    let city = Extractor::<City, _>::new(model)
        .run("capital of France")
        .await
        .unwrap();
    assert_eq!(
        city,
        City {
            name: "Paris".into()
        }
    );
    let seen = seen.lock().unwrap();
    assert!(seen[0].output.is_some());
    assert_eq!(
        seen[1].messages.len(),
        3,
        "the bad reply and the complaint are appended"
    );
}

#[tokio::test]
async fn tool_mode_forces_a_submit_call_and_unwraps_non_object_schemas() {
    let call = ToolCall::new(
        "c1",
        "submit_Array_of_string",
        serde_json::json!({"value": ["a", "b"]}),
    );
    let model = Scripted::new(false, vec![AssistantContent::ToolCall(call)]);
    let seen = model.seen.clone();
    let names = Extractor::<Vec<String>, _>::new(model)
        .run("two letters")
        .await
        .unwrap();
    assert_eq!(names, ["a", "b"]);
    let seen = seen.lock().unwrap();
    assert_eq!(
        seen[0].tool_choice,
        ToolChoice::Tool("submit_Array_of_string".into())
    );
}

#[tokio::test]
async fn repairs_are_bounded() {
    let bad = || AssistantContent::Text(Text::new("nope"));
    let model = Scripted::new(true, vec![bad(), bad()]);
    let seen = model.seen.clone();
    let error = Extractor::<City, _>::new(model)
        .with_max_repairs(1)
        .run("x")
        .await
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidOutput);
    assert_eq!(seen.lock().unwrap().len(), 2);
}
