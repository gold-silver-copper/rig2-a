use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::StreamExt;
use rig2_core::content::ToolOutput;
use rig2_core::store::{InMemoryStore, Record, VectorStore};
use rig2_core::tasks::{Embedding, EmbeddingRequest, EmbeddingResponse};
use rig2_core::{Model, ModelInfo};
use rig2_testkit::{Reply, ScriptedModel};

use super::*;
use crate::memory::InMemory;
use crate::retrieval::Retriever;

/// Add two numbers.
#[crate::tool]
fn add(a: i64, b: i64) -> i64 {
    a + b
}

struct Delete;

impl Tool for Delete {
    fn definition(&self) -> rig2_core::completion::ToolDefinition {
        rig2_core::completion::ToolDefinition {
            name: "delete".into(),
            description: "Delete everything.".into(),
            parameters: serde_json::json!({"type": "object"}),
        }
    }

    fn call(
        &self,
        _: serde_json::Value,
        _: ToolContext,
    ) -> BoxFuture<'static, Result<Vec<ToolOutput>>> {
        Box::pin(async { Ok(vec![ToolOutput::Text("deleted".into())]) })
    }

    fn requires_approval(&self) -> bool {
        true
    }
}

#[tokio::test]
async fn prompt_runs_the_tool_loop_and_returns_the_final_text() {
    let model = ScriptedModel::new([
        Reply::tool_calls(vec![
            ToolCall::new("1", "add", serde_json::json!({"a": 2, "b": 3})),
            ToolCall::new("2", "add", serde_json::json!({"a": 10, "b": 1})),
        ]),
        Reply::text("5 and 11"),
    ]);
    let agent = Agent::builder(model.clone())
        .preamble("math")
        .tool(Add)
        .build();
    assert_eq!(agent.prompt("add things").await.unwrap(), "5 and 11");
    let second = &model.requests()[1];
    assert_eq!(second.system.as_deref(), Some("math"));
    assert_eq!(second.tools[0].name, "add");
    let rig2_core::content::Message::User { content } = second.messages.last().unwrap() else {
        panic!()
    };
    assert_eq!(
        content.len(),
        2,
        "both parallel results come back in one turn"
    );
}

#[tokio::test]
async fn an_unknown_tool_is_reported_to_the_model_not_raised() {
    let model = ScriptedModel::new([
        Reply::tool_call("1", "nope", serde_json::json!({})),
        Reply::text("sorry"),
    ]);
    let agent = Agent::builder(model.clone()).build();
    assert_eq!(agent.prompt("x").await.unwrap(), "sorry");
    let last = model.requests()[1].messages.last().unwrap().clone();
    assert!(last.text().is_empty());
    assert!(
        serde_json::to_string(&last)
            .unwrap()
            .contains("no tool named")
    );
}

#[tokio::test]
async fn a_model_failure_ends_the_run_with_the_error() {
    let model = ScriptedModel::new([Reply::Fail(Error::new(ErrorKind::Auth, "bad key"))]);
    let error = Agent::builder(model).build().prompt("x").await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Auth);
}

#[tokio::test]
async fn stream_yields_model_deltas_tool_events_and_the_outcome_in_order() {
    let model = ScriptedModel::new([
        Reply::tool_call("1", "add", serde_json::json!({"a": 1, "b": 1})),
        Reply::text("two"),
    ]);
    let agent = Agent::builder(model).tool(Add).build();
    let events: Vec<AgentEvent> = agent.stream("1+1").map(Result::unwrap).collect().await;
    let kinds: Vec<&str> = events
        .iter()
        .map(|e| match e {
            AgentEvent::Model(_) => "model",
            AgentEvent::Reply(_) => "reply",
            AgentEvent::ToolCall(_) => "call",
            AgentEvent::ToolResult(_) => "result",
            AgentEvent::NeedsApproval(_) => "approval",
            AgentEvent::Done(_) => "done",
        })
        .collect();
    let first_call = kinds.iter().position(|k| *k == "call").unwrap();
    assert!(kinds[..first_call].contains(&"model"));
    assert_eq!(kinds.last(), Some(&"done"));
    assert!(matches!(
        events.last(),
        Some(AgentEvent::Done(Outcome::Answered { .. }))
    ));
}

struct Stopper;

impl Hook for Stopper {
    fn before_tool(&self, call: &ToolCall) -> BoxFuture<'static, Result<Control<()>>> {
        let veto = call.name == "add";
        Box::pin(async move {
            Ok(if veto {
                Control::Stop("no adding".into())
            } else {
                Control::Continue(())
            })
        })
    }

    fn before_model(
        &self,
        mut request: CompletionRequest,
    ) -> BoxFuture<'static, Result<Control<CompletionRequest>>> {
        request.temperature = Some(0.0);
        Box::pin(async move { Ok(Control::Continue(request)) })
    }
}

#[tokio::test]
async fn hooks_rewrite_requests_and_veto_tool_calls() {
    let model = ScriptedModel::new([
        Reply::tool_call("1", "add", serde_json::json!({"a": 1, "b": 1})),
        Reply::text("ok"),
    ]);
    let agent = Agent::builder(model.clone())
        .tool(Add)
        .hook(Stopper)
        .build();
    agent.prompt("x").await.unwrap();
    let requests = model.requests();
    assert_eq!(requests[0].temperature, Some(0.0));
    assert!(
        serde_json::to_string(&requests[1].messages)
            .unwrap()
            .contains("vetoed: no adding")
    );
}

#[tokio::test]
async fn a_run_waiting_for_approval_pauses_and_resumes_after_a_decision() {
    let model = ScriptedModel::new([
        Reply::tool_call("1", "delete", serde_json::json!({})),
        Reply::text("deleted it"),
    ]);
    let agent = Agent::builder(model).tool(Delete).build();
    let run = agent
        .drive(agent.start(Vec::new(), "clean up"), |_| {})
        .await
        .unwrap();
    assert!(matches!(run.next(), Step::Approve(_)));
    let mut saved: AgentRun = serde_json::from_str(&serde_json::to_string(&run).unwrap()).unwrap();
    saved.decide(&[("1".into(), Decision::Approve)]).unwrap();
    let done = agent.drive(saved, |_| {}).await.unwrap();
    let Step::Done(outcome) = done.next() else {
        panic!()
    };
    assert_eq!(outcome.into_response().unwrap().text(), "deleted it");
    assert_eq!(
        agent.prompt("again").await.unwrap_err().kind(),
        ErrorKind::NotFound
    );
}

struct Always;

impl Approver for Always {
    fn decide(&self, calls: Vec<ToolCall>) -> BoxFuture<'static, Result<Vec<(String, Decision)>>> {
        Box::pin(async move {
            Ok(calls
                .into_iter()
                .map(|c| (c.id, Decision::Approve))
                .collect())
        })
    }
}

#[tokio::test]
async fn an_approver_decides_without_pausing() {
    let model = ScriptedModel::new([
        Reply::tool_call("1", "delete", serde_json::json!({})),
        Reply::text("done"),
    ]);
    let agent = Agent::builder(model).tool(Delete).approver(Always).build();
    assert_eq!(agent.prompt("x").await.unwrap(), "done");
}

#[tokio::test]
async fn memory_carries_the_conversation_across_prompts() {
    let memory = InMemory::default();
    let model = ScriptedModel::new([Reply::text("hi Ada"), Reply::text("your name is Ada")]);
    let agent = Agent::builder(model.clone())
        .memory(memory.clone(), "chat-1")
        .build();
    agent.prompt("I am Ada").await.unwrap();
    agent.prompt("who am I?").await.unwrap();
    assert_eq!(model.requests()[1].messages.len(), 3);
    assert_eq!(memory.load("chat-1").await.unwrap().len(), 4);
}

/// Embeds by counting letters `a` and `b`.
struct Letters(ModelInfo, Arc<AtomicUsize>);

impl Model<Embedding> for Letters {
    fn info(&self) -> &ModelInfo {
        &self.0
    }

    fn capabilities(&self) -> ModelCard {
        ModelCard::unknown("letters")
    }

    fn invoke(&self, request: EmbeddingRequest) -> BoxFuture<'static, Result<EmbeddingResponse>> {
        self.1.fetch_add(1, Ordering::SeqCst);
        let embeddings = request
            .texts
            .iter()
            .map(|t| vec![t.matches('a').count() as f32, t.matches('b').count() as f32])
            .collect();
        Box::pin(async move {
            Ok(EmbeddingResponse {
                embeddings,
                ..EmbeddingResponse::default()
            })
        })
    }
}

#[tokio::test]
async fn retrieval_injects_the_best_match_as_dynamic_context() {
    let store = Arc::new(InMemoryStore::default());
    store
        .upsert(vec![
            Record::new(
                "aaa",
                vec![3.0, 0.0],
                serde_json::json!({"text": "about a"}),
            ),
            Record::new(
                "bbb",
                vec![0.0, 3.0],
                serde_json::json!({"text": "about b"}),
            ),
        ])
        .await
        .unwrap();
    let embedder: Arc<dyn Model<Embedding>> =
        Arc::new(Letters(ModelInfo::new("t", "letters"), Arc::default()));
    let retriever = Retriever::new(embedder, store).with_top_k(1);
    let model = ScriptedModel::new([Reply::text("ok")]);
    let agent = Agent::builder(model.clone())
        .preamble("base")
        .hook(retriever.clone())
        .tool(retriever)
        .build();
    agent.prompt("tell me about bb").await.unwrap();
    let system = model.requests()[0].system.clone().unwrap();
    assert!(system.starts_with("base"));
    assert!(
        system.contains("about b") && !system.contains("about a"),
        "{system}"
    );
}
