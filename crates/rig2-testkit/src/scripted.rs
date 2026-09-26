use std::collections::VecDeque;
use std::sync::{Arc, Mutex, PoisonError};

use rig2_core::catalog::ModelCard;
use rig2_core::completion::{
    Completion, CompletionRequest, CompletionResponse, Finish, FinishReason, StreamEvent,
    StreamWriter, Usage, collect,
};
use rig2_core::content::{AssistantContent, Text, ToolCall};
use rig2_core::{BoxFuture, BoxStream, Error, ErrorKind, Model, ModelInfo, Result, StreamingModel};

/// One scripted reply.
#[derive(Debug, Clone)]
pub enum Reply {
    /// Stream this content, text in two halves, and finish.
    Content(Vec<AssistantContent>, Usage, Option<f64>),
    /// Fail with this error.
    Fail(Error),
}

impl Reply {
    /// A text reply.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Content(
            vec![AssistantContent::Text(Text::new(text))],
            Usage {
                input_tokens: 10,
                output_tokens: 5,
                ..Usage::default()
            },
            None,
        )
    }

    /// A reply calling one tool.
    pub fn tool_call(id: &str, name: &str, arguments: serde_json::Value) -> Self {
        Self::tool_calls(vec![ToolCall::new(id, name, arguments)])
    }

    /// A reply calling several tools.
    pub fn tool_calls(calls: Vec<ToolCall>) -> Self {
        Self::Content(
            calls.into_iter().map(AssistantContent::ToolCall).collect(),
            Usage {
                input_tokens: 10,
                output_tokens: 5,
                ..Usage::default()
            },
            None,
        )
    }

    /// Set the reply's cost.
    pub fn with_cost(self, cost: f64) -> Self {
        match self {
            Self::Content(content, usage, _) => Self::Content(content, usage, Some(cost)),
            fail @ Self::Fail(_) => fail,
        }
    }
}

/// A completion model that replies from a script.
///
/// Each call takes the next reply; when the script runs out, calls fail with
/// [`ErrorKind::NotFound`]. Replies stream through a real
/// [`StreamWriter`], so their events are canonical. Clones share the script
/// and the log of requests.
#[derive(Debug, Clone)]
pub struct ScriptedModel {
    info: ModelInfo,
    card: ModelCard,
    replies: Arc<Mutex<VecDeque<Reply>>>,
    requests: Arc<Mutex<Vec<CompletionRequest>>>,
}

impl ScriptedModel {
    /// A model replying with `replies` in order.
    pub fn new(replies: impl IntoIterator<Item = Reply>) -> Self {
        Self {
            info: ModelInfo::new("scripted", "scripted"),
            card: ModelCard::unknown("scripted"),
            replies: Arc::new(Mutex::new(replies.into_iter().collect())),
            requests: Arc::default(),
        }
    }

    /// Report `card` as the capabilities.
    pub fn with_card(mut self, card: ModelCard) -> Self {
        self.card = card;
        self
    }

    /// Every request received so far.
    pub fn requests(&self) -> Vec<CompletionRequest> {
        self.requests
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn next_events(&self, request: CompletionRequest) -> Result<Vec<StreamEvent>> {
        self.requests
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(request);
        let reply = self
            .replies
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop_front()
            .ok_or_else(|| Error::new(ErrorKind::NotFound, "the script has no more replies"))?;
        let (content, usage, cost) = match reply {
            Reply::Content(content, usage, cost) => (content, usage, cost),
            Reply::Fail(error) => return Err(error),
        };
        let has_calls = content
            .iter()
            .any(|c| matches!(c, AssistantContent::ToolCall(_)));
        let mut out = StreamWriter::default();
        for part in content {
            match part {
                AssistantContent::Text(text) => {
                    let block = out.text();
                    let mid = text
                        .text
                        .char_indices()
                        .nth(text.text.chars().count() / 2)
                        .map_or(0, |(i, _)| i);
                    let (head, tail) = text.text.split_at_checked(mid).unwrap_or((&text.text, ""));
                    out.push(&block, head);
                    out.push(&block, tail);
                    out.end_text(block);
                }
                AssistantContent::ToolCall(call) => {
                    let block = out.tool_call(call.id, call.name);
                    out.push(&block, &call.arguments.to_string());
                    out.end_tool_call(block)?;
                }
                other => out.part(other),
            }
        }
        let finish_reason = if has_calls {
            FinishReason::ToolCalls
        } else {
            FinishReason::Stop
        };
        out.finish(Finish {
            usage,
            finish_reason,
            cost,
            ..Finish::default()
        })
    }
}

impl Model<Completion> for ScriptedModel {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn capabilities(&self) -> ModelCard {
        self.card.clone()
    }

    fn invoke(&self, request: CompletionRequest) -> BoxFuture<'static, Result<CompletionResponse>> {
        let events = self.next_events(request);
        Box::pin(async move {
            collect(Box::pin(futures::stream::iter(events?.into_iter().map(Ok)))).await
        })
    }
}

impl StreamingModel<Completion> for ScriptedModel {
    fn invoke_stream(
        &self,
        request: CompletionRequest,
    ) -> BoxFuture<'static, Result<BoxStream<'static, Result<StreamEvent>>>> {
        let events = self.next_events(request);
        Box::pin(async move {
            Ok(Box::pin(futures::stream::iter(events?.into_iter().map(Ok)))
                as BoxStream<'static, Result<StreamEvent>>)
        })
    }
}
