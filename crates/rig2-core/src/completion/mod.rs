//! Completion: the chat task, its request and response, and its stream.
//!
//! Providers write replies through a [`StreamWriter`], whose API cannot
//! produce an ill-formed stream. Streaming replies go through
//! [`decode_stream`], unary ones through [`respond`]; [`Collect`] turns
//! either into a [`CompletionResponse`].

mod request;
mod response;
mod stream;

pub use request::{
    CacheHint, CompletionRequest, OutputSchema, ReasoningEffort, ToolChoice, ToolDefinition,
};
pub use response::{CompletionResponse, FinishReason, Usage};
pub use stream::{
    BlockKind, Collect, Delta, Finish, OpenBlock, ReasoningBlock, StreamDecoder, StreamEvent,
    StreamWriter, TextBlock, ToolCallBlock, check_canonical, collect, decode_stream, respond,
};

use crate::catalog::ModelCard;
use crate::{StreamingTask, Task};

/// Generate a reply to a conversation.
#[derive(Debug, Clone, Copy)]
pub struct Completion;

impl Task for Completion {
    const NAME: &'static str = "chat";
    type Input = CompletionRequest;
    type Output = CompletionResponse;
    type Capabilities = ModelCard;

    fn record(span: &tracing::Span, output: &CompletionResponse) {
        span.record("gen_ai.usage.input_tokens", output.usage.input_tokens);
        span.record("gen_ai.usage.output_tokens", output.usage.output_tokens);
        span.record(
            "gen_ai.response.finish_reasons",
            tracing::field::debug(&output.finish_reason),
        );
        if let Some(id) = &output.id {
            span.record("gen_ai.response.id", id.as_str());
        }
        if let Some(model) = &output.model {
            span.record("gen_ai.response.model", model.as_str());
        }
        if let Some(cost) = output.cost {
            span.record("rig2.cost_usd", cost);
        }
    }
}

impl StreamingTask for Completion {
    type Event = StreamEvent;

    fn record_event(span: &tracing::Span, event: &StreamEvent) {
        if let StreamEvent::Finish(finish) = event {
            span.record("gen_ai.usage.input_tokens", finish.usage.input_tokens);
            span.record("gen_ai.usage.output_tokens", finish.usage.output_tokens);
            span.record(
                "gen_ai.response.finish_reasons",
                tracing::field::debug(&finish.finish_reason),
            );
            if let Some(id) = &finish.id {
                span.record("gen_ai.response.id", id.as_str());
            }
            if let Some(cost) = finish.cost {
                span.record("rig2.cost_usd", cost);
            }
        }
    }
}

#[cfg(test)]
mod tests;
