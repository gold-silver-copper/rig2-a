use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::catalog::Pricing;
use crate::content::{AssistantContent, Message, Reasoning, ToolCall, join_text};
use crate::{Error, ErrorKind, Result};

/// A finished completion.
///
/// `raw` is the provider's own reply document. It has the same shape whether
/// the reply was streamed or not: a streaming decoder assembles it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CompletionResponse {
    /// What the model produced, in order.
    pub content: Vec<AssistantContent>,
    /// Token counts.
    pub usage: Usage,
    /// Why generation stopped.
    pub finish_reason: FinishReason,
    /// The provider's response id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// The model that answered, as the provider reports it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// The cost in US dollars, when the catalog has the model's pricing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<f64>,
    /// The provider's reply document.
    #[serde(default)]
    pub raw: serde_json::Value,
}

impl CompletionResponse {
    /// The text parts, joined.
    pub fn text(&self) -> String {
        join_text(self.content.iter().filter_map(|c| match c {
            AssistantContent::Text(t) => Some(t.text.as_str()),
            _ => None,
        }))
    }

    /// The tool calls, in order.
    pub fn tool_calls(&self) -> Vec<&ToolCall> {
        self.content
            .iter()
            .filter_map(|c| match c {
                AssistantContent::ToolCall(call) => Some(call),
                _ => None,
            })
            .collect()
    }

    /// The reasoning parts, in order.
    pub fn reasoning(&self) -> Vec<&Reasoning> {
        self.content
            .iter()
            .filter_map(|c| match c {
                AssistantContent::Reasoning(r) => Some(r),
                _ => None,
            })
            .collect()
    }

    /// Parse the text as JSON into `T`.
    ///
    /// A Markdown code fence around the JSON is tolerated. Fails with
    /// [`ErrorKind::InvalidOutput`] when the text is not a valid `T`.
    pub fn parse<T: DeserializeOwned>(&self) -> Result<T> {
        let text = self.text();
        let body = strip_fence(&text);
        serde_json::from_str(body).map_err(|e| {
            Error::new(
                ErrorKind::InvalidOutput,
                format!("reply is not the requested type: {e}"),
            )
        })
    }

    /// The reply as an assistant message, to append to a conversation.
    pub fn message(&self) -> Message {
        Message::Assistant {
            content: self.content.clone(),
        }
    }
}

fn strip_fence(text: &str) -> &str {
    let trimmed = text.trim();
    let Some(rest) = trimmed.strip_prefix("```") else {
        return trimmed;
    };
    let rest = rest.split_once('\n').map_or(rest, |(_, body)| body);
    rest.strip_suffix("```").unwrap_or(rest).trim()
}

/// Token counts for one call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Usage {
    /// Input tokens, including cached ones.
    pub input_tokens: u64,
    /// Output tokens, including reasoning.
    pub output_tokens: u64,
    /// Input tokens served from the provider's cache.
    #[serde(default)]
    pub cached_input_tokens: u64,
    /// Input tokens written to the provider's cache.
    #[serde(default)]
    pub cache_write_tokens: u64,
    /// Output tokens spent on reasoning.
    #[serde(default)]
    pub reasoning_tokens: u64,
}

impl Usage {
    /// Input plus output tokens.
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }

    /// The cost in US dollars at `pricing`.
    pub fn cost(&self, pricing: &Pricing) -> f64 {
        let uncached = self
            .input_tokens
            .saturating_sub(self.cached_input_tokens + self.cache_write_tokens);
        let per = |tokens: u64, price: f64| tokens as f64 * price / 1_000_000.0;
        per(uncached, pricing.input)
            + per(
                self.cached_input_tokens,
                pricing.cached_input.unwrap_or(pricing.input),
            )
            + per(
                self.cache_write_tokens,
                pricing.cache_write.unwrap_or(pricing.input),
            )
            + per(self.output_tokens, pricing.output)
    }
}

impl std::ops::AddAssign for Usage {
    fn add_assign(&mut self, other: Self) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.cached_input_tokens += other.cached_input_tokens;
        self.cache_write_tokens += other.cache_write_tokens;
        self.reasoning_tokens += other.reasoning_tokens;
    }
}

/// Why generation stopped.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    /// The model finished its reply.
    #[default]
    Stop,
    /// The output token limit was reached.
    Length,
    /// The model is waiting for tool results.
    ToolCalls,
    /// The provider filtered the output.
    ContentFilter,
    /// The model refused.
    Refusal,
    /// A reason with no portable equivalent, as the provider named it.
    Other(String),
}
