use serde::{Deserialize, Serialize};

use crate::content::{Extensions, Message};

/// Everything a completion call sends, in portable form.
///
/// Fields a provider does not support are either mapped to its nearest
/// equivalent or rejected with [`ErrorKind::Unsupported`](crate::ErrorKind),
/// never silently dropped. Provider-only options go in `extensions`.
///
/// ```
/// use rig2_core::completion::CompletionRequest;
///
/// let request = CompletionRequest::from("Name three primes.").with_system("Be terse.").with_max_tokens(64);
/// assert_eq!(request.messages.len(), 1);
/// ```
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CompletionRequest {
    /// Instructions that frame the conversation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// The conversation so far, oldest first.
    pub messages: Vec<Message>,
    /// Tools the model may call.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    /// Whether and which tool the model must call.
    #[serde(default, skip_serializing_if = "ToolChoice::is_auto")]
    pub tool_choice: ToolChoice,
    /// A JSON schema the reply must follow.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<OutputSchema>,
    /// The most tokens to generate, reasoning included where the provider
    /// counts it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Sampling temperature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Nucleus sampling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Sequences that end generation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
    /// How hard a reasoning model should think.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningEffort>,
    /// Whether to ask the provider to cache the request prefix.
    #[serde(default, skip_serializing_if = "CacheHint::is_none")]
    pub cache: CacheHint,
    /// Provider-specific options.
    #[serde(default, skip_serializing_if = "Extensions::is_empty")]
    pub extensions: Extensions,
}

impl CompletionRequest {
    /// A request with these messages and nothing else set.
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            ..Self::default()
        }
    }

    /// Set the system prompt.
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Add a tool.
    pub fn with_tool(mut self, tool: ToolDefinition) -> Self {
        self.tools.push(tool);
        self
    }

    /// Set the tool choice.
    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = choice;
        self
    }

    /// Require the reply to follow a JSON schema.
    pub fn with_output(mut self, output: OutputSchema) -> Self {
        self.output = Some(output);
        self
    }

    /// Set the output token limit.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set the sampling temperature.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set the reasoning effort.
    pub fn with_reasoning(mut self, effort: ReasoningEffort) -> Self {
        self.reasoning = Some(effort);
        self
    }

    /// Set the cache hint.
    pub fn with_cache(mut self, cache: CacheHint) -> Self {
        self.cache = cache;
        self
    }

    /// Replace the extensions.
    pub fn with_extensions(mut self, extensions: Extensions) -> Self {
        self.extensions = extensions;
        self
    }
}

impl From<&str> for CompletionRequest {
    fn from(prompt: &str) -> Self {
        Self::new(vec![Message::user(prompt)])
    }
}

impl From<String> for CompletionRequest {
    fn from(prompt: String) -> Self {
        Self::new(vec![Message::user(prompt)])
    }
}

impl From<Message> for CompletionRequest {
    fn from(message: Message) -> Self {
        Self::new(vec![message])
    }
}

impl From<Vec<Message>> for CompletionRequest {
    fn from(messages: Vec<Message>) -> Self {
        Self::new(messages)
    }
}

/// A tool the model may call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// The name the model uses to call it.
    pub name: String,
    /// What it does, for the model.
    pub description: String,
    /// The JSON schema of its arguments.
    pub parameters: serde_json::Value,
}

/// Whether and which tool the model must call.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    /// The model decides.
    #[default]
    Auto,
    /// The model must not call a tool.
    None,
    /// The model must call some tool.
    Required,
    /// The model must call this tool.
    Tool(String),
}

impl ToolChoice {
    fn is_auto(&self) -> bool {
        *self == Self::Auto
    }
}

/// A JSON schema the reply must follow.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputSchema {
    /// A short name for the schema, required by some providers.
    pub name: String,
    /// The JSON schema.
    pub schema: serde_json::Value,
}

impl OutputSchema {
    /// The schema of `T`, generated with `schemars`.
    pub fn of<T: schemars::JsonSchema>() -> Self {
        let schema = schemars::schema_for!(T);
        Self {
            name: sanitize_name(&T::schema_name()),
            schema: schema.to_value(),
        }
    }
}

/// Keep only the characters every provider accepts in a schema or tool name.
pub(crate) fn sanitize_name(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    cleaned.chars().take(64).collect()
}

/// How hard a reasoning model should think.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    /// As little as the model allows.
    Minimal,
    /// Low.
    Low,
    /// Medium.
    Medium,
    /// High.
    High,
}

impl ReasoningEffort {
    /// A token budget for providers that take one instead of an effort level.
    pub fn budget_tokens(self) -> u32 {
        match self {
            Self::Minimal => 1024,
            Self::Low => 2048,
            Self::Medium => 8192,
            Self::High => 24_576,
        }
    }
}

/// Whether to ask the provider to cache the request prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheHint {
    /// No caching beyond what the provider does on its own.
    #[default]
    None,
    /// Cache the system prompt, tools and conversation so far, so the next
    /// request that repeats them costs less.
    Prefix,
}

impl CacheHint {
    #[allow(clippy::trivially_copy_pass_by_ref)] // serde's `skip_serializing_if` passes a reference
    fn is_none(&self) -> bool {
        *self == Self::None
    }
}
