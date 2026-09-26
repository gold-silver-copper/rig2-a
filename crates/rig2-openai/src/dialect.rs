//! OpenAI-compatible providers as data.

/// How a provider differs from OpenAI's own API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Quirks {
    /// Which field carries the output token limit.
    pub max_tokens: MaxTokens,
    /// Whether it takes `reasoning_effort`.
    pub reasoning_effort: bool,
    /// Whether replies carry reasoning as `reasoning_content` (DeepSeek) or
    /// `reasoning` (Groq, OpenRouter).
    pub reasoning_content: bool,
    /// Whether it takes `stream_options.include_usage`.
    pub stream_usage: bool,
    /// Whether it takes `response_format` with a JSON schema.
    pub json_schema: bool,
    /// Whether tool results may carry images.
    pub tool_result_images: bool,
}

/// Which field carries the output token limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaxTokens {
    /// `max_completion_tokens`, OpenAI's current field.
    MaxCompletionTokens,
    /// `max_tokens`, the older field most compatible providers take.
    MaxTokens,
}

/// An OpenAI-compatible provider: where it lives, where its key is, and how
/// it differs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dialect {
    /// The provider name, as models and the registry report it.
    pub name: &'static str,
    /// The API base URL, without a trailing slash.
    pub base_url: &'static str,
    /// The environment variable holding the key.
    pub key_var: &'static str,
    /// `responses` or `chat`: the API a spec without an `api` option uses.
    pub preferred_api: &'static str,
    /// How it differs from OpenAI.
    pub quirks: Quirks,
    /// Its model catalog, as JSON.
    pub catalog: &'static str,
}

const OPENAI_QUIRKS: Quirks = Quirks {
    max_tokens: MaxTokens::MaxCompletionTokens,
    reasoning_effort: true,
    reasoning_content: false,
    stream_usage: true,
    json_schema: true,
    tool_result_images: false,
};

/// OpenAI itself.
pub const OPENAI: Dialect = Dialect {
    name: "openai",
    base_url: "https://api.openai.com/v1",
    key_var: "OPENAI_API_KEY",
    preferred_api: "responses",
    quirks: OPENAI_QUIRKS,
    catalog: include_str!("../catalogs/openai.json"),
};
