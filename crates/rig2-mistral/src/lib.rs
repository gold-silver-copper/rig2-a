//! Mistral for rig2.
//!
//! Mistral's API is OpenAI-compatible, so Mistral is a [`Dialect`] of the
//! OpenAI client: its chat models use the Chat Completions mapping, and its
//! embedding models the embeddings mapping. Reasoning models return their
//! thinking as content chunks, which the mapping reads as reasoning.
//!
//! ```no_run
//! # async fn demo() -> rig2_core::Result<()> {
//! use rig2_core::Model;
//!
//! let mistral = rig2_mistral::from_env()?;
//! let reply = mistral.chat("mistral-small-latest").run("Name a prime.").await?;
//! println!("{}", reply.text());
//! # Ok(()) }
//! ```

use rig2_core::{Result, Secret};
use rig2_openai::{Dialect, MaxTokens, OpenAIConfig, Quirks};

/// Mistral as an OpenAI-compatible dialect.
pub const DIALECT: Dialect = Dialect {
    name: "mistral",
    base_url: "https://api.mistral.ai/v1",
    key_var: "MISTRAL_API_KEY",
    preferred_api: "chat",
    quirks: Quirks {
        max_tokens: MaxTokens::MaxTokens,
        reasoning_effort: false,
        reasoning_content: false,
        stream_usage: false,
        json_schema: true,
        tool_result_images: false,
    },
    catalog: include_str!("../catalogs/mistral.json"),
};

/// Mistral's configuration with `key`.
pub fn config(key: impl Into<Secret>) -> OpenAIConfig {
    OpenAIConfig::for_dialect(DIALECT, key)
}

/// Mistral's configuration with the key from `MISTRAL_API_KEY`.
pub fn config_from_env() -> Result<OpenAIConfig> {
    OpenAIConfig::dialect_from_env(DIALECT)
}

/// A Mistral client with the key from `MISTRAL_API_KEY`, on the shared
/// reqwest client.
#[cfg(feature = "reqwest")]
pub fn from_env() -> Result<rig2_openai::OpenAI> {
    rig2_openai::OpenAI::dialect_from_env(DIALECT)
}
