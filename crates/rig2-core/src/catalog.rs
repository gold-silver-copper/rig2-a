//! The model catalog: what each model can do and what it costs.
//!
//! A catalog is plain data, checked into each provider crate as JSON. Each
//! capability records whether it is claimed (by the provider's listing or
//! docs) or verified (by a live call), and when. Models report their card as
//! their capabilities, agents read context windows from it, and usage times
//! pricing gives the cost of every response.
//!
//! ```
//! use rig2_core::catalog::{Capability, Catalog};
//!
//! let catalog = Catalog::parse(r#"{"provider":"demo","models":[{"id":"m","capabilities":{"tools":{"supported":true,"source":"verified","at":"2026-09-26"}}}]}"#).unwrap();
//! assert!(catalog.card("m").supports(Capability::Tools));
//! assert!(!catalog.card("unknown").supports(Capability::Tools));
//! ```

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::Result;

/// Something a model may or may not support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    /// Streamed replies.
    Streaming,
    /// Tool calls.
    Tools,
    /// Several tool calls in one reply.
    ParallelTools,
    /// Replies constrained to a JSON schema.
    StructuredOutput,
    /// Images in the input.
    ImageInput,
    /// Audio in the input.
    AudioInput,
    /// Documents (PDF) in the input.
    FileInput,
    /// Reasoning (thinking) output.
    Reasoning,
    /// Prompt caching.
    PromptCaching,
    /// Images in the output.
    ImageOutput,
}

impl Capability {
    /// Every capability, in a fixed order.
    pub const ALL: [Self; 10] = [
        Self::Streaming,
        Self::Tools,
        Self::ParallelTools,
        Self::StructuredOutput,
        Self::ImageInput,
        Self::AudioInput,
        Self::FileInput,
        Self::Reasoning,
        Self::PromptCaching,
        Self::ImageOutput,
    ];
}

/// Where a capability entry comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Evidence {
    /// The provider's listing or documentation says so.
    Claimed,
    /// A live call showed it.
    Verified,
}

/// One capability entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Claim {
    /// Whether the model supports the capability.
    pub supported: bool,
    /// Where the answer comes from.
    pub source: Evidence,
    /// When it was established, as `YYYY-MM-DD`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<String>,
}

/// Prices in US dollars per million tokens.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Pricing {
    /// Uncached input.
    pub input: f64,
    /// Output, reasoning included.
    pub output: f64,
    /// Input served from the cache, when priced differently.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_input: Option<f64>,
    /// Input written to the cache, when priced differently.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write: Option<f64>,
}

/// Everything the catalog knows about one model.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ModelCard {
    /// The provider's model id.
    pub id: String,
    /// The most input tokens the model accepts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u32>,
    /// The most output tokens the model produces.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output: Option<u32>,
    /// The embedding width, for embedding models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    /// What each capability claim says.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub capabilities: BTreeMap<Capability, Claim>,
    /// Token prices.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pricing: Option<Pricing>,
}

impl ModelCard {
    /// A card that knows nothing but the id.
    pub fn unknown(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            ..Self::default()
        }
    }

    /// Whether the catalog says the model supports `capability`.
    ///
    /// An unknown capability counts as unsupported.
    pub fn supports(&self, capability: Capability) -> bool {
        self.capabilities
            .get(&capability)
            .is_some_and(|c| c.supported)
    }

    /// The cost of `usage` at this model's prices, when known.
    pub fn cost(&self, usage: &crate::completion::Usage) -> Option<f64> {
        self.pricing.as_ref().map(|p| usage.cost(p))
    }
}

/// A provider's catalog: its listing date and its model cards.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Catalog {
    /// The provider name.
    pub provider: String,
    /// When the provider's model listing was fetched, as `YYYY-MM-DD`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub listed_at: Option<String>,
    /// The models.
    pub models: Vec<ModelCard>,
}

impl Catalog {
    /// Parse a catalog from its JSON form.
    pub fn parse(json: &str) -> Result<Self> {
        Ok(serde_json::from_str(json)?)
    }

    /// The card for `id`, or an unknown card when the catalog lacks it.
    pub fn card(&self, id: &str) -> ModelCard {
        self.models
            .iter()
            .find(|m| m.id == id)
            .cloned()
            .unwrap_or_else(|| ModelCard::unknown(id))
    }
}
