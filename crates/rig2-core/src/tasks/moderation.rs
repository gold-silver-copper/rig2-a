use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::Task;
use crate::catalog::ModelCard;

/// Classify texts for harmful content.
#[derive(Debug, Clone, Copy)]
pub struct Moderation;

impl Task for Moderation {
    const NAME: &'static str = "moderation";
    type Input = ModerationRequest;
    type Output = ModerationResponse;
    type Capabilities = ModelCard;
}

/// Texts to classify.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ModerationRequest {
    /// The texts, one result each.
    pub inputs: Vec<String>,
}

impl From<&str> for ModerationRequest {
    fn from(text: &str) -> Self {
        Self {
            inputs: vec![text.to_owned()],
        }
    }
}

/// One result per input, in input order.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ModerationResponse {
    /// The results.
    pub results: Vec<ModerationResult>,
}

/// The classification of one text.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ModerationResult {
    /// Whether any category was flagged.
    pub flagged: bool,
    /// Scores by the provider's category names, from 0 to 1.
    pub scores: BTreeMap<String, f32>,
}
