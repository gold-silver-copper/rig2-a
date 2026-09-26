use serde::{Deserialize, Serialize};

use crate::Task;
use crate::catalog::ModelCard;

/// Order documents by relevance to a query.
#[derive(Debug, Clone, Copy)]
pub struct Rerank;

impl Task for Rerank {
    const NAME: &'static str = "rerank";
    type Input = RerankRequest;
    type Output = RerankResponse;
    type Capabilities = ModelCard;
}

/// A query and the documents to rank.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RerankRequest {
    /// The query.
    pub query: String,
    /// The documents.
    pub documents: Vec<String>,
    /// Return at most this many results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_n: Option<u32>,
}

impl RerankRequest {
    /// Rank `documents` against `query`.
    pub fn new(
        query: impl Into<String>,
        documents: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            query: query.into(),
            documents: documents.into_iter().map(Into::into).collect(),
            top_n: None,
        }
    }

    /// Keep only the best `n`.
    pub fn with_top_n(mut self, n: u32) -> Self {
        self.top_n = Some(n);
        self
    }
}

/// Documents ranked best first.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct RerankResponse {
    /// The ranking.
    pub results: Vec<RerankResult>,
}

/// One ranked document.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RerankResult {
    /// The document's position in the request.
    pub index: u32,
    /// Its relevance score; higher is more relevant.
    pub score: f32,
}
