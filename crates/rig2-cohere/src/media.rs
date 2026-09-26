//! Embeddings and rerank.

use std::sync::Arc;

use rig2_core::catalog::ModelCard;
use rig2_core::completion::Usage;
use rig2_core::http::{read_json, send, set};
use rig2_core::tasks::{
    Embedding, EmbeddingRequest, EmbeddingResponse, InputType, Rerank, RerankRequest,
    RerankResponse, RerankResult,
};
use rig2_core::{BoxFuture, Error, ErrorKind, Model, ModelInfo, Result};
use serde_json::{Value, json};

use crate::{Cohere, PROVIDER};

fn decode(what: &str) -> Error {
    Error::new(ErrorKind::Decode, format!("the reply has no {what}")).with_provider(PROVIDER)
}

fn billed_input(doc: &Value) -> u64 {
    doc.pointer("/meta/billed_units/input_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0)
}

/// A Cohere embedding model.
#[derive(Debug, Clone)]
pub struct EmbeddingModel {
    client: Cohere,
    info: ModelInfo,
    card: ModelCard,
    dimensions: Option<u32>,
}

impl EmbeddingModel {
    pub(crate) fn new(client: Cohere, model: &str) -> Self {
        let card = client.card(model);
        Self {
            client,
            info: Cohere::info(model),
            card,
            dimensions: None,
        }
    }

    /// Ask for vectors of this width, for models that can shorten them. A
    /// request's own `dimensions` wins.
    pub fn with_ndims(mut self, dimensions: u32) -> Self {
        self.dimensions = Some(dimensions);
        self
    }
}

impl Model<Embedding> for EmbeddingModel {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn capabilities(&self) -> ModelCard {
        self.card.clone()
    }

    /// Texts without an input type embed as documents, which Cohere
    /// requires a choice for.
    fn invoke(&self, request: EmbeddingRequest) -> BoxFuture<'static, Result<EmbeddingResponse>> {
        let input_type = match request.input_type {
            Some(InputType::Query) => "search_query",
            Some(InputType::Document) | None => "search_document",
        };
        let mut body = json!({
            "model": self.info.model,
            "texts": request.texts,
            "input_type": input_type,
            "embedding_types": ["float"],
        });
        if let Some(dimensions) = request.dimensions.or(self.dimensions) {
            set(&mut body, "output_dimension", dimensions);
        }
        let http = Arc::clone(self.client.http());
        let request = self.client.post_json("/embed", &body);
        Box::pin(async move {
            let doc = read_json(send(&*http, PROVIDER, request?).await?, PROVIDER).await?;
            let embeddings: Vec<Vec<f32>> = serde_json::from_value(
                doc.pointer("/embeddings/float")
                    .cloned()
                    .ok_or_else(|| decode("embeddings"))?,
            )?;
            Ok(EmbeddingResponse {
                embeddings,
                usage: Usage {
                    input_tokens: billed_input(&doc),
                    ..Usage::default()
                },
            })
        })
    }
}

/// A Cohere rerank model.
#[derive(Debug, Clone)]
pub struct RerankModel {
    client: Cohere,
    info: ModelInfo,
    card: ModelCard,
}

impl RerankModel {
    pub(crate) fn new(client: Cohere, model: &str) -> Self {
        let card = client.card(model);
        Self {
            client,
            info: Cohere::info(model),
            card,
        }
    }
}

impl Model<Rerank> for RerankModel {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn capabilities(&self) -> ModelCard {
        self.card.clone()
    }

    fn invoke(&self, request: RerankRequest) -> BoxFuture<'static, Result<RerankResponse>> {
        let mut body = json!({ "model": self.info.model, "query": request.query, "documents": request.documents });
        if let Some(top_n) = request.top_n {
            set(&mut body, "top_n", top_n);
        }
        let http = Arc::clone(self.client.http());
        let request = self.client.post_json("/rerank", &body);
        Box::pin(async move {
            let doc = read_json(send(&*http, PROVIDER, request?).await?, PROVIDER).await?;
            let results = doc
                .get("results")
                .and_then(Value::as_array)
                .ok_or_else(|| decode("results"))?
                .iter()
                .map(|r| {
                    Ok(RerankResult {
                        index: r
                            .get("index")
                            .and_then(Value::as_u64)
                            .and_then(|i| u32::try_from(i).ok())
                            .ok_or_else(|| decode("index"))?,
                        score: serde_json::from_value(
                            r.get("relevance_score")
                                .cloned()
                                .ok_or_else(|| decode("score"))?,
                        )?,
                    })
                })
                .collect::<Result<_>>()?;
            Ok(RerankResponse { results })
        })
    }
}
