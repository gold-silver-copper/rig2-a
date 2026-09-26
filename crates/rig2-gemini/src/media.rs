//! Embeddings, image generation and the model listing.

use std::sync::Arc;

use bytes::Bytes;
use rig2_core::catalog::ModelCard;
use rig2_core::completion::{CompletionRequest, respond};
use rig2_core::content::AssistantContent;
use rig2_core::http::{read_json, send};
use rig2_core::tasks::{
    Embedding, EmbeddingRequest, EmbeddingResponse, GeneratedImage, ImageGeneration,
    ImageGenerationRequest, ImageGenerationResponse, InputType, ListedModel, ModelListing,
    ModelListingRequest,
};
use rig2_core::vision::EncodedImage;
use rig2_core::{BoxFuture, Error, ErrorKind, Model, ModelInfo, Result};
use serde_json::{Value, json};

use crate::{Gemini, PROVIDER, generate};

fn decode(what: &str) -> Error {
    Error::new(ErrorKind::Decode, format!("the reply has no {what}")).with_provider(PROVIDER)
}

/// A Gemini embedding model.
#[derive(Debug, Clone)]
pub struct EmbeddingModel {
    client: Gemini,
    info: ModelInfo,
    card: ModelCard,
    dimensions: Option<u32>,
}

impl EmbeddingModel {
    pub(crate) fn new(client: Gemini, model: &str) -> Self {
        let card = client.card(model);
        Self {
            client,
            info: Gemini::info(model),
            card,
            dimensions: None,
        }
    }

    /// Ask for vectors of this width. A request's own `dimensions` wins.
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

    fn invoke(&self, request: EmbeddingRequest) -> BoxFuture<'static, Result<EmbeddingResponse>> {
        let name = format!("models/{}", self.info.model);
        let task = request.input_type.map(|t| match t {
            InputType::Query => "RETRIEVAL_QUERY",
            InputType::Document => "RETRIEVAL_DOCUMENT",
        });
        let dimensions = request.dimensions.or(self.dimensions);
        let requests: Vec<Value> = request
            .texts
            .iter()
            .map(|text| {
                let mut one = json!({ "model": name, "content": { "parts": [{ "text": text }] } });
                if let Some(task) = task {
                    rig2_core::http::set(&mut one, "taskType", task);
                }
                if let Some(dimensions) = dimensions {
                    rig2_core::http::set(&mut one, "outputDimensionality", dimensions);
                }
                one
            })
            .collect();
        let http = Arc::clone(self.client.http());
        let request = self.client.post_json(
            &format!("/{name}:batchEmbedContents"),
            &json!({ "requests": requests }),
        );
        Box::pin(async move {
            let doc = read_json(send(&*http, PROVIDER, request?).await?, PROVIDER).await?;
            let embeddings = doc
                .get("embeddings")
                .and_then(Value::as_array)
                .ok_or_else(|| decode("embeddings"))?
                .iter()
                .map(|e| {
                    Ok(serde_json::from_value::<Vec<f32>>(
                        e.get("values").cloned().ok_or_else(|| decode("values"))?,
                    )?)
                })
                .collect::<Result<_>>()?;
            Ok(EmbeddingResponse {
                embeddings,
                ..EmbeddingResponse::default()
            })
        })
    }
}

/// A Gemini model with image output, used for image generation.
#[derive(Debug, Clone)]
pub struct ImageModel {
    client: Gemini,
    info: ModelInfo,
    card: ModelCard,
}

impl ImageModel {
    pub(crate) fn new(client: Gemini, model: &str) -> Self {
        let card = client.card(model);
        Self {
            client,
            info: Gemini::info(model),
            card,
        }
    }
}

impl Model<ImageGeneration> for ImageModel {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn capabilities(&self) -> ModelCard {
        self.card.clone()
    }

    fn invoke(
        &self,
        request: ImageGenerationRequest,
    ) -> BoxFuture<'static, Result<ImageGenerationResponse>> {
        let http = Arc::clone(self.client.http());
        let path = format!("/models/{}:generateContent", self.info.model);
        let body = generate::request_body(&CompletionRequest::from(request.prompt.as_str()), true)
            .and_then(|b| self.client.post_json(&path, &b));
        Box::pin(async move {
            let doc = read_json(send(&*http, PROVIDER, body?).await?, PROVIDER).await?;
            let response = respond(|out| generate::write_reply(&doc, None, out))?;
            let images: Vec<GeneratedImage> = response
                .content
                .into_iter()
                .filter_map(|part| match part {
                    AssistantContent::Image(image) => {
                        image.bytes().map(|(media_type, bytes)| GeneratedImage {
                            image: EncodedImage { media_type, bytes },
                            revised_prompt: None,
                        })
                    }
                    _ => None,
                })
                .collect();
            if images.is_empty() {
                return Err(decode("image"));
            }
            Ok(ImageGenerationResponse {
                images,
                usage: response.usage,
            })
        })
    }
}

/// Gemini's model listing.
#[derive(Debug, Clone)]
pub struct ModelList {
    client: Gemini,
    info: ModelInfo,
}

impl ModelList {
    pub(crate) fn new(client: Gemini) -> Self {
        Self {
            client,
            info: Gemini::info(""),
        }
    }
}

impl Model<ModelListing> for ModelList {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn capabilities(&self) {}

    fn invoke(
        &self,
        _request: ModelListingRequest,
    ) -> BoxFuture<'static, Result<Vec<ListedModel>>> {
        let http = Arc::clone(self.client.http());
        let request = self
            .client
            .request(http::Method::GET, "/models?pageSize=1000", Bytes::new());
        Box::pin(async move {
            let doc = read_json(send(&*http, PROVIDER, request?).await?, PROVIDER).await?;
            Ok(doc
                .get("models")
                .and_then(Value::as_array)
                .ok_or_else(|| decode("models"))?
                .iter()
                .filter_map(|m| {
                    Some(ListedModel {
                        id: m
                            .get("name")?
                            .as_str()?
                            .trim_start_matches("models/")
                            .to_owned(),
                        name: m
                            .get("displayName")
                            .and_then(Value::as_str)
                            .map(str::to_owned),
                        owned_by: Some("google".into()),
                    })
                })
                .collect())
        })
    }
}
