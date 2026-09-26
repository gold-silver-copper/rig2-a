//! The non-chat endpoints: embeddings, images, speech, transcription,
//! moderation and the model listing.

use std::sync::Arc;

use bytes::Bytes;
use rig2_core::catalog::ModelCard;
use rig2_core::completion::Usage;
use rig2_core::http::{Multipart, read_body, read_json, send, set};
use rig2_core::tasks::{
    AudioFormat, Embedding, EmbeddingRequest, EmbeddingResponse, GeneratedImage, ImageGeneration,
    ImageGenerationRequest, ImageGenerationResponse, ImageQuality, ListedModel, ModelListing,
    ModelListingRequest, Moderation, ModerationRequest, ModerationResponse, ModerationResult,
    Speech, SpeechRequest, SpeechResponse, Transcription, TranscriptionRequest,
    TranscriptionResponse,
};
use rig2_core::vision::EncodedImage;
use rig2_core::{BoxFuture, Error, ErrorKind, Model, ModelInfo, Result};
use serde_json::{Value, json};

use crate::OpenAI;

fn decode(what: &str) -> Error {
    Error::new(ErrorKind::Decode, format!("the reply has no {what}"))
}

/// Parse `{"data": [{"embedding": [...]}], "usage": {...}}`.
pub(crate) fn embedding_response(doc: &Value) -> Result<EmbeddingResponse> {
    let mut rows: Vec<(u64, Vec<f32>)> = doc
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| decode("data"))?
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let index = row
                .get("index")
                .and_then(Value::as_u64)
                .unwrap_or_else(|| u64::try_from(i).unwrap_or(0));
            let vector: Vec<f32> = serde_json::from_value(
                row.get("embedding")
                    .cloned()
                    .ok_or_else(|| decode("embedding"))?,
            )?;
            Ok((index, vector))
        })
        .collect::<Result<_>>()?;
    rows.sort_by_key(|(index, _)| *index);
    let input = doc
        .pointer("/usage/prompt_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    Ok(EmbeddingResponse {
        embeddings: rows.into_iter().map(|(_, v)| v).collect(),
        usage: Usage {
            input_tokens: input,
            ..Usage::default()
        },
    })
}

macro_rules! model_type {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[derive(Debug, Clone)]
        pub struct $name {
            client: OpenAI,
            info: ModelInfo,
            card: ModelCard,
        }

        impl $name {
            pub(crate) fn new(client: OpenAI, model: &str) -> Self {
                let (info, card) = (client.info(model), client.card(model));
                Self { client, info, card }
            }
        }
    };
}

/// An embedding model.
#[derive(Debug, Clone)]
pub struct EmbeddingModel {
    client: OpenAI,
    info: ModelInfo,
    card: ModelCard,
    dimensions: Option<u32>,
}

impl EmbeddingModel {
    pub(crate) fn new(client: OpenAI, model: &str) -> Self {
        let (info, card) = (client.info(model), client.card(model));
        Self {
            client,
            info,
            card,
            dimensions: None,
        }
    }

    /// Ask for vectors of this width, for models that can shorten them.
    /// A request's own `dimensions` wins.
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
        let mut body =
            json!({ "model": self.info.model, "input": request.texts, "encoding_format": "float" });
        if let Some(dimensions) = request.dimensions.or(self.dimensions) {
            set(&mut body, "dimensions", json!(dimensions));
        }
        let (http, provider) = (Arc::clone(self.client.http()), self.client.provider());
        let request = self.client.post_json("/embeddings", &body);
        Box::pin(async move {
            let doc = read_json(send(&*http, provider, request?).await?, provider).await?;
            embedding_response(&doc)
        })
    }
}

model_type! {
    /// An image generation model.
    ImageModel
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
        let gpt_image = self.info.model.starts_with("gpt-image");
        let mut body = json!({ "model": self.info.model, "prompt": request.prompt, "n": request.count.unwrap_or(1) });
        if let Some((w, h)) = request.size {
            set(&mut body, "size", json!(format!("{w}x{h}")));
        }
        if let Some(quality) = request.quality {
            set(
                &mut body,
                "quality",
                json!(match (quality, gpt_image) {
                    (ImageQuality::Low, true) => "low",
                    (ImageQuality::Medium, true) => "medium",
                    (ImageQuality::High, true) => "high",
                    (ImageQuality::High, false) => "hd",
                    (_, false) => "standard",
                }),
            );
        }
        if !gpt_image {
            set(&mut body, "response_format", json!("b64_json"));
        }
        let (http, provider) = (Arc::clone(self.client.http()), self.client.provider());
        let request = self.client.post_json("/images/generations", &body);
        Box::pin(async move {
            let doc = read_json(send(&*http, provider, request?).await?, provider).await?;
            let images = doc
                .get("data")
                .and_then(Value::as_array)
                .ok_or_else(|| decode("data"))?
                .iter()
                .map(|row| {
                    let b64 = row
                        .get("b64_json")
                        .and_then(Value::as_str)
                        .ok_or_else(|| decode("b64_json"))?;
                    let (_, bytes) =
                        rig2_core::http::parse_data_url(&format!("data:image/png;base64,{b64}"))?;
                    Ok(GeneratedImage {
                        image: EncodedImage::png(bytes),
                        revised_prompt: row
                            .get("revised_prompt")
                            .and_then(Value::as_str)
                            .map(str::to_owned),
                    })
                })
                .collect::<Result<_>>()?;
            let usage = Usage {
                input_tokens: doc
                    .pointer("/usage/input_tokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
                output_tokens: doc
                    .pointer("/usage/output_tokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
                ..Usage::default()
            };
            Ok(ImageGenerationResponse { images, usage })
        })
    }
}

model_type! {
    /// A text-to-speech model. The default voice is `alloy`.
    SpeechModel
}

impl Model<Speech> for SpeechModel {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn capabilities(&self) -> ModelCard {
        self.card.clone()
    }

    fn invoke(&self, request: SpeechRequest) -> BoxFuture<'static, Result<SpeechResponse>> {
        let format = request.format.unwrap_or(AudioFormat::Mp3);
        let mut body = json!({
            "model": self.info.model,
            "input": request.text,
            "voice": request.voice.clone().unwrap_or_else(|| "alloy".into()),
            "response_format": format.extension().replace("ogg", "opus"),
        });
        if let Some(speed) = request.speed {
            set(&mut body, "speed", json!(speed));
        }
        let (http, provider) = (Arc::clone(self.client.http()), self.client.provider());
        let request = self.client.post_json("/audio/speech", &body);
        Box::pin(async move {
            let response = send(&*http, provider, request?).await?;
            let audio: Bytes = read_body(response.into_body()).await?;
            Ok(SpeechResponse { audio, format })
        })
    }
}

model_type! {
    /// A speech-to-text model.
    TranscriptionModel
}

impl Model<Transcription> for TranscriptionModel {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn capabilities(&self) -> ModelCard {
        self.card.clone()
    }

    fn invoke(
        &self,
        request: TranscriptionRequest,
    ) -> BoxFuture<'static, Result<TranscriptionResponse>> {
        let file_name = format!("audio.{}", request.format.extension());
        let mut form = Multipart::new()
            .text("model", &self.info.model)
            .text("response_format", "json")
            .file(
                "file",
                &file_name,
                request.format.media_type(),
                &request.audio,
            );
        if let Some(language) = &request.language {
            form = form.text("language", language);
        }
        if let Some(prompt) = &request.prompt {
            form = form.text("prompt", prompt);
        }
        let (content_type, body) = form.finish();
        let (http, provider) = (Arc::clone(self.client.http()), self.client.provider());
        let request = self.client.request(
            http::Method::POST,
            "/audio/transcriptions",
            Some(&content_type),
            body,
        );
        Box::pin(async move {
            let doc = read_json(send(&*http, provider, request?).await?, provider).await?;
            Ok(TranscriptionResponse {
                text: doc
                    .get("text")
                    .and_then(Value::as_str)
                    .ok_or_else(|| decode("text"))?
                    .to_owned(),
                language: doc
                    .get("language")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                duration: doc.get("duration").and_then(Value::as_f64),
            })
        })
    }
}

model_type! {
    /// A moderation model.
    ModerationModel
}

impl Model<Moderation> for ModerationModel {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn capabilities(&self) -> ModelCard {
        self.card.clone()
    }

    fn invoke(&self, request: ModerationRequest) -> BoxFuture<'static, Result<ModerationResponse>> {
        let body = json!({ "model": self.info.model, "input": request.inputs });
        let (http, provider) = (Arc::clone(self.client.http()), self.client.provider());
        let request = self.client.post_json("/moderations", &body);
        Box::pin(async move {
            let doc = read_json(send(&*http, provider, request?).await?, provider).await?;
            let results = doc
                .get("results")
                .and_then(Value::as_array)
                .ok_or_else(|| decode("results"))?
                .iter()
                .map(|r| ModerationResult {
                    flagged: r.get("flagged").and_then(Value::as_bool).unwrap_or(false),
                    scores: r
                        .get("category_scores")
                        .and_then(Value::as_object)
                        .into_iter()
                        .flatten()
                        .filter_map(|(k, v)| {
                            Some((k.clone(), serde_json::from_value::<f32>(v.clone()).ok()?))
                        })
                        .collect(),
                })
                .collect();
            Ok(ModerationResponse { results })
        })
    }
}

/// The provider's model listing.
#[derive(Debug, Clone)]
pub struct ModelList {
    client: OpenAI,
    info: ModelInfo,
}

impl ModelList {
    pub(crate) fn new(client: OpenAI) -> Self {
        let info = client.info("");
        Self { client, info }
    }
}

/// Parse `{"data": [{"id": ..., "owned_by": ...}]}`.
pub(crate) fn listing(doc: &Value) -> Result<Vec<ListedModel>> {
    Ok(doc
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| decode("data"))?
        .iter()
        .filter_map(|m| {
            Some(ListedModel {
                id: m.get("id")?.as_str()?.to_owned(),
                name: m.get("name").and_then(Value::as_str).map(str::to_owned),
                owned_by: m.get("owned_by").and_then(Value::as_str).map(str::to_owned),
            })
        })
        .collect())
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
        let (http, provider) = (Arc::clone(self.client.http()), self.client.provider());
        let request = self
            .client
            .request(http::Method::GET, "/models", None, Bytes::new());
        Box::pin(async move {
            let doc = read_json(send(&*http, provider, request?).await?, provider).await?;
            listing(&doc)
        })
    }
}
