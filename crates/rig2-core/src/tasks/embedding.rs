use serde::{Deserialize, Serialize};

use crate::Task;
use crate::catalog::ModelCard;
use crate::completion::Usage;
use crate::vision::EncodedImage;

/// Turn texts into vectors.
#[derive(Debug, Clone, Copy)]
pub struct Embedding;

impl Task for Embedding {
    const NAME: &'static str = "embeddings";
    type Input = EmbeddingRequest;
    type Output = EmbeddingResponse;
    type Capabilities = ModelCard;

    fn record(span: &tracing::Span, output: &EmbeddingResponse) {
        span.record("gen_ai.usage.input_tokens", output.usage.input_tokens);
    }
}

/// Texts to embed.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct EmbeddingRequest {
    /// The texts, one vector each.
    pub texts: Vec<String>,
    /// What the texts are for, for providers that embed queries and
    /// documents differently.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_type: Option<InputType>,
    /// The vector width, for models that can shorten their output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
}

impl EmbeddingRequest {
    /// Embed these texts.
    pub fn new(texts: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            texts: texts.into_iter().map(Into::into).collect(),
            ..Self::default()
        }
    }

    /// Say what the texts are for.
    pub fn with_input_type(mut self, input_type: InputType) -> Self {
        self.input_type = Some(input_type);
        self
    }

    /// Ask for vectors of this width.
    pub fn with_dimensions(mut self, dimensions: u32) -> Self {
        self.dimensions = Some(dimensions);
        self
    }
}

impl From<&str> for EmbeddingRequest {
    fn from(text: &str) -> Self {
        Self::new([text])
    }
}

impl From<String> for EmbeddingRequest {
    fn from(text: String) -> Self {
        Self::new([text])
    }
}

impl From<Vec<String>> for EmbeddingRequest {
    fn from(texts: Vec<String>) -> Self {
        Self::new(texts)
    }
}

/// What embedded texts are for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputType {
    /// A search query.
    Query,
    /// A document to be searched.
    Document,
}

/// Vectors, one per input, in input order.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    /// The vectors.
    pub embeddings: Vec<Vec<f32>>,
    /// Token counts.
    #[serde(default)]
    pub usage: Usage,
}

/// Turn images into vectors.
#[derive(Debug, Clone, Copy)]
pub struct ImageEmbedding;

impl Task for ImageEmbedding {
    const NAME: &'static str = "image_embeddings";
    type Input = ImageEmbeddingRequest;
    type Output = EmbeddingResponse;
    type Capabilities = ModelCard;
}

/// Images to embed.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ImageEmbeddingRequest {
    /// The images, one vector each.
    pub images: Vec<EncodedImage>,
}

impl From<EncodedImage> for ImageEmbeddingRequest {
    fn from(image: EncodedImage) -> Self {
        Self {
            images: vec![image],
        }
    }
}
