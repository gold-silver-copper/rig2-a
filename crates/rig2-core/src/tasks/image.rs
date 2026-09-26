use serde::{Deserialize, Serialize};

use crate::Task;
use crate::catalog::ModelCard;
use crate::completion::Usage;
use crate::vision::EncodedImage;

/// Generate images from a prompt.
#[derive(Debug, Clone, Copy)]
pub struct ImageGeneration;

impl Task for ImageGeneration {
    const NAME: &'static str = "image_generation";
    type Input = ImageGenerationRequest;
    type Output = ImageGenerationResponse;
    type Capabilities = ModelCard;
}

/// What to draw.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ImageGenerationRequest {
    /// The description.
    pub prompt: String,
    /// Width and height in pixels, from the sizes the model supports.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<(u32, u32)>,
    /// How many images.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub count: Option<u32>,
    /// The quality tier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<ImageQuality>,
}

impl ImageGenerationRequest {
    /// Set the size.
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.size = Some((width, height));
        self
    }

    /// Set the quality.
    pub fn with_quality(mut self, quality: ImageQuality) -> Self {
        self.quality = Some(quality);
        self
    }
}

impl From<&str> for ImageGenerationRequest {
    fn from(prompt: &str) -> Self {
        Self {
            prompt: prompt.to_owned(),
            ..Self::default()
        }
    }
}

impl From<String> for ImageGenerationRequest {
    fn from(prompt: String) -> Self {
        Self {
            prompt,
            ..Self::default()
        }
    }
}

/// A quality tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageQuality {
    /// The cheapest.
    Low,
    /// The default.
    Medium,
    /// The best.
    High,
}

/// Generated images.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ImageGenerationResponse {
    /// The images.
    pub images: Vec<GeneratedImage>,
    /// Token counts, for token-priced models.
    #[serde(default)]
    pub usage: Usage,
}

/// One generated image.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeneratedImage {
    /// The image.
    pub image: EncodedImage,
    /// The prompt as the provider rewrote it, if it did.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revised_prompt: Option<String>,
}
