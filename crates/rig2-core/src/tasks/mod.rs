//! The built-in tasks besides completion and vision.
//!
//! Each task is a unit struct implementing [`Task`](crate::Task), with plain
//! data request and response types. Providers implement
//! [`Model<T>`](crate::Model) for the tasks they support.

mod audio;
mod embedding;
mod image;
mod listing;
mod moderation;
mod rerank;

pub use audio::{
    AudioFormat, Speech, SpeechRequest, SpeechResponse, Transcription, TranscriptionRequest,
    TranscriptionResponse,
};
pub use embedding::{
    Embedding, EmbeddingRequest, EmbeddingResponse, ImageEmbedding, ImageEmbeddingRequest,
    InputType,
};
pub use image::{
    GeneratedImage, ImageGeneration, ImageGenerationRequest, ImageGenerationResponse, ImageQuality,
};
pub use listing::{ListedModel, ModelListing, ModelListingRequest};
pub use moderation::{Moderation, ModerationRequest, ModerationResponse, ModerationResult};
pub use rerank::{Rerank, RerankRequest, RerankResponse, RerankResult};
