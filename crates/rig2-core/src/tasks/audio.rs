use bytes::Bytes;
use serde::{Deserialize, Serialize};

use crate::Task;
use crate::catalog::ModelCard;

/// Turn text into speech.
#[derive(Debug, Clone, Copy)]
pub struct Speech;

impl Task for Speech {
    const NAME: &'static str = "speech";
    type Input = SpeechRequest;
    type Output = SpeechResponse;
    type Capabilities = ModelCard;
}

/// What to say, and how.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SpeechRequest {
    /// The text.
    pub text: String,
    /// The provider's voice name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    /// The audio format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<AudioFormat>,
    /// Playback speed, 1.0 being normal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed: Option<f32>,
}

impl SpeechRequest {
    /// Choose the voice.
    pub fn with_voice(mut self, voice: impl Into<String>) -> Self {
        self.voice = Some(voice.into());
        self
    }

    /// Choose the format.
    pub fn with_format(mut self, format: AudioFormat) -> Self {
        self.format = Some(format);
        self
    }
}

impl From<&str> for SpeechRequest {
    fn from(text: &str) -> Self {
        Self {
            text: text.to_owned(),
            ..Self::default()
        }
    }
}

/// An audio encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioFormat {
    /// MP3.
    Mp3,
    /// WAV.
    Wav,
    /// Opus in Ogg.
    Opus,
    /// FLAC.
    Flac,
    /// AAC.
    Aac,
    /// Raw 16-bit little-endian PCM.
    Pcm,
}

impl AudioFormat {
    /// The media type.
    pub fn media_type(self) -> &'static str {
        match self {
            Self::Mp3 => "audio/mpeg",
            Self::Wav => "audio/wav",
            Self::Opus => "audio/ogg",
            Self::Flac => "audio/flac",
            Self::Aac => "audio/aac",
            Self::Pcm => "audio/pcm",
        }
    }

    /// The usual file extension.
    pub fn extension(self) -> &'static str {
        match self {
            Self::Mp3 => "mp3",
            Self::Wav => "wav",
            Self::Opus => "ogg",
            Self::Flac => "flac",
            Self::Aac => "aac",
            Self::Pcm => "pcm",
        }
    }
}

/// Synthesized speech.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeechResponse {
    /// The audio bytes.
    #[serde(with = "crate::base64_bytes")]
    pub audio: Bytes,
    /// Their format.
    pub format: AudioFormat,
}

/// Turn speech into text.
#[derive(Debug, Clone, Copy)]
pub struct Transcription;

impl Task for Transcription {
    const NAME: &'static str = "transcription";
    type Input = TranscriptionRequest;
    type Output = TranscriptionResponse;
    type Capabilities = ModelCard;
}

/// Audio to transcribe.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TranscriptionRequest {
    /// The audio bytes.
    #[serde(with = "crate::base64_bytes")]
    pub audio: Bytes,
    /// Their format.
    pub format: AudioFormat,
    /// The spoken language as an ISO-639-1 code, if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Text that guides the transcription, such as expected vocabulary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
}

impl TranscriptionRequest {
    /// Transcribe these bytes.
    pub fn new(audio: impl Into<Bytes>, format: AudioFormat) -> Self {
        Self {
            audio: audio.into(),
            format,
            language: None,
            prompt: None,
        }
    }

    /// Say which language is spoken.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }
}

/// A transcript.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct TranscriptionResponse {
    /// The text.
    pub text: String,
    /// The detected language, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// The audio length in seconds, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration: Option<f64>,
}
