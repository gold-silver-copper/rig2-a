//! Messages and the content parts they are made of.
//!
//! A [`Message`] is either a user turn or an assistant turn, and each has its
//! own content type, so a user message cannot carry a tool call and an
//! assistant message cannot carry a tool result. Media carry their bytes
//! (serialized as base64) or a URL. Provider-specific data sits in typed
//! [`Extensions`].
//!
//! ```
//! use rig2_core::content::{Message, UserContent};
//!
//! let message = Message::user("What is in this picture?");
//! assert!(matches!(&message, Message::User { content } if matches!(content[0], UserContent::Text(_))));
//! ```

use std::collections::BTreeMap;

use bytes::Bytes;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::{Error, ErrorKind, Result};

/// One turn of a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum Message {
    /// What the user (or the application on the user's behalf) said,
    /// including tool results.
    User {
        /// The parts of the turn, in order.
        content: Vec<UserContent>,
    },
    /// What the model said, including tool calls and reasoning.
    Assistant {
        /// The parts of the turn, in order.
        content: Vec<AssistantContent>,
    },
}

impl Message {
    /// A user turn with one text part.
    pub fn user(text: impl Into<String>) -> Self {
        Self::User {
            content: vec![UserContent::Text(Text::new(text))],
        }
    }

    /// An assistant turn with one text part.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self::Assistant {
            content: vec![AssistantContent::Text(Text::new(text))],
        }
    }

    /// A user turn carrying tool results.
    pub fn tool_results(results: impl IntoIterator<Item = ToolResult>) -> Self {
        Self::User {
            content: results.into_iter().map(UserContent::ToolResult).collect(),
        }
    }

    /// The text parts of the message, joined.
    pub fn text(&self) -> String {
        match self {
            Self::User { content } => join_text(content.iter().filter_map(|c| match c {
                UserContent::Text(t) => Some(t.text.as_str()),
                _ => None,
            })),
            Self::Assistant { content } => join_text(content.iter().filter_map(|c| match c {
                AssistantContent::Text(t) => Some(t.text.as_str()),
                _ => None,
            })),
        }
    }
}

/// A user turn with one text part.
impl From<&str> for Message {
    fn from(text: &str) -> Self {
        Self::user(text)
    }
}

/// A user turn with one text part.
impl From<String> for Message {
    fn from(text: String) -> Self {
        Self::user(text)
    }
}

pub(crate) fn join_text<'a>(parts: impl Iterator<Item = &'a str>) -> String {
    parts.collect::<Vec<_>>().join("")
}

/// A part of a user turn.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UserContent {
    /// Text.
    Text(Text),
    /// An image.
    Image(Image),
    /// An audio clip.
    Audio(Audio),
    /// A document, such as a PDF.
    File(File),
    /// The result of a tool call the assistant made.
    ToolResult(ToolResult),
}

/// A part of an assistant turn.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AssistantContent {
    /// Text, possibly with citations.
    Text(Text),
    /// A request to call a tool.
    ToolCall(ToolCall),
    /// The model's reasoning, as the provider exposes it.
    Reasoning(Reasoning),
    /// An image the model generated.
    Image(Image),
}

/// Text, with the sources it cites.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Text {
    /// The text.
    pub text: String,
    /// The sources this text cites, if the provider reported any.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub citations: Vec<Citation>,
    /// Provider-specific data.
    #[serde(default, skip_serializing_if = "Extensions::is_empty")]
    pub extensions: Extensions,
}

impl Text {
    /// Text with no citations.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            ..Self::default()
        }
    }
}

/// A source that a text part cites.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Citation {
    /// The cited passage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cited_text: Option<String>,
    /// The source's title.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// The source's URL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// The index of the cited document in the request, for document citations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub document_index: Option<u32>,
    /// Provider-specific data, such as character offsets.
    #[serde(default, skip_serializing_if = "Extensions::is_empty")]
    pub extensions: Extensions,
}

/// Where media bytes come from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    /// The bytes themselves, serialized as base64.
    Bytes(#[serde(with = "crate::base64_bytes")] Bytes),
    /// A URL the provider fetches.
    Url(String),
}

/// An image in a message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Image {
    /// The image data or its URL.
    pub source: Source,
    /// The media type, such as `image/png`. Required for bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    /// Provider-specific data, such as OpenAI's `detail`.
    #[serde(default, skip_serializing_if = "Extensions::is_empty")]
    pub extensions: Extensions,
}

impl Image {
    /// An image from bytes with a media type.
    pub fn from_bytes(bytes: impl Into<Bytes>, media_type: impl Into<String>) -> Self {
        Self {
            source: Source::Bytes(bytes.into()),
            media_type: Some(media_type.into()),
            extensions: Extensions::default(),
        }
    }

    /// An image the provider fetches from a URL.
    pub fn from_url(url: impl Into<String>) -> Self {
        Self {
            source: Source::Url(url.into()),
            media_type: None,
            extensions: Extensions::default(),
        }
    }
}

/// An audio clip in a message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Audio {
    /// The audio data or its URL.
    pub source: Source,
    /// The media type, such as `audio/wav`.
    pub media_type: String,
    /// Provider-specific data.
    #[serde(default, skip_serializing_if = "Extensions::is_empty")]
    pub extensions: Extensions,
}

/// A document in a message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct File {
    /// The file data or its URL.
    pub source: Source,
    /// The media type, such as `application/pdf`.
    pub media_type: String,
    /// The file name shown to the model, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Provider-specific data.
    #[serde(default, skip_serializing_if = "Extensions::is_empty")]
    pub extensions: Extensions,
}

/// A request from the model to call a tool.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    /// The id the result must refer to.
    pub id: String,
    /// The tool's name.
    pub name: String,
    /// The arguments, parsed from the model's JSON.
    pub arguments: serde_json::Value,
    /// Provider-specific data, such as a Gemini thought signature.
    #[serde(default, skip_serializing_if = "Extensions::is_empty")]
    pub extensions: Extensions,
}

impl ToolCall {
    /// A tool call with parsed arguments.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments,
            extensions: Extensions::default(),
        }
    }
}

/// The result of running a tool, sent back to the model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolResult {
    /// The id of the call this answers.
    pub call_id: String,
    /// The tool's name. Some providers require it.
    pub name: String,
    /// What the tool returned.
    pub output: Vec<ToolOutput>,
    /// Whether the tool failed. The output then describes the failure.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub is_error: bool,
}

impl ToolResult {
    /// A successful result with one text output.
    pub fn text(
        call_id: impl Into<String>,
        name: impl Into<String>,
        text: impl Into<String>,
    ) -> Self {
        Self {
            call_id: call_id.into(),
            name: name.into(),
            output: vec![ToolOutput::Text(text.into())],
            is_error: false,
        }
    }

    /// A failed result describing the error.
    pub fn error(
        call_id: impl Into<String>,
        name: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            is_error: true,
            ..Self::text(call_id, name, message)
        }
    }

    /// The text outputs joined, with JSON outputs rendered as JSON.
    pub fn text_output(&self) -> String {
        self.output
            .iter()
            .filter_map(|o| match o {
                ToolOutput::Text(t) => Some(t.clone()),
                ToolOutput::Json(v) => Some(v.to_string()),
                ToolOutput::Image(_) => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// One piece of a tool's output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum ToolOutput {
    /// Text.
    Text(String),
    /// Structured data.
    Json(serde_json::Value),
    /// An image, for providers that accept images in tool results.
    Image(Image),
}

/// The model's reasoning.
///
/// Providers differ in what they expose: summarized text, a signature that
/// must be sent back verbatim, or an encrypted blob. A part keeps whatever the
/// provider sent so the next turn can return it unchanged.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Reasoning {
    /// The reasoning text or summary, possibly empty.
    pub text: String,
    /// A signature that authenticates the text (Anthropic, Gemini).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    /// Reasoning the provider returns only in encrypted form.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encrypted: Option<String>,
    /// Provider-specific data, such as an item id.
    #[serde(default, skip_serializing_if = "Extensions::is_empty")]
    pub extensions: Extensions,
}

/// A typed provider-specific value stored in [`Extensions`].
///
/// ```
/// use rig2_core::content::{Extension, Extensions};
/// use serde::{Deserialize, Serialize};
///
/// #[derive(Serialize, Deserialize, PartialEq, Debug)]
/// struct Detail(String);
/// impl Extension for Detail {
///     const KEY: &'static str = "example.detail";
/// }
///
/// let mut ext = Extensions::default();
/// ext.insert(&Detail("low".into())).unwrap();
/// assert_eq!(ext.get::<Detail>().unwrap(), Some(Detail("low".into())));
/// ```
pub trait Extension: Serialize + DeserializeOwned {
    /// The key, prefixed with the provider name: `openai.detail`.
    const KEY: &'static str;
}

/// Provider-specific values, keyed by provider-prefixed names.
///
/// Values are stored as JSON so that messages stay plain data, and read back
/// through the typed [`Extension`] that owns the key.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Extensions(BTreeMap<String, serde_json::Value>);

impl Extensions {
    /// Whether there are no values.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Store `value` under its key, replacing any earlier value.
    pub fn insert<E: Extension>(&mut self, value: &E) -> Result<()> {
        self.0
            .insert(E::KEY.to_owned(), serde_json::to_value(value)?);
        Ok(())
    }

    /// The value under `E`'s key, if present.
    ///
    /// Fails with [`ErrorKind::Decode`] when a value is present but does not
    /// decode as `E`.
    pub fn get<E: Extension>(&self) -> Result<Option<E>> {
        self.0
            .get(E::KEY)
            .map(|v| {
                serde_json::from_value(v.clone()).map_err(|e| {
                    Error::new(ErrorKind::Decode, format!("extension `{}`: {e}", E::KEY))
                })
            })
            .transpose()
    }

    /// Remove and return `E`'s value.
    pub fn remove<E: Extension>(&mut self) -> Result<Option<E>> {
        let value = self.get::<E>()?;
        self.0.remove(E::KEY);
        Ok(value)
    }
}

#[cfg(test)]
mod tests;
