//! HTTP cassettes: record provider exchanges once, replay them in tests.
//!
//! A [`Cassette`] is an [`HttpClient`]. In replay mode it answers each
//! request from a JSON file, matching method, URI and body. In record mode
//! it sends requests to a real client and writes the exchanges, scrubbed of
//! secrets, when [`Cassette::finish`] is called. [`Cassette::open`] picks
//! the mode: it records only when `RIG2_LIVE=1` and the file is missing (or
//! `RIG2_RERECORD=1`), so reruns replay by default.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use bytes::Bytes;
use rig2_core::http::{Body, HttpClient, Response, SharedClient, full_body, read_body};
use rig2_core::{BoxFuture, Error, ErrorKind, Result};
use serde::{Deserialize, Serialize};

use crate::scrub::{is_sensitive_header, scrub_text};

/// A body as stored: JSON when it parses, text when it is UTF-8, base64
/// otherwise.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoredBody {
    /// A JSON document.
    Json(serde_json::Value),
    /// Text, such as a server-sent event stream.
    Text(String),
    /// Anything else.
    Base64(String),
}

impl StoredBody {
    fn capture(bytes: &[u8]) -> Self {
        match std::str::from_utf8(bytes) {
            Ok(text) => {
                let text = scrub_text(text);
                match serde_json::from_str(&text) {
                    Ok(json) if !text.trim().is_empty() => Self::Json(json),
                    _ => Self::Text(text),
                }
            }
            Err(_) => Self::Base64(STANDARD.encode(bytes)),
        }
    }

    fn bytes(&self) -> Result<Bytes> {
        Ok(match self {
            Self::Json(json) => Bytes::from(serde_json::to_vec(json)?),
            Self::Text(text) => Bytes::from(text.clone()),
            Self::Base64(data) => {
                Bytes::from(STANDARD.decode(data).map_err(|e| {
                    Error::new(ErrorKind::Decode, format!("bad cassette body: {e}"))
                })?)
            }
        })
    }
}

/// One recorded request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredRequest {
    /// The method.
    pub method: String,
    /// The URI.
    pub uri: String,
    /// The headers, without credentials.
    pub headers: Vec<(String, String)>,
    /// The body.
    pub body: StoredBody,
}

/// One recorded response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredResponse {
    /// The status code.
    pub status: u16,
    /// The headers, without cookies or account ids.
    pub headers: Vec<(String, String)>,
    /// The whole body.
    pub body: StoredBody,
}

/// A request and its response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Interaction {
    /// What was sent.
    pub request: StoredRequest,
    /// What came back.
    pub response: StoredResponse,
}

/// The file format: interactions in the order they happened.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Tape {
    /// The interactions.
    pub interactions: Vec<Interaction>,
}

fn headers(map: &http::HeaderMap) -> Vec<(String, String)> {
    map.iter()
        .filter(|(name, _)| !is_sensitive_header(name.as_str()))
        .filter_map(|(name, value)| {
            Some((name.as_str().to_owned(), scrub_text(value.to_str().ok()?)))
        })
        .collect()
}

fn capture_request(request: &http::Request<Body>) -> StoredRequest {
    StoredRequest {
        method: request.method().to_string(),
        uri: scrub_text(&request.uri().to_string()),
        headers: headers(request.headers()),
        body: StoredBody::capture(request.body()),
    }
}

enum Mode {
    Replay { tape: Option<Tape>, used: Vec<bool> },
    Record { inner: SharedClient, tape: Tape },
}

impl std::fmt::Debug for Mode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Replay { .. } => "Replay",
            Self::Record { .. } => "Record",
        })
    }
}

/// An [`HttpClient`] that records to, or replays from, a JSON file.
#[derive(Debug, Clone)]
pub struct Cassette {
    path: PathBuf,
    mode: Arc<Mutex<Mode>>,
}

/// Whether live recording is switched on (`RIG2_LIVE=1`).
pub fn live() -> bool {
    std::env::var("RIG2_LIVE").is_ok_and(|v| v == "1")
}

impl Cassette {
    /// Replay `path`, or record it when live and missing (or re-recording).
    pub fn open(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref().to_owned();
        let rerecord = std::env::var("RIG2_RERECORD").is_ok_and(|v| v == "1");
        if live() && (rerecord || !path.exists()) {
            Self::record(path, Arc::new(rig2_reqwest::ReqwestClient::default()))
        } else {
            Self::replay(path)
        }
    }

    /// Replay `path`. A missing or unreadable file makes every send fail.
    pub fn replay(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref().to_owned();
        let tape = std::fs::read(&path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Tape>(&bytes).ok());
        let used = vec![false; tape.as_ref().map_or(0, |t| t.interactions.len())];
        Self {
            path,
            mode: Arc::new(Mutex::new(Mode::Replay { tape, used })),
        }
    }

    /// Send through `inner` and record to `path`.
    pub fn record(path: impl AsRef<Path>, inner: SharedClient) -> Self {
        Self {
            path: path.as_ref().to_owned(),
            mode: Arc::new(Mutex::new(Mode::Record {
                inner,
                tape: Tape::default(),
            })),
        }
    }

    /// Whether this cassette records.
    pub fn is_recording(&self) -> bool {
        matches!(
            *self.mode.lock().unwrap_or_else(PoisonError::into_inner),
            Mode::Record { .. }
        )
    }

    /// Whether this cassette has something to replay, or records.
    pub fn is_available(&self) -> bool {
        match &*self.mode.lock().unwrap_or_else(PoisonError::into_inner) {
            Mode::Replay { tape, .. } => tape.is_some(),
            Mode::Record { .. } => true,
        }
    }

    /// Write the recording, when recording. Replaying writes nothing.
    pub fn finish(&self) -> std::io::Result<()> {
        let mode = self.mode.lock().unwrap_or_else(PoisonError::into_inner);
        if let Mode::Record { tape, .. } = &*mode {
            if let Some(dir) = self.path.parent() {
                std::fs::create_dir_all(dir)?;
            }
            let mut json = serde_json::to_string_pretty(tape).map_err(std::io::Error::other)?;
            json.push('\n');
            std::fs::write(&self.path, json)?;
        }
        Ok(())
    }

    fn replay_one(&self, request: &StoredRequest) -> Result<Response> {
        let mut mode = self.mode.lock().unwrap_or_else(PoisonError::into_inner);
        let Mode::Replay { tape, used } = &mut *mode else {
            return Err(Error::new(ErrorKind::Other, "not replaying"));
        };
        let Some(tape) = tape else {
            return Err(Error::new(
                ErrorKind::NotFound,
                format!(
                    "no cassette at {}; record it with RIG2_LIVE=1",
                    self.path.display()
                ),
            ));
        };
        let found = tape
            .interactions
            .iter()
            .enumerate()
            .find(|(i, interaction)| {
                !used[*i]
                    && interaction.request.method == request.method
                    && interaction.request.uri == request.uri
                    && interaction.request.body == request.body
            });
        let Some((index, interaction)) = found else {
            return Err(Error::new(
                ErrorKind::NotFound,
                format!(
                    "{} has no unused interaction for {} {} with this body; re-record with RIG2_LIVE=1 RIG2_RERECORD=1",
                    self.path.display(),
                    request.method,
                    request.uri
                ),
            ));
        };
        used[index] = true;
        let mut builder = http::Response::builder().status(interaction.response.status);
        for (name, value) in &interaction.response.headers {
            builder = builder.header(name, value);
        }
        builder
            .body(full_body(interaction.response.body.bytes()?))
            .map_err(|e| Error::new(ErrorKind::Decode, format!("bad cassette response: {e}")))
    }
}

impl HttpClient for Cassette {
    fn send(&self, request: http::Request<Body>) -> BoxFuture<'static, Result<Response>> {
        let stored = capture_request(&request);
        let inner = match &*self.mode.lock().unwrap_or_else(PoisonError::into_inner) {
            Mode::Record { inner, .. } => Some(Arc::clone(inner)),
            Mode::Replay { .. } => None,
        };
        let Some(inner) = inner else {
            return Box::pin(std::future::ready(self.replay_one(&stored)));
        };
        let mode = Arc::clone(&self.mode);
        Box::pin(async move {
            let response = inner.send(request).await?;
            let (parts, body) = response.into_parts();
            let body = read_body(body).await?;
            let recorded = StoredResponse {
                status: parts.status.as_u16(),
                headers: headers(&parts.headers),
                body: StoredBody::capture(&body),
            };
            if let Mode::Record { tape, .. } =
                &mut *mode.lock().unwrap_or_else(PoisonError::into_inner)
            {
                tape.interactions.push(Interaction {
                    request: stored,
                    response: recorded,
                });
            }
            Ok(http::Response::from_parts(parts, full_body(body)))
        })
    }
}

#[cfg(test)]
mod tests;
