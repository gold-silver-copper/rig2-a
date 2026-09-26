//! The one error type for every model call, transport and store.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A result whose error is [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// What went wrong, independent of the provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ErrorKind {
    /// The request never got a response: DNS, TLS, connection or I/O failure.
    Transport,
    /// The request or stream took longer than allowed.
    Timeout,
    /// The credentials were missing, wrong or expired (HTTP 401).
    Auth,
    /// The credentials are valid but lack access (HTTP 403).
    PermissionDenied,
    /// The model or resource does not exist (HTTP 404).
    NotFound,
    /// The provider throttled the request (HTTP 429).
    RateLimited,
    /// The provider rejected the request as malformed (HTTP 400, 422).
    InvalidRequest,
    /// The provider is down or overloaded (HTTP 5xx, 529).
    Unavailable,
    /// The provider's response could not be decoded.
    Decode,
    /// A tool call's arguments were not valid JSON.
    MalformedToolInput,
    /// A model's output did not match the requested schema.
    InvalidOutput,
    /// The provider, model or backend does not support what was asked.
    Unsupported,
    /// The client is misconfigured, for example a missing API key.
    Config,
    /// The operation was cancelled.
    Cancelled,
    /// A limit was reached: turns, tokens or cost.
    Limit,
    /// Anything else.
    Other,
}

impl ErrorKind {
    /// The kind for an HTTP error status.
    pub fn from_status(status: u16) -> Self {
        match status {
            400 | 413 | 422 => Self::InvalidRequest,
            401 => Self::Auth,
            403 => Self::PermissionDenied,
            404 => Self::NotFound,
            408 => Self::Timeout,
            429 => Self::RateLimited,
            500..=599 => Self::Unavailable,
            _ => Self::Other,
        }
    }
}

/// An error from a model call, a transport or a store.
///
/// It carries the provider's HTTP status, body, request id and headers when
/// there was a response, so a caller can see exactly what the provider said.
/// It serializes for recordings; the source error is kept as its message.
#[derive(Clone)]
pub struct Error(Box<Inner>);

#[derive(Clone, Serialize, Deserialize)]
struct Inner {
    kind: ErrorKind,
    message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    status: Option<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    body: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    request_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    headers: Vec<(String, String)>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    retry_after: Option<Duration>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        serialize_with = "ser_source",
        deserialize_with = "de_source"
    )]
    source: Option<Source>,
}

#[cfg(not(target_family = "wasm"))]
type Source = Arc<dyn std::error::Error + Send + Sync>;
#[cfg(target_family = "wasm")]
type Source = Arc<dyn std::error::Error>;

impl Error {
    /// An error of `kind` with a message.
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self(Box::new(Inner {
            kind,
            message: message.into(),
            provider: None,
            status: None,
            body: None,
            request_id: None,
            headers: Vec::new(),
            retry_after: None,
            source: None,
        }))
    }

    /// An error for an HTTP error response.
    ///
    /// The kind follows the status. `headers` should already be free of
    /// secrets; `retry_after` is read from them when present.
    pub fn from_response(status: u16, body: impl Into<String>, headers: &http::HeaderMap) -> Self {
        let body = body.into();
        let message = extract_message(&body).unwrap_or_else(|| format!("HTTP {status}"));
        // A 429 whose limit is zero is not throttling: the account has no
        // quota for this model, and waiting will not help.
        let no_quota = status == 429
            && headers.iter().any(|(name, value)| {
                name.as_str().starts_with("x-ratelimit-limit") && value.as_bytes() == b"0"
            });
        let kind = if no_quota {
            ErrorKind::PermissionDenied
        } else {
            kind_from_body(&body).unwrap_or_else(|| ErrorKind::from_status(status))
        };
        let mut error = Self::new(kind, message);
        error.0.status = Some(status);
        error.0.body = (!body.is_empty()).then_some(body);
        error.0.retry_after = retry_after(headers);
        error.0.request_id = ["x-request-id", "request-id", "x-amzn-requestid", "cf-ray"]
            .iter()
            .find_map(|name| headers.get(*name)?.to_str().ok().map(str::to_owned));
        error.0.headers = headers
            .iter()
            .filter(|(name, _)| !is_sensitive_header(name.as_str()))
            .filter_map(|(name, value)| {
                Some((name.as_str().to_owned(), value.to_str().ok()?.to_owned()))
            })
            .collect();
        error
    }

    /// Attach the error that caused this one.
    #[cfg(not(target_family = "wasm"))]
    pub fn with_source(mut self, source: impl std::error::Error + Send + Sync + 'static) -> Self {
        self.0.source = Some(Arc::new(source));
        self
    }

    /// Attach the error that caused this one.
    #[cfg(target_family = "wasm")]
    pub fn with_source(mut self, source: impl std::error::Error + 'static) -> Self {
        self.0.source = Some(Arc::new(source));
        self
    }

    /// Name the provider the error came from.
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.0.provider = Some(provider.into());
        self
    }

    /// Set how long the provider asked the caller to wait.
    pub fn with_retry_after(mut self, delay: Duration) -> Self {
        self.0.retry_after = Some(delay);
        self
    }

    /// What went wrong.
    pub fn kind(&self) -> ErrorKind {
        self.0.kind
    }

    /// A human-readable description, the provider's own when there is one.
    pub fn message(&self) -> &str {
        &self.0.message
    }

    /// The provider the error came from, when known.
    pub fn provider(&self) -> Option<&str> {
        self.0.provider.as_deref()
    }

    /// The HTTP status, when the provider responded.
    pub fn status(&self) -> Option<u16> {
        self.0.status
    }

    /// The response body, when the provider responded with one.
    pub fn body(&self) -> Option<&str> {
        self.0.body.as_deref()
    }

    /// The provider's request id, when it sent one.
    pub fn request_id(&self) -> Option<&str> {
        self.0.request_id.as_deref()
    }

    /// The response headers, without credentials or cookies.
    pub fn headers(&self) -> &[(String, String)] {
        &self.0.headers
    }

    /// How long the provider asked the caller to wait before retrying.
    pub fn retry_after(&self) -> Option<Duration> {
        self.0.retry_after
    }

    /// Whether repeating the same request may succeed.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self.0.kind,
            ErrorKind::Transport
                | ErrorKind::Timeout
                | ErrorKind::RateLimited
                | ErrorKind::Unavailable
        )
    }
}

fn is_sensitive_header(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    matches!(
        name.as_str(),
        "authorization"
            | "proxy-authorization"
            | "cookie"
            | "set-cookie"
            | "x-api-key"
            | "api-key"
            | "x-goog-api-key"
            | "openai-organization"
            | "openai-project"
            | "anthropic-organization-id"
    ) || name.contains("token")
        || name.contains("secret")
}

fn retry_after(headers: &http::HeaderMap) -> Option<Duration> {
    if let Some(ms) = headers
        .get("retry-after-ms")
        .and_then(|v| v.to_str().ok()?.parse::<u64>().ok())
    {
        return Some(Duration::from_millis(ms));
    }
    let value = headers.get(http::header::RETRY_AFTER)?.to_str().ok()?;
    value
        .trim()
        .parse::<f64>()
        .ok()
        .filter(|s| *s >= 0.0)
        .map(Duration::from_secs_f64)
}

/// A kind named by the body, which some providers use instead of the HTTP
/// status: Google's `status` codes, and its `API_KEY_INVALID` reason, which
/// arrives with HTTP 400.
fn kind_from_body(body: &str) -> Option<ErrorKind> {
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    let error = value.get("error")?;
    let invalid_key = error
        .get("details")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .any(|d| d.get("reason").and_then(serde_json::Value::as_str) == Some("API_KEY_INVALID"));
    if invalid_key {
        return Some(ErrorKind::Auth);
    }
    match error.get("status").and_then(serde_json::Value::as_str)? {
        "UNAUTHENTICATED" => Some(ErrorKind::Auth),
        "PERMISSION_DENIED" => Some(ErrorKind::PermissionDenied),
        "NOT_FOUND" => Some(ErrorKind::NotFound),
        "RESOURCE_EXHAUSTED" => Some(ErrorKind::RateLimited),
        "UNAVAILABLE" | "INTERNAL" => Some(ErrorKind::Unavailable),
        "DEADLINE_EXCEEDED" => Some(ErrorKind::Timeout),
        _ => None,
    }
}

/// The `message` field most providers put in an error body.
fn extract_message(body: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    let candidates = [
        value.pointer("/error/message"),
        value.pointer("/message"),
        value.pointer("/error"),
        value.pointer("/detail"),
    ];
    candidates
        .into_iter()
        .flatten()
        .find_map(|v| v.as_str().map(str::to_owned))
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut s = f.debug_struct("Error");
        s.field("kind", &self.0.kind)
            .field("message", &self.0.message);
        if let Some(provider) = &self.0.provider {
            s.field("provider", provider);
        }
        if let Some(status) = self.0.status {
            s.field("status", &status);
        }
        if let Some(request_id) = &self.0.request_id {
            s.field("request_id", request_id);
        }
        if let Some(source) = &self.0.source {
            s.field("source", &source.to_string());
        }
        s.finish_non_exhaustive()
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(provider) = &self.0.provider {
            write!(f, "{provider}: ")?;
        }
        write!(f, "{}", self.0.message)?;
        if let Some(status) = self.0.status {
            write!(f, " (HTTP {status})")?;
        }
        Ok(())
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.0
            .source
            .as_deref()
            .map(|s| s as &(dyn std::error::Error + 'static))
    }
}

/// Errors are equal when their recorded forms are equal.
impl PartialEq for Error {
    fn eq(&self, other: &Self) -> bool {
        serde_json::to_value(self).ok() == serde_json::to_value(other).ok()
    }
}

impl Serialize for Error {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Error {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Inner::deserialize(deserializer).map(|inner| Self(Box::new(inner)))
    }
}

/// A source error restored from a recording: only its message survives.
#[derive(Debug)]
struct RecordedSource(String);

impl fmt::Display for RecordedSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for RecordedSource {}

#[expect(
    clippy::ref_option,
    reason = "serde's `serialize_with` passes `&Option`"
)]
fn ser_source<S: Serializer>(source: &Option<Source>, serializer: S) -> Result<S::Ok, S::Error> {
    match source {
        Some(source) => serializer.serialize_some(&source.to_string()),
        None => serializer.serialize_none(),
    }
}

fn de_source<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<Source>, D::Error> {
    let message = Option::<String>::deserialize(deserializer)?;
    Ok(message.map(|m| Arc::new(RecordedSource(m)) as Source))
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Self::new(ErrorKind::Decode, error.to_string()).with_source(error)
    }
}

#[cfg(test)]
mod tests;
