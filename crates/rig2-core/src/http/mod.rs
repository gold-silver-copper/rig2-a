//! The transport contract and the helpers providers use over it.
//!
//! [`HttpClient`] is the one thing a transport implements: send a request,
//! get a response whose body is a byte stream. Middleware is an `HttpClient`
//! that wraps another. Providers build requests and parse responses as plain
//! values, and use [`send`], [`read_json`], [`sse`], [`ndjson`] and
//! [`Multipart`] to talk to the client.
//!
//! ```
//! use rig2_core::http::{Body, HttpClient, Response};
//! use rig2_core::{BoxFuture, Result};
//!
//! /// Answers every request with an empty 204.
//! struct NoContent;
//!
//! impl HttpClient for NoContent {
//!     fn send(&self, _request: http::Request<Body>) -> BoxFuture<'static, Result<Response>> {
//!         Box::pin(async { Ok(http::Response::builder().status(204).body(rig2_core::http::empty_body()).unwrap()) })
//!     }
//! }
//! ```

mod multipart;
mod sse;
mod ws;

use std::sync::Arc;

use bytes::{Bytes, BytesMut};
use futures::StreamExt;
use serde::de::DeserializeOwned;

pub use multipart::Multipart;
pub use sse::{SseEvent, SseParser, sse};
pub use ws::{WebSocket, WebSocketClient, WsMessage, WsSink};

use crate::{BoxFuture, BoxStream, Error, ErrorKind, MaybeSend, MaybeSync, Result};

/// A request body: the whole payload.
pub type Body = Bytes;

/// A response body: chunks as they arrive.
pub type ByteStream = BoxStream<'static, Result<Bytes>>;

/// A response with a streaming body.
pub type Response = http::Response<ByteStream>;

/// Sends HTTP requests.
///
/// Implementations report transport failures as [`ErrorKind::Transport`] or
/// [`ErrorKind::Timeout`], and return every response, whatever its status:
/// mapping statuses to errors is the provider's job.
pub trait HttpClient: MaybeSend + MaybeSync {
    /// Send `request`. The future resolves when the response headers arrive.
    fn send(&self, request: http::Request<Body>) -> BoxFuture<'static, Result<Response>>;
}

impl<C: HttpClient + ?Sized> HttpClient for Arc<C> {
    fn send(&self, request: http::Request<Body>) -> BoxFuture<'static, Result<Response>> {
        (**self).send(request)
    }
}

/// A shared, type-erased client, as provider clients hold it.
pub type SharedClient = Arc<dyn HttpClient>;

/// An empty response body.
pub fn empty_body() -> ByteStream {
    Box::pin(futures::stream::empty())
}

/// A response body holding `bytes`.
pub fn full_body(bytes: impl Into<Bytes>) -> ByteStream {
    let bytes = bytes.into();
    Box::pin(futures::stream::once(async move { Ok(bytes) }))
}

/// Read a whole body.
pub async fn read_body(mut body: ByteStream) -> Result<Bytes> {
    let mut all = BytesMut::new();
    while let Some(chunk) = body.next().await {
        all.extend_from_slice(&chunk?);
    }
    Ok(all.freeze())
}

/// Send `request` and return the response if its status is a success.
///
/// A non-success status becomes an [`Error`] carrying the status, body,
/// request id and headers, tagged with `provider`.
pub async fn send(
    client: &dyn HttpClient,
    provider: &str,
    request: http::Request<Body>,
) -> Result<Response> {
    let response = client
        .send(request)
        .await
        .map_err(|e| e.with_provider(provider))?;
    if response.status().is_success() {
        return Ok(response);
    }
    let (parts, body) = response.into_parts();
    let body = read_body(body).await.unwrap_or_default();
    Err(Error::from_response(
        parts.status.as_u16(),
        String::from_utf8_lossy(&body),
        &parts.headers,
    )
    .with_provider(provider))
}

/// Read a whole response body as JSON, keeping the parsed document.
pub async fn read_json<T: DeserializeOwned>(
    response: Response,
    provider: &str,
) -> Result<(T, serde_json::Value)> {
    let body = read_body(response.into_body()).await?;
    let raw: serde_json::Value = serde_json::from_slice(&body).map_err(|e| {
        Error::new(ErrorKind::Decode, format!("response is not JSON: {e}")).with_provider(provider)
    })?;
    let parsed = serde_json::from_value(raw.clone()).map_err(|e| {
        Error::new(ErrorKind::Decode, format!("unexpected response: {e}")).with_provider(provider)
    })?;
    Ok((parsed, raw))
}

/// Build a JSON `POST` request with `headers`.
pub fn post_json(
    uri: &str,
    headers: &[(&str, &str)],
    body: &impl serde::Serialize,
) -> Result<http::Request<Body>> {
    let mut builder =
        http::Request::post(uri).header(http::header::CONTENT_TYPE, "application/json");
    for (name, value) in headers {
        builder = builder.header(*name, *value);
    }
    builder
        .body(Bytes::from(serde_json::to_vec(body)?))
        .map_err(invalid_request)
}

/// Build a `GET` request with `headers`.
pub fn get(uri: &str, headers: &[(&str, &str)]) -> Result<http::Request<Body>> {
    let mut builder = http::Request::get(uri);
    for (name, value) in headers {
        builder = builder.header(*name, *value);
    }
    builder.body(Bytes::new()).map_err(invalid_request)
}

/// Map an `http` builder error to an [`ErrorKind::InvalidRequest`] error.
pub fn invalid_request(error: http::Error) -> Error {
    Error::new(
        ErrorKind::InvalidRequest,
        format!("could not build the request: {error}"),
    )
    .with_source(error)
}

/// Split a byte stream into newline-delimited JSON values.
pub fn ndjson(body: ByteStream) -> BoxStream<'static, Result<serde_json::Value>> {
    let lines = LineSplitter::default();
    Box::pin(futures::stream::unfold(
        (
            body,
            lines,
            std::collections::VecDeque::<Vec<u8>>::new(),
            false,
        ),
        |(mut body, mut lines, mut ready, mut done)| async move {
            loop {
                if let Some(line) = ready.pop_front() {
                    let parsed = serde_json::from_slice::<serde_json::Value>(&line).map_err(|e| {
                        Error::new(ErrorKind::Decode, format!("bad NDJSON line: {e}"))
                    });
                    return Some((parsed, (body, lines, ready, done)));
                }
                if done {
                    return None;
                }
                match body.next().await {
                    Some(Ok(chunk)) => {
                        ready.extend(lines.push(&chunk).into_iter().filter(|l| !l.is_empty()));
                    }
                    Some(Err(error)) => {
                        done = true;
                        return Some((Err(error), (body, lines, ready, done)));
                    }
                    None => {
                        done = true;
                        ready.extend(lines.finish().into_iter().filter(|l| !l.is_empty()));
                    }
                }
            }
        },
    ))
}

/// Splits bytes into lines, keeping partial lines between chunks.
#[derive(Debug, Default)]
pub(crate) struct LineSplitter {
    partial: Vec<u8>,
}

impl LineSplitter {
    /// Complete lines in `chunk`, without their terminators.
    pub(crate) fn push(&mut self, chunk: &[u8]) -> Vec<Vec<u8>> {
        let mut lines = Vec::new();
        for &byte in chunk {
            if byte == b'\n' {
                let mut line = std::mem::take(&mut self.partial);
                if line.last() == Some(&b'\r') {
                    line.pop();
                }
                lines.push(line);
            } else {
                self.partial.push(byte);
            }
        }
        lines
    }

    /// The last line, if the input did not end with a newline.
    pub(crate) fn finish(&mut self) -> Option<Vec<u8>> {
        let mut line = std::mem::take(&mut self.partial);
        if line.last() == Some(&b'\r') {
            line.pop();
        }
        (!line.is_empty()).then_some(line)
    }
}

#[cfg(test)]
mod tests;
