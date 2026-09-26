//! The reqwest HTTP client for rig2.
//!
//! [`ReqwestClient`] implements [`HttpClient`] over a `reqwest::Client`.
//! Its `Default` is one process-wide client, built on first use and shared
//! by every clone, so that every provider shares one connection pool. If the
//! client cannot be built, construction still succeeds and every send
//! reports the build error.
//!
//! ```no_run
//! use rig2_core::http::HttpClient;
//! use rig2_reqwest::ReqwestClient;
//!
//! let shared = ReqwestClient::default();
//! let custom = ReqwestClient::from(reqwest::Client::builder().user_agent("my-app").build().unwrap());
//! # let _ = (shared, custom);
//! ```

use std::sync::{Arc, OnceLock};

use bytes::Bytes;
use futures::TryStreamExt;
use rig2_core::http::{Body, HttpClient, Response};
use rig2_core::{BoxFuture, Error, ErrorKind, Result};

/// An [`HttpClient`] backed by reqwest.
#[derive(Debug, Clone)]
pub struct ReqwestClient {
    client: Result<reqwest::Client, Arc<str>>,
}

impl Default for ReqwestClient {
    /// The process-wide shared client.
    fn default() -> Self {
        static SHARED: OnceLock<Result<reqwest::Client, Arc<str>>> = OnceLock::new();
        let client = SHARED.get_or_init(|| {
            reqwest::Client::builder()
                .build()
                .map_err(|e| Arc::from(e.to_string()))
        });
        Self {
            client: client.clone(),
        }
    }
}

impl From<reqwest::Client> for ReqwestClient {
    fn from(client: reqwest::Client) -> Self {
        Self { client: Ok(client) }
    }
}

fn transport(error: &reqwest::Error) -> Error {
    let kind = if error.is_timeout() {
        ErrorKind::Timeout
    } else {
        ErrorKind::Transport
    };
    Error::new(kind, error.to_string())
}

impl HttpClient for ReqwestClient {
    fn send(&self, request: http::Request<Body>) -> BoxFuture<'static, Result<Response>> {
        let client = self.client.clone();
        Box::pin(async move {
            let client = client.map_err(|e| {
                Error::new(
                    ErrorKind::Transport,
                    format!("the HTTP client could not be built: {e}"),
                )
            })?;
            let request = reqwest::Request::try_from(request.map(reqwest::Body::from))
                .map_err(|e| Error::new(ErrorKind::InvalidRequest, e.to_string()))?;
            let response = client.execute(request).await.map_err(|e| transport(&e))?;
            let mut builder = http::Response::builder().status(response.status().as_u16());
            if let Some(headers) = builder.headers_mut() {
                headers.extend(
                    response
                        .headers()
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone())),
                );
            }
            let body = response
                .bytes_stream()
                .map_err(|e| transport(&e))
                .map_ok(Bytes::from);
            builder
                .body(Box::pin(body) as rig2_core::http::ByteStream)
                .map_err(|e| Error::new(ErrorKind::Decode, e.to_string()))
        })
    }
}

#[cfg(test)]
mod tests;
