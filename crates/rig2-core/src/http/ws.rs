use std::pin::Pin;

use bytes::Bytes;
use futures::Sink;

use crate::{BoxFuture, BoxStream, Error, MaybeSend, MaybeSync, Result};

/// A WebSocket message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WsMessage {
    /// A text frame.
    Text(String),
    /// A binary frame.
    Binary(Bytes),
    /// The peer is closing the connection.
    Close,
}

/// The sending half of a WebSocket.
#[cfg(not(target_family = "wasm"))]
pub type WsSink = Pin<Box<dyn Sink<WsMessage, Error = Error> + Send>>;
/// The sending half of a WebSocket.
#[cfg(target_family = "wasm")]
pub type WsSink = Pin<Box<dyn Sink<WsMessage, Error = Error>>>;

/// An open WebSocket: a sink for outgoing messages and a stream of incoming
/// ones. Pings are answered by the transport.
pub struct WebSocket {
    /// Outgoing messages.
    pub sink: WsSink,
    /// Incoming messages. The stream ends when the connection closes.
    pub stream: BoxStream<'static, Result<WsMessage>>,
}

impl std::fmt::Debug for WebSocket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebSocket").finish_non_exhaustive()
    }
}

/// Opens WebSocket connections: the WebSocket counterpart of
/// [`HttpClient`](super::HttpClient).
pub trait WebSocketClient: MaybeSend + MaybeSync {
    /// Connect with the method, URI and headers of `request`.
    fn connect(&self, request: http::Request<()>) -> BoxFuture<'static, Result<WebSocket>>;
}
