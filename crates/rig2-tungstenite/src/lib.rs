//! The tokio-tungstenite WebSocket client for rig2.
//!
//! [`TungsteniteClient`] implements [`WebSocketClient`]. It needs a Tokio
//! runtime, and is native-only.
//!
//! ```no_run
//! # async fn demo() -> rig2_core::Result<()> {
//! use futures::{SinkExt, StreamExt};
//! use rig2_core::http::{WebSocketClient, WsMessage};
//! use rig2_tungstenite::TungsteniteClient;
//!
//! let request = http::Request::get("wss://echo.example/").body(()).unwrap();
//! let mut socket = TungsteniteClient.connect(request).await?;
//! socket.sink.send(WsMessage::Text("hello".into())).await?;
//! let reply = socket.stream.next().await;
//! # Ok(()) }
//! ```

use futures::{SinkExt, StreamExt};
use rig2_core::http::{WebSocket, WebSocketClient, WsMessage};
use rig2_core::{BoxFuture, Error, ErrorKind, Result};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::{self, Message};

/// A [`WebSocketClient`] backed by tokio-tungstenite.
#[derive(Debug, Clone, Copy, Default)]
pub struct TungsteniteClient;

fn transport(error: &tungstenite::Error) -> Error {
    match error {
        tungstenite::Error::Http(response) => {
            let body = response
                .body()
                .as_deref()
                .map(String::from_utf8_lossy)
                .unwrap_or_default();
            Error::from_response(response.status().as_u16(), body, response.headers())
        }
        other => Error::new(ErrorKind::Transport, other.to_string()),
    }
}

impl WebSocketClient for TungsteniteClient {
    fn connect(&self, request: http::Request<()>) -> BoxFuture<'static, Result<WebSocket>> {
        Box::pin(async move {
            // Start from tungstenite's handshake request, which carries the
            // WebSocket headers, and add the caller's headers.
            let mut handshake = request
                .uri()
                .clone()
                .into_client_request()
                .map_err(|e| transport(&e))?;
            handshake.headers_mut().extend(
                request
                    .headers()
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone())),
            );
            let (socket, _) = tokio_tungstenite::connect_async(handshake)
                .await
                .map_err(|e| transport(&e))?;
            let (sink, stream) = socket.split();
            let sink = sink
                .with(|message: WsMessage| {
                    std::future::ready(Ok::<_, tungstenite::Error>(match message {
                        WsMessage::Text(text) => Message::Text(text.into()),
                        WsMessage::Binary(bytes) => Message::Binary(bytes),
                        WsMessage::Close => Message::Close(None),
                    }))
                })
                .sink_map_err(|e| transport(&e));
            let stream = stream.filter_map(|message| {
                std::future::ready(match message {
                    Ok(Message::Text(text)) => Some(Ok(WsMessage::Text(text.to_string()))),
                    Ok(Message::Binary(bytes)) => Some(Ok(WsMessage::Binary(bytes))),
                    Ok(Message::Close(_)) => Some(Ok(WsMessage::Close)),
                    Ok(Message::Ping(_) | Message::Pong(_) | Message::Frame(_)) => None,
                    Err(error) => Some(Err(transport(&error))),
                })
            });
            Ok(WebSocket {
                sink: Box::pin(sink),
                stream: Box::pin(stream),
            })
        })
    }
}

#[cfg(test)]
mod tests;
