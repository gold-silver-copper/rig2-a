use futures::{SinkExt, StreamExt};

use super::*;

#[tokio::test]
async fn it_sends_and_receives_through_an_echo_server() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut socket = tokio_tungstenite::accept_async(stream).await.unwrap();
        while let Some(Ok(message)) = socket.next().await {
            if message.is_text() || message.is_binary() {
                socket.send(message).await.unwrap();
            }
        }
    });
    let request = http::Request::get(format!("ws://{address}/"))
        .body(())
        .unwrap();
    let mut socket = TungsteniteClient.connect(request).await.unwrap();
    socket
        .sink
        .send(WsMessage::Text("ping".into()))
        .await
        .unwrap();
    assert_eq!(
        socket.stream.next().await.unwrap().unwrap(),
        WsMessage::Text("ping".into())
    );
}

#[tokio::test]
async fn a_refused_connection_is_a_transport_error() {
    let request = http::Request::get("ws://127.0.0.1:1/").body(()).unwrap();
    let error = TungsteniteClient.connect(request).await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Transport);
}
