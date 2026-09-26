use rig2_core::http::{HttpClient, read_body};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use super::*;

/// Serve one raw HTTP response to the first connection.
async fn serve_once(response: &'static str) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut buffer = [0_u8; 4096];
        let _ = socket.read(&mut buffer).await.unwrap();
        socket.write_all(response.as_bytes()).await.unwrap();
        socket.shutdown().await.unwrap();
    });
    format!("http://{address}/v1/test")
}

#[tokio::test]
async fn it_returns_every_status_with_headers_and_a_streamed_body() {
    let uri = serve_once(
        "HTTP/1.1 429 Too Many Requests\r\nretry-after: 3\r\ncontent-length: 5\r\n\r\nslow!",
    )
    .await;
    let client = ReqwestClient::default();
    let request = http::Request::post(&uri)
        .body(bytes::Bytes::from_static(b"{}"))
        .unwrap();
    let response = client.send(request).await.unwrap();
    assert_eq!(response.status(), 429);
    assert_eq!(response.headers()["retry-after"], "3");
    assert_eq!(read_body(response.into_body()).await.unwrap(), "slow!");
}

#[tokio::test]
async fn a_connection_failure_is_a_transport_error() {
    let client = ReqwestClient::default();
    let request = http::Request::get("http://127.0.0.1:1/")
        .body(bytes::Bytes::new())
        .unwrap();
    let Err(error) = client.send(request).await else {
        panic!("expected an error")
    };
    assert_eq!(error.kind(), ErrorKind::Transport);
}
