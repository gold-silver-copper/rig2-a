use std::sync::Arc;

use rig2_core::http::{Body, HttpClient, Response, full_body, read_body};
use rig2_core::{BoxFuture, ErrorKind, Result};

use super::*;

/// Answers with the request body reversed, and a cookie.
struct Mirror;

impl HttpClient for Mirror {
    fn send(&self, request: http::Request<Body>) -> BoxFuture<'static, Result<Response>> {
        let mut body = request.body().to_vec();
        body.reverse();
        Box::pin(async move {
            Ok(http::Response::builder()
                .status(200)
                .header("set-cookie", "s=1")
                .body(full_body(body))
                .unwrap())
        })
    }
}

fn request(body: &str) -> http::Request<Body> {
    let key = format!("{}{}", "sk-", "x".repeat(30));
    http::Request::post("https://api.example.test/v1/echo")
        .header("authorization", format!("Bearer {key}"))
        .body(Bytes::from(body.to_owned()))
        .unwrap()
}

#[tokio::test]
async fn a_recording_replays_by_method_uri_and_body_without_credentials() {
    let path = std::env::temp_dir().join(format!("rig2-cassette-{}.json", std::process::id()));
    let recorder = Cassette::record(&path, Arc::new(Mirror));
    let live = recorder.send(request("abc")).await.unwrap();
    assert_eq!(read_body(live.into_body()).await.unwrap(), "cba");
    recorder.send(request("xyz")).await.unwrap();
    recorder.finish().unwrap();

    let written = std::fs::read_to_string(&path).unwrap();
    assert!(
        !written.contains("authorization") && !written.contains("set-cookie"),
        "{written}"
    );

    let replay = Cassette::replay(&path);
    let second = replay.send(request("xyz")).await.unwrap();
    assert_eq!(read_body(second.into_body()).await.unwrap(), "zyx");
    let first = replay.send(request("abc")).await.unwrap();
    assert_eq!(read_body(first.into_body()).await.unwrap(), "cba");
    let Err(error) = replay.send(request("abc")).await else {
        panic!("each interaction replays once")
    };
    assert_eq!(error.kind(), ErrorKind::NotFound);
    std::fs::remove_file(&path).unwrap();
}

#[tokio::test]
async fn a_missing_cassette_fails_with_a_hint() {
    let replay = Cassette::replay("/definitely/not/here.json");
    assert!(!replay.is_available());
    let Err(error) = replay.send(request("{}")).await else {
        panic!("expected an error")
    };
    assert!(error.message().contains("RIG2_LIVE=1"));
}
