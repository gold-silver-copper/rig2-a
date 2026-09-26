use super::*;

#[test]
fn sse_joins_data_lines_ignores_comments_and_handles_split_chunks() {
    let mut parser = SseParser::default();
    let mut events =
        parser.push(b": keep-alive\nevent: delta\ndata: {\"a\":\ndata: 1}\n\ndata: tw");
    events.extend(parser.push(b"o\r\n\r\n"));
    events.extend(parser.finish());
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].event.as_deref(), Some("delta"));
    assert_eq!(events[0].data, "{\"a\":\n1}");
    assert_eq!(events[1].data, "two");
}

#[test]
fn sse_dispatches_a_final_event_without_a_trailing_blank_line() {
    let mut parser = SseParser::default();
    assert!(parser.push(b"data: [DONE]").is_empty());
    assert_eq!(parser.finish().unwrap().data, "[DONE]");
}

#[test]
fn ndjson_splits_lines_across_chunks() {
    let body: ByteStream = Box::pin(futures::stream::iter(
        [&b"{\"a\":1}\n{\"b\""[..], &b":2}\n\n{\"c\":3}"[..]]
            .map(|c| Ok(Bytes::copy_from_slice(c))),
    ));
    let values: Vec<_> = futures::executor::block_on_stream(ndjson(body))
        .map(Result::unwrap)
        .collect();
    assert_eq!(
        values,
        vec![
            serde_json::json!({"a":1}),
            serde_json::json!({"b":2}),
            serde_json::json!({"c":3})
        ]
    );
}

struct Status(u16);

impl HttpClient for Status {
    fn send(&self, _request: http::Request<Body>) -> crate::BoxFuture<'static, Result<Response>> {
        let status = self.0;
        Box::pin(async move {
            Ok(http::Response::builder()
                .status(status)
                .header("x-request-id", "req_1")
                .body(full_body(r#"{"error":{"message":"no such model"}}"#))
                .unwrap())
        })
    }
}

#[test]
fn send_turns_an_error_status_into_a_typed_error() {
    let request = get("https://example.invalid/v1/models", &[]).unwrap();
    let Err(error) = futures::executor::block_on(send(&Status(404), "demo", request)) else {
        panic!("expected an error")
    };
    assert_eq!(error.kind(), ErrorKind::NotFound);
    assert_eq!(error.message(), "no such model");
    assert_eq!(error.provider(), Some("demo"));
    assert_eq!(error.request_id(), Some("req_1"));
}
