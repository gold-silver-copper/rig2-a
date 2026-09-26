use super::*;

#[test]
fn a_response_error_takes_its_kind_message_and_retry_after_from_the_response() {
    let mut headers = http::HeaderMap::new();
    headers.insert("retry-after", "2".parse().unwrap());
    headers.insert("x-request-id", "req_123".parse().unwrap());
    headers.insert("set-cookie", "session=secret".parse().unwrap());
    let error = Error::from_response(429, r#"{"error":{"message":"slow down"}}"#, &headers);
    assert_eq!(error.kind(), ErrorKind::RateLimited);
    assert_eq!(error.message(), "slow down");
    assert_eq!(error.retry_after(), Some(Duration::from_secs(2)));
    assert_eq!(error.request_id(), Some("req_123"));
    assert!(error.is_retryable());
    assert!(error.headers().iter().all(|(name, _)| name != "set-cookie"));
}

#[test]
fn an_error_round_trips_through_json_keeping_its_source_message() {
    let io = std::io::Error::other("connection reset");
    let error = Error::new(ErrorKind::Transport, "send failed").with_source(io);
    let json = serde_json::to_string(&error).unwrap();
    let back: Error = serde_json::from_str(&json).unwrap();
    assert_eq!(back.kind(), ErrorKind::Transport);
    assert_eq!(
        std::error::Error::source(&back).unwrap().to_string(),
        "connection reset"
    );
}

#[test]
fn client_errors_are_not_retryable() {
    for status in [400, 401, 403, 404] {
        let error = Error::from_response(status, "", &http::HeaderMap::new());
        assert!(!error.is_retryable(), "{status}");
    }
}
