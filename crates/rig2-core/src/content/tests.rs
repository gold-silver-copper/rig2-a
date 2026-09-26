use super::*;

#[test]
fn media_bytes_serialize_as_base64_and_round_trip() {
    let message = Message::User {
        content: vec![UserContent::Image(Image::from_bytes(
            vec![0_u8, 1, 2, 255],
            "image/png",
        ))],
    };
    let json = serde_json::to_value(&message).unwrap();
    assert_eq!(json["content"][0]["source"]["bytes"], "AAEC/w==");
    let back: Message = serde_json::from_value(json).unwrap();
    assert_eq!(back, message);
}

#[test]
fn an_extension_of_the_wrong_shape_is_a_decode_error() {
    #[derive(Debug, Serialize, Deserialize)]
    struct Count(u32);
    impl Extension for Count {
        const KEY: &'static str = "test.count";
    }
    let mut ext = Extensions::default();
    ext.0
        .insert("test.count".into(), serde_json::json!("not a number"));
    assert_eq!(ext.get::<Count>().unwrap_err().kind(), ErrorKind::Decode);
}

#[test]
fn message_text_joins_only_text_parts() {
    let message = Message::Assistant {
        content: vec![
            AssistantContent::Text(Text::new("a")),
            AssistantContent::ToolCall(ToolCall::new("1", "t", serde_json::json!({}))),
            AssistantContent::Text(Text::new("b")),
        ],
    };
    assert_eq!(message.text(), "ab");
}
