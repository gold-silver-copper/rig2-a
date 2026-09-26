use rig2_core::completion::{Collect, OutputSchema, ToolDefinition, check_canonical};
use rig2_core::content::{File, Image, ToolResult};
use rig2_core::http::SseParser;

use super::*;
use crate::dialect::OPENAI;

fn events(decoder: &mut ChatDecoder, sse: &str) -> Vec<StreamEvent> {
    let mut parser = SseParser::default();
    let mut frames = parser.push(sse.as_bytes());
    frames.extend(parser.finish());
    let mut out = StreamWriter::default();
    let mut all = Vec::new();
    for frame in frames {
        let done = decoder.feed(frame, &mut out).unwrap();
        all.extend(out.drain());
        if let Some(finish) = done {
            all.extend(out.finish(finish).unwrap());
            return all;
        }
    }
    let finish = decoder.end(&mut out).unwrap();
    all.extend(out.drain());
    all.extend(out.finish(finish).unwrap());
    all
}

#[test]
fn the_request_maps_system_tools_results_images_and_limits() {
    let request = CompletionRequest::new(vec![
        Message::User {
            content: vec![
                UserContent::Text(Text::new("look")),
                UserContent::Image(Image::from_bytes(vec![1, 2, 3], "image/png")),
                UserContent::File(File {
                    source: Source::Bytes(vec![4].into()),
                    media_type: "application/pdf".into(),
                    name: Some("a.pdf".into()),
                    extensions: Default::default(),
                }),
            ],
        },
        Message::Assistant {
            content: vec![AssistantContent::ToolCall(
                rig2_core::content::ToolCall::new("call_1", "lookup", json!({"q": "x"})),
            )],
        },
        Message::tool_results([ToolResult::text("call_1", "lookup", "found")]),
    ])
    .with_system("be brief")
    .with_tool(ToolDefinition {
        name: "lookup".into(),
        description: "Look up.".into(),
        parameters: json!({"type": "object"}),
    })
    .with_tool_choice(ToolChoice::Required)
    .with_output(OutputSchema {
        name: "answer".into(),
        schema: json!({"type": "object"}),
    })
    .with_max_tokens(100)
    .with_reasoning(ReasoningEffort::Minimal);
    let body = chat_request_body(&OPENAI, "gpt-5-mini", &request, true).unwrap();
    assert_eq!(
        body["messages"][0],
        json!({"role": "system", "content": "be brief"})
    );
    assert_eq!(
        body["messages"][1]["content"][1]["image_url"]["url"],
        "data:image/png;base64,AQID"
    );
    assert_eq!(
        body["messages"][1]["content"][2]["file"]["filename"],
        "a.pdf"
    );
    assert_eq!(
        body["messages"][2]["tool_calls"][0]["function"]["arguments"],
        "{\"q\":\"x\"}"
    );
    assert_eq!(
        body["messages"][3],
        json!({"role": "tool", "tool_call_id": "call_1", "content": "found"})
    );
    assert_eq!(body["tool_choice"], "required");
    assert_eq!(body["response_format"]["json_schema"]["name"], "answer");
    assert_eq!(body["max_completion_tokens"], 100);
    assert_eq!(body["reasoning_effort"], "minimal");
    assert_eq!(body["stream_options"]["include_usage"], true);
}

#[test]
fn an_image_in_a_tool_result_is_unsupported() {
    let result = ToolResult {
        call_id: "1".into(),
        name: "t".into(),
        output: vec![ToolOutput::Image(Image::from_url(
            "https://example.com/a.png",
        ))],
        is_error: false,
    };
    let request = CompletionRequest::new(vec![Message::tool_results([result])]);
    let error = chat_request_body(&OPENAI, "m", &request, false).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unsupported);
}

const STREAM: &str = r#"data: {"id":"c1","model":"m","created":1,"choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}

data: {"id":"c1","choices":[{"index":0,"delta":{"content":"lo"}}]}

data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"get_weather","arguments":"{\"ci"}}]}}]}

data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_b","function":{"name":"get_weather","arguments":"{\"city\":\"Tokyo\"}"}}]}}]}

data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ty\":\"Paris\"}"}}]}}]}

data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}

data: {"id":"c1","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":7,"prompt_tokens_details":{"cached_tokens":2}}}

data: [DONE]

"#;

#[test]
fn a_stream_decodes_to_canonical_events_and_a_unary_shaped_document() {
    let mut decoder = ChatDecoder::new("openai", None, true);
    let events = events(&mut decoder, STREAM);
    check_canonical(&events).unwrap();
    let mut collect = Collect::default();
    for event in events {
        collect.push(event);
    }
    let response = collect.finish().unwrap();
    assert_eq!(response.text(), "Hello");
    let calls = response.tool_calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].arguments, json!({"city": "Paris"}));
    assert_eq!(calls[1].arguments, json!({"city": "Tokyo"}));
    assert_eq!(response.finish_reason, FinishReason::ToolCalls);
    assert_eq!(response.usage.input_tokens, 10);
    assert_eq!(response.usage.cached_input_tokens, 2);

    // The assembled document parses exactly like a unary reply.
    let unary = respond(|out| write_reply("openai", &response.raw, None, out)).unwrap();
    assert_eq!(unary.content, response.content);
    assert_eq!(unary.usage, response.usage);
}

#[test]
fn a_stream_cut_before_its_finish_reason_is_an_error() {
    let mut decoder = ChatDecoder::new("openai", None, true);
    let mut parser = SseParser::default();
    let mut out = StreamWriter::default();
    for frame in parser.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"x\"}}]}\n\n") {
        decoder.feed(frame, &mut out).unwrap();
    }
    assert_eq!(decoder.end(&mut out).unwrap_err().kind(), ErrorKind::Decode);
}

#[test]
fn an_error_inside_a_stream_is_typed() {
    let mut decoder = ChatDecoder::new("openai", None, true);
    let frame = SseEvent {
        data: r#"{"error":{"message":"slow down","type":"rate_limit_exceeded"}}"#.into(),
        ..SseEvent::default()
    };
    let error = decoder
        .feed(frame, &mut StreamWriter::default())
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::RateLimited);
}

#[test]
fn a_unary_refusal_is_reported_as_one() {
    let doc = json!({"choices":[{"message":{"content":null,"refusal":"I can't help with that."},"finish_reason":"stop"}]});
    let response = respond(|out| write_reply("openai", &doc, None, out)).unwrap();
    assert_eq!(response.finish_reason, FinishReason::Refusal);
    assert_eq!(response.text(), "I can't help with that.");
}
