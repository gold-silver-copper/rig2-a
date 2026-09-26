use rig2_core::completion::{Collect, ReasoningEffort, ToolDefinition, check_canonical};
use rig2_core::content::{ToolCall, ToolResult};
use rig2_core::http::SseParser;

use super::*;

#[test]
fn the_request_marks_the_cache_prefix_and_budgets_thinking() {
    let request = CompletionRequest::new(vec![
        Message::user("hi"),
        Message::Assistant {
            content: vec![
                AssistantContent::Reasoning(Reasoning {
                    text: "t".into(),
                    signature: Some("sig".into()),
                    ..Reasoning::default()
                }),
                AssistantContent::ToolCall(ToolCall::new("tu_1", "f", json!({"a": 1}))),
            ],
        },
        Message::tool_results([ToolResult::text("tu_1", "f", "ok")]),
    ])
    .with_system("sys")
    .with_tool(ToolDefinition {
        name: "f".into(),
        description: "F.".into(),
        parameters: json!({"type": "object"}),
    })
    .with_tool_choice(ToolChoice::Required)
    .with_cache(CacheHint::Prefix)
    .with_reasoning(ReasoningEffort::Low)
    .with_max_tokens(100);
    let (body, betas) =
        request_body("claude-x", &ModelCard::unknown("claude-x"), &request, false).unwrap();
    assert!(betas.is_empty());
    assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
    assert_eq!(body["tools"][0]["cache_control"]["type"], "ephemeral");
    assert_eq!(
        body["messages"][2]["content"][0]["cache_control"]["type"],
        "ephemeral"
    );
    assert_eq!(
        body["messages"][1]["content"][0],
        json!({"type": "thinking", "thinking": "t", "signature": "sig"})
    );
    assert_eq!(body["tool_choice"], json!({"type": "any"}));
    assert_eq!(body["thinking"]["budget_tokens"], 2048);
    assert_eq!(body["max_tokens"], 2048 + 1024);
}

const STREAM: &str = r#"event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-x","content":[],"stop_reason":null,"usage":{"input_tokens":12,"cache_read_input_tokens":3,"output_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Hmm."}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"SIG"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}

event: ping
data: {"type":"ping"}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Checking."}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: content_block_start
data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"city\": "}}

event: content_block_delta
data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"\"Paris\"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":2}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":40}}

event: message_stop
data: {"type":"message_stop"}

"#;

#[test]
fn a_stream_decodes_to_the_same_response_as_its_assembled_message() {
    let mut decoder = MessagesDecoder::new("anthropic", None);
    let mut parser = SseParser::default();
    let mut out = StreamWriter::default();
    let mut events = Vec::new();
    let mut finish = None;
    for frame in parser.push(STREAM.as_bytes()) {
        let done = decoder.feed(frame, &mut out).unwrap();
        events.extend(out.drain());
        if done.is_some() {
            finish = done;
            break;
        }
    }
    events.extend(out.finish(finish.unwrap()).unwrap());
    check_canonical(&events).unwrap();
    let mut collect = Collect::default();
    for event in events {
        collect.push(event);
    }
    let streamed = collect.finish().unwrap();
    assert_eq!(streamed.text(), "Checking.");
    assert_eq!(streamed.reasoning()[0].signature.as_deref(), Some("SIG"));
    assert_eq!(streamed.tool_calls()[0].arguments, json!({"city": "Paris"}));
    assert_eq!(streamed.finish_reason, FinishReason::ToolCalls);
    assert_eq!(streamed.usage.input_tokens, 15);
    assert_eq!(streamed.usage.output_tokens, 40);

    let unary = respond(|out| write_reply("anthropic", &streamed.raw, None, out)).unwrap();
    assert_eq!(unary.content, streamed.content);
    assert_eq!(unary.usage, streamed.usage);
}

#[test]
fn an_overloaded_error_event_is_retryable() {
    let mut decoder = MessagesDecoder::new("anthropic", None);
    let frame = SseEvent {
        data: r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#
            .into(),
        ..SseEvent::default()
    };
    let error = decoder
        .feed(frame, &mut StreamWriter::default())
        .unwrap_err();
    assert!(error.is_retryable());
}
