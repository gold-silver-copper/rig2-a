use rig2_core::completion::{Collect, check_canonical};
use rig2_core::content::{Reasoning, ToolCall, ToolResult};
use rig2_core::http::SseParser;

use super::*;

fn decode(sse: &str) -> Vec<StreamEvent> {
    let mut decoder = ResponsesDecoder::new("openai", None);
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
    panic!("the stream did not finish");
}

#[test]
fn a_conversation_with_reasoning_and_tools_maps_to_stateless_items() {
    let mut reasoning = Reasoning {
        text: "thinking".into(),
        encrypted: Some("enc".into()),
        ..Reasoning::default()
    };
    reasoning.extensions.insert(&ItemId("rs_1".into())).unwrap();
    let request = CompletionRequest::new(vec![
        Message::user("weather?"),
        Message::Assistant {
            content: vec![
                AssistantContent::Reasoning(reasoning),
                AssistantContent::ToolCall(ToolCall::new(
                    "call_1",
                    "get_weather",
                    json!({"city": "Paris"}),
                )),
            ],
        },
        Message::tool_results([ToolResult::text("call_1", "get_weather", "sunny")]),
    ])
    .with_system("brief");
    let body = request_body("gpt-5-mini", &request, false).unwrap();
    assert_eq!(body["store"], false);
    assert_eq!(body["instructions"], "brief");
    assert_eq!(
        body["input"][0]["content"][0],
        json!({"type": "input_text", "text": "weather?"})
    );
    assert_eq!(body["input"][1]["type"], "reasoning");
    assert_eq!(body["input"][1]["id"], "rs_1");
    assert_eq!(body["input"][1]["encrypted_content"], "enc");
    assert_eq!(body["input"][2]["arguments"], "{\"city\":\"Paris\"}");
    assert_eq!(
        body["input"][3],
        json!({"type": "function_call_output", "call_id": "call_1", "output": "sunny"})
    );
}

#[test]
fn stop_sequences_are_unsupported() {
    let mut request = CompletionRequest::from("x");
    request.stop = vec!["END".into()];
    assert_eq!(
        request_body("m", &request, false).unwrap_err().kind(),
        ErrorKind::Unsupported
    );
}

const STREAM: &str = r#"event: response.created
data: {"type":"response.created","response":{"id":"resp_1","status":"in_progress"}}

event: response.output_item.added
data: {"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs_1","summary":[]}}

event: response.reasoning_summary_text.delta
data: {"type":"response.reasoning_summary_text.delta","output_index":0,"summary_index":0,"delta":"Think."}

event: response.output_item.done
data: {"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"Think."}],"encrypted_content":"enc"}}

event: response.output_item.added
data: {"type":"response.output_item.added","output_index":1,"item":{"type":"message","role":"assistant","content":[]}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","output_index":1,"content_index":0,"delta":"Hi "}

event: response.output_text.delta
data: {"type":"response.output_text.delta","output_index":1,"content_index":0,"delta":"there"}

event: response.output_item.done
data: {"type":"response.output_item.done","output_index":1,"item":{"type":"message","content":[{"type":"output_text","text":"Hi there","annotations":[{"type":"url_citation","url":"https://example.com","title":"Ex"}]}]}}

event: response.output_item.added
data: {"type":"response.output_item.added","output_index":2,"item":{"type":"function_call","call_id":"call_1","name":"get_weather","arguments":""}}

event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","output_index":2,"delta":"{\"city\":"}

event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","output_index":2,"delta":"\"Paris\"}"}

event: response.output_item.done
data: {"type":"response.output_item.done","output_index":2,"item":{"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}

event: response.completed
data: {"type":"response.completed","response":{"id":"resp_1","model":"gpt-5-mini","status":"completed","output":[{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"Think."}],"encrypted_content":"enc"},{"type":"message","content":[{"type":"output_text","text":"Hi there","annotations":[{"type":"url_citation","url":"https://example.com","title":"Ex"}]}]},{"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"Paris\"}"}],"usage":{"input_tokens":20,"output_tokens":9,"output_tokens_details":{"reasoning_tokens":4},"input_tokens_details":{"cached_tokens":0}}}}

"#;

#[test]
fn a_stream_decodes_to_the_same_response_as_its_completed_document() {
    let events = decode(STREAM);
    check_canonical(&events).unwrap();
    let mut collect = Collect::default();
    for event in events {
        collect.push(event);
    }
    let streamed = collect.finish().unwrap();
    assert_eq!(streamed.text(), "Hi there");
    assert_eq!(streamed.content.len(), 3);
    assert_eq!(streamed.reasoning()[0].encrypted.as_deref(), Some("enc"));
    assert_eq!(streamed.finish_reason, FinishReason::ToolCalls);
    assert_eq!(streamed.usage.reasoning_tokens, 4);

    let unary = respond(|out| write_reply("openai", &streamed.raw, None, out)).unwrap();
    assert_eq!(unary.content, streamed.content);
    assert_eq!(unary.id.as_deref(), Some("resp_1"));
}

#[test]
fn a_failed_response_is_an_error() {
    let mut decoder = ResponsesDecoder::new("openai", None);
    let frame = SseEvent {
        data: r#"{"type":"response.failed","response":{"error":{"code":"server_error","message":"boom"}}}"#.into(),
        ..SseEvent::default()
    };
    let error = decoder
        .feed(frame, &mut StreamWriter::default())
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Unavailable);
    assert_eq!(error.message(), "boom");
}
