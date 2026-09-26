use rig2_core::completion::{Collect, OutputSchema, ToolDefinition, check_canonical};
use rig2_core::content::ToolResult;
use rig2_core::http::SseParser;

use super::*;

#[test]
fn the_request_maps_roles_tools_schema_thinking_and_signatures() {
    let mut call = ToolCall::new("call_1", "get_weather", json!({"city": "Paris"}));
    call.extensions
        .insert(&ThoughtSignature("sig".into()))
        .unwrap();
    let request = CompletionRequest::new(vec![
        Message::user("weather?"),
        Message::Assistant {
            content: vec![AssistantContent::ToolCall(call)],
        },
        Message::tool_results([ToolResult::text("call_1", "get_weather", "sunny")]),
    ])
    .with_system("brief")
    .with_tool(ToolDefinition {
        name: "get_weather".into(),
        description: "W.".into(),
        parameters: json!({"type": "object"}),
    })
    .with_tool_choice(ToolChoice::Tool("get_weather".into()))
    .with_output(OutputSchema {
        name: "o".into(),
        schema: json!({"type": "object"}),
    })
    .with_reasoning(ReasoningEffort::Low);
    let body = request_body(&request, false).unwrap();
    assert_eq!(body["systemInstruction"]["parts"][0]["text"], "brief");
    assert_eq!(body["contents"][1]["role"], "model");
    assert_eq!(body["contents"][1]["parts"][0]["thoughtSignature"], "sig");
    assert_eq!(
        body["contents"][2]["parts"][0]["functionResponse"],
        json!({"name": "get_weather", "response": {"output": "sunny"}})
    );
    assert_eq!(
        body["toolConfig"]["functionCallingConfig"]["allowedFunctionNames"][0],
        "get_weather"
    );
    assert_eq!(
        body["generationConfig"]["responseMimeType"],
        "application/json"
    );
    assert_eq!(
        body["generationConfig"]["thinkingConfig"]["thinkingBudget"],
        2048
    );
}

const STREAM: &str = r#"data: {"candidates":[{"content":{"parts":[{"text":"Plan","thought":true}],"role":"model"}}],"modelVersion":"gemini-x","responseId":"r1"}

data: {"candidates":[{"content":{"parts":[{"text":"Hel"}],"role":"model"}}]}

data: {"candidates":[{"content":{"parts":[{"text":"lo"}],"role":"model"}}]}

data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"city":"Paris"}},"thoughtSignature":"SIG"}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":9,"candidatesTokenCount":4,"thoughtsTokenCount":6},"modelVersion":"gemini-x","responseId":"r1"}

"#;

#[test]
fn a_stream_decodes_to_the_same_response_as_its_merged_document() {
    let mut decoder = GenerateDecoder::new(None);
    let mut parser = SseParser::default();
    let mut out = StreamWriter::default();
    let mut events = Vec::new();
    for frame in parser.push(STREAM.as_bytes()) {
        assert!(decoder.feed(frame, &mut out).unwrap().is_none());
        events.extend(out.drain());
    }
    let finish = decoder.end(&mut out).unwrap();
    events.extend(out.drain());
    events.extend(out.finish(finish).unwrap());
    check_canonical(&events).unwrap();
    let mut collect = Collect::default();
    for event in events {
        collect.push(event);
    }
    let streamed = collect.finish().unwrap();
    assert_eq!(streamed.text(), "Hello");
    assert_eq!(streamed.reasoning()[0].text, "Plan");
    let call = streamed.tool_calls()[0].clone();
    assert_eq!(call.id, "call_1");
    assert_eq!(
        call.extensions.get::<ThoughtSignature>().unwrap(),
        Some(ThoughtSignature("SIG".into()))
    );
    assert_eq!(streamed.finish_reason, FinishReason::ToolCalls);
    assert_eq!(streamed.usage.output_tokens, 10);
    assert_eq!(streamed.usage.reasoning_tokens, 6);

    let unary = respond(|out| write_reply(&streamed.raw, None, out)).unwrap();
    assert_eq!(unary.content, streamed.content);
    assert_eq!(unary.usage, streamed.usage);
}
