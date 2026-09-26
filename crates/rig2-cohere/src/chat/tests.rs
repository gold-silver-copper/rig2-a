use rig2_core::completion::{Collect, check_canonical};
use rig2_core::http::SseParser;

use super::*;

const STREAM: &str = r#"event: message-start
data: {"id":"c1","type":"message-start","delta":{"message":{"role":"assistant","content":[],"tool_plan":"","tool_calls":[],"citations":[]}}}

event: tool-plan-delta
data: {"type":"tool-plan-delta","delta":{"message":{"tool_plan":"I will check."}}}

event: tool-call-start
data: {"type":"tool-call-start","index":0,"delta":{"message":{"tool_calls":{"id":"get_weather_1","type":"function","function":{"name":"get_weather","arguments":""}}}}}

event: tool-call-delta
data: {"type":"tool-call-delta","index":0,"delta":{"message":{"tool_calls":{"function":{"arguments":"{\"city\":\"Paris\"}"}}}}}

event: tool-call-end
data: {"type":"tool-call-end","index":0}

event: message-end
data: {"type":"message-end","delta":{"finish_reason":"TOOL_CALL","usage":{"billed_units":{"input_tokens":20,"output_tokens":8},"tokens":{"input_tokens":900,"output_tokens":30}}}}

"#;

#[test]
fn a_stream_decodes_to_the_same_response_as_its_document() {
    let mut decoder = ChatDecoder::new(None);
    let mut parser = SseParser::default();
    let mut out = StreamWriter::default();
    let mut events = Vec::new();
    let mut finish = None;
    for frame in parser.push(STREAM.as_bytes()) {
        let done = decoder.feed(frame, &mut out).unwrap();
        events.extend(out.drain());
        if done.is_some() {
            finish = done;
        }
    }
    events.extend(out.finish(finish.unwrap()).unwrap());
    check_canonical(&events).unwrap();
    let mut collect = Collect::default();
    for event in events {
        collect.push(event);
    }
    let streamed = collect.finish().unwrap();
    assert_eq!(streamed.reasoning()[0].text, "I will check.");
    assert_eq!(streamed.tool_calls()[0].arguments, json!({"city": "Paris"}));
    assert_eq!(streamed.finish_reason, FinishReason::ToolCalls);
    assert_eq!(streamed.usage.input_tokens, 900);

    let unary = respond(|out| write_reply(&streamed.raw, None, out)).unwrap();
    assert_eq!(unary.content, streamed.content);
}

#[test]
fn a_tool_plan_goes_back_with_its_calls() {
    let message = assistant_message(&[
        AssistantContent::Reasoning(Reasoning {
            text: "plan".into(),
            ..Reasoning::default()
        }),
        AssistantContent::ToolCall(rig2_core::content::ToolCall::new("t1", "f", json!({}))),
    ]);
    assert_eq!(message["tool_plan"], "plan");
    assert_eq!(message["tool_calls"][0]["function"]["arguments"], "{}");
}
