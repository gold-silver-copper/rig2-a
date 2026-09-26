use proptest::prelude::*;

use super::*;
use crate::content::AssistantContent;

#[derive(Debug, Clone)]
enum Op {
    OpenText,
    OpenReasoning,
    OpenTool(u8),
    Push(usize, String),
    End(usize),
    Drop(usize),
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        Just(Op::OpenText),
        Just(Op::OpenReasoning),
        any::<u8>().prop_map(Op::OpenTool),
        (0..6_usize, "[a-z ]{0,6}").prop_map(|(slot, s)| Op::Push(slot, s)),
        (0..6_usize).prop_map(Op::End),
        (0..6_usize).prop_map(Op::Drop),
    ]
}

enum Handle {
    Text(TextBlock),
    Reasoning(ReasoningBlock),
    Tool(ToolCallBlock, String),
}

/// What each block should end up holding, by open order.
#[derive(Debug, Clone, PartialEq)]
enum Expect {
    Text(String),
    Reasoning(String),
    Tool(serde_json::Value),
}

fn run(ops: &[Op]) -> (Vec<StreamEvent>, Vec<Expect>) {
    let mut out = StreamWriter::default();
    let mut events = Vec::new();
    let mut slots: Vec<(usize, Handle)> = Vec::new();
    let mut expect: Vec<Expect> = Vec::new();
    for op in ops {
        match op {
            Op::OpenText => {
                slots.push((expect.len(), Handle::Text(out.text())));
                expect.push(Expect::Text(String::new()));
            }
            Op::OpenReasoning => {
                slots.push((expect.len(), Handle::Reasoning(out.reasoning())));
                expect.push(Expect::Reasoning(String::new()));
            }
            Op::OpenTool(n) => {
                let args = serde_json::json!({ "n": n });
                let pending = args.to_string();
                slots.push((
                    expect.len(),
                    Handle::Tool(out.tool_call(format!("call_{n}"), "t"), pending),
                ));
                expect.push(Expect::Tool(args));
            }
            Op::Push(slot, s) if !slots.is_empty() => {
                let slot = slot % slots.len();
                let (at, handle) = &mut slots[slot];
                match handle {
                    Handle::Text(b) => {
                        out.push(b, s);
                        if let Expect::Text(t) = &mut expect[*at] {
                            t.push_str(s);
                        }
                    }
                    Handle::Reasoning(b) => {
                        out.push(b, s);
                        if let Expect::Reasoning(t) = &mut expect[*at] {
                            t.push_str(s);
                        }
                    }
                    Handle::Tool(b, pending) => {
                        let take = pending.len().min(3);
                        let chunk: String = pending.drain(..take).collect();
                        out.push(b, &chunk);
                    }
                }
            }
            Op::End(slot) | Op::Drop(slot) if !slots.is_empty() => {
                let (_, handle) = slots.remove(slot % slots.len());
                match (op, handle) {
                    (Op::End(_), Handle::Text(b)) => out.end_text(b),
                    (Op::End(_), Handle::Reasoning(b)) => out.end_reasoning(b),
                    (Op::End(_), Handle::Tool(b, pending)) => {
                        out.push(&b, &pending);
                        out.end_tool_call(b).unwrap();
                    }
                    (_, Handle::Tool(b, pending)) => {
                        // A dropped tool call still has its arguments completed first.
                        out.push(&b, &pending);
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        events.extend(out.drain());
    }
    // Blocks still open are ended by `finish`; complete their arguments first.
    for (_, handle) in &slots {
        if let Handle::Tool(b, pending) = handle {
            out.push(b, pending);
        }
    }
    events.extend(out.drain());
    events.extend(out.finish(Finish::default()).unwrap());
    (events, expect)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// Any sequence of writer calls yields a canonical stream, and collecting
    /// it gives back exactly the blocks written, in open order.
    #[test]
    fn any_writer_sequence_is_canonical_and_collects_to_what_was_written(ops in proptest::collection::vec(op(), 0..40)) {
        let (events, expect) = run(&ops);
        prop_assert!(check_canonical(&events).is_ok(), "{:?}", check_canonical(&events));
        let mut collect = Collect::default();
        for event in events {
            collect.push(event);
        }
        let response = collect.finish().unwrap();
        let got: Vec<Expect> = response
            .content
            .iter()
            .map(|c| match c {
                AssistantContent::Text(t) => Expect::Text(t.text.clone()),
                AssistantContent::Reasoning(r) => Expect::Reasoning(r.text.clone()),
                AssistantContent::ToolCall(c) => Expect::Tool(c.arguments.clone()),
                AssistantContent::Image(_) => panic!("no images are written"),
            })
            .collect();
        prop_assert_eq!(got, expect);
    }
}

#[test]
fn malformed_tool_arguments_are_a_typed_error() {
    let mut out = StreamWriter::default();
    let call = out.tool_call("c1", "lookup");
    out.push(&call, "{\"q\": ");
    let error = out.end_tool_call(call).unwrap_err();
    assert_eq!(error.kind(), crate::ErrorKind::MalformedToolInput);
}

#[test]
fn check_canonical_rejects_events_after_finish_and_unended_blocks() {
    let finish = StreamEvent::Finish(Finish::default());
    assert!(check_canonical(&[finish.clone(), finish.clone()]).is_err());
    let start = StreamEvent::BlockStart {
        index: 0,
        kind: BlockKind::Text,
    };
    assert!(check_canonical(&[start, finish]).is_err());
    assert!(check_canonical(&[]).is_err());
}

struct Lines;

impl StreamDecoder<String> for Lines {
    fn feed(&mut self, frame: String, out: &mut StreamWriter) -> crate::Result<Option<Finish>> {
        if frame == "DONE" {
            return Ok(Some(Finish::default()));
        }
        let text = out.text();
        out.push(&text, &frame);
        // The handle is dropped: finish ends the block.
        Ok(None)
    }

    fn end(&mut self, _out: &mut StreamWriter) -> crate::Result<Finish> {
        Err(crate::Error::new(crate::ErrorKind::Decode, "truncated"))
    }
}

#[test]
fn the_driver_emits_one_finish_and_stops_at_the_first_terminal_frame() {
    let frames = futures::stream::iter(["a", "b", "DONE", "ignored"].map(|s| Ok(s.to_owned())));
    let events: Vec<_> = futures::executor::block_on_stream(decode_stream(Box::pin(frames), Lines))
        .map(Result::unwrap)
        .collect();
    check_canonical(&events).unwrap();
    let response = futures::executor::block_on(collect(Box::pin(futures::stream::iter(
        events.into_iter().map(Ok),
    ))))
    .unwrap();
    assert_eq!(response.text(), "ab");
}

#[test]
fn a_truncated_stream_ends_with_the_decoders_error() {
    let frames = futures::stream::iter(["a"].map(|s| Ok(s.to_owned())));
    let items: Vec<_> =
        futures::executor::block_on_stream(decode_stream(Box::pin(frames), Lines)).collect();
    assert!(items.last().unwrap().is_err());
}

#[test]
fn parse_tolerates_a_code_fence() {
    let response = respond(|out| {
        out.part(AssistantContent::Text(crate::content::Text::new(
            "```json\n{\"a\": 1}\n```",
        )));
        Ok(Finish::default())
    })
    .unwrap();
    let value: serde_json::Value = response.parse().unwrap();
    assert_eq!(value["a"], 1);
}
