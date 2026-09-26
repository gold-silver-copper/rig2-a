use rig2_core::completion::{Finish, FinishReason, respond};
use rig2_core::content::{AssistantContent, Text};

use super::*;

fn reply(parts: Vec<AssistantContent>, cost: f64) -> CompletionResponse {
    respond(|out| {
        for part in parts {
            out.part(part);
        }
        Ok(Finish {
            cost: Some(cost),
            finish_reason: FinishReason::Stop,
            ..Finish::default()
        })
    })
    .unwrap()
}

fn call(id: &str, name: &str) -> AssistantContent {
    AssistantContent::ToolCall(ToolCall::new(id, name, serde_json::json!({})))
}

fn text(t: &str) -> AssistantContent {
    AssistantContent::Text(Text::new(t))
}

#[test]
fn a_tool_loop_calls_the_model_runs_tools_and_answers() {
    let mut run = AgentRun::new(RunSettings::default(), Vec::new(), Message::user("hi"));
    run.model_replied(reply(vec![call("1", "t")], 0.0)).unwrap();
    let Step::RunTools(calls) = run.next() else {
        panic!("{:?}", run.next())
    };
    assert_eq!(calls.len(), 1);
    run.tools_returned(vec![ToolResult::text("1", "t", "ok")])
        .unwrap();
    assert!(matches!(run.next(), Step::CallModel(ref r) if r.messages.len() == 3));
    run.model_replied(reply(vec![text("done")], 0.0)).unwrap();
    let Step::Done(outcome) = run.next() else {
        panic!()
    };
    assert_eq!(outcome.into_response().unwrap().text(), "done");
    assert_eq!(run.turns(), 2);
}

#[test]
fn feeding_the_wrong_result_is_an_error_and_changes_nothing() {
    let mut run = AgentRun::new(RunSettings::default(), Vec::new(), Message::user("hi"));
    let before = run.clone();
    assert_eq!(
        run.tools_returned(Vec::new()).unwrap_err().kind(),
        ErrorKind::InvalidRequest
    );
    assert_eq!(run, before);
}

#[test]
fn turn_and_cost_limits_end_the_run() {
    let settings = RunSettings {
        max_turns: 1,
        ..RunSettings::default()
    };
    let mut run = AgentRun::new(settings, Vec::new(), Message::user("hi"));
    run.model_replied(reply(vec![call("1", "t")], 0.0)).unwrap();
    assert_eq!(run.next(), Step::Done(Outcome::MaxTurns));

    let settings = RunSettings {
        max_cost: Some(0.01),
        ..RunSettings::default()
    };
    let mut run = AgentRun::new(settings, Vec::new(), Message::user("hi"));
    run.model_replied(reply(vec![call("1", "t")], 0.02))
        .unwrap();
    assert_eq!(run.next(), Step::Done(Outcome::MaxCost));
}

#[test]
fn approval_gates_only_the_tools_that_need_it_and_denials_reach_the_model() {
    let settings = RunSettings {
        needs_approval: vec!["delete".into()],
        ..RunSettings::default()
    };
    let mut run = AgentRun::new(settings, Vec::new(), Message::user("clean up"));
    run.model_replied(reply(vec![call("1", "delete"), call("2", "list")], 0.0))
        .unwrap();
    assert!(matches!(run.next(), Step::Approve(ref calls) if calls.len() == 2));
    run.decide(&[(
        "1".into(),
        Decision::Deny {
            reason: "not today".into(),
        },
    )])
    .unwrap();
    let Step::RunTools(calls) = run.next() else {
        panic!()
    };
    assert_eq!(
        calls.iter().map(|c| c.id.as_str()).collect::<Vec<_>>(),
        ["2"]
    );
    run.tools_returned(vec![ToolResult::text("2", "list", "a, b")])
        .unwrap();
    let Message::User { content } = run.messages().last().unwrap() else {
        panic!()
    };
    assert_eq!(
        content.len(),
        2,
        "the denial and the result are both sent back"
    );
}

#[test]
fn a_saved_run_resumes_with_the_step_that_was_in_flight() {
    let mut run = AgentRun::new(RunSettings::default(), Vec::new(), Message::user("hi"));
    run.model_replied(reply(vec![call("1", "t")], 0.0)).unwrap();
    let saved = serde_json::to_string(&run).unwrap();
    let resumed: AgentRun = serde_json::from_str(&saved).unwrap();
    assert_eq!(resumed.next(), run.next());
}

#[test]
fn trimming_drops_whole_old_exchanges_and_keeps_tool_pairs() {
    let long = "x".repeat(4000);
    let messages = vec![
        Message::user(long.clone()),
        Message::Assistant {
            content: vec![call("1", "t")],
        },
        Message::tool_results([ToolResult::text("1", "t", long)]),
        Message::assistant("ok"),
        Message::user("latest"),
    ];
    let kept = trim(&messages, 50);
    assert_eq!(kept, vec![Message::user("latest")]);
    let everything = trim(&messages, 1_000_000);
    assert_eq!(everything.len(), messages.len());
}
