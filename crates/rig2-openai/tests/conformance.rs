//! The conformance suite against OpenAI, on both APIs. Replays cassettes;
//! `RIG2_LIVE=1` records missing ones.

use std::path::Path;
use std::sync::Arc;

use rig2_openai::OpenAIConfig;
use rig2_testkit::conformance::{Models, Suite, assert_conformant, run};

fn suite(api: &'static str) -> Suite {
    Suite {
        fixtures: Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures/conformance")
            .join(api),
        key_var: Some("OPENAI_API_KEY"),
        build: Box::new(move |setup| {
            let client = OpenAIConfig::new(setup.api_key).connect(setup.http);
            match api {
                "chat" => Arc::new(client.chat(setup.model)),
                _ => Arc::new(client.responses(setup.model)),
            }
        }),
        models: Models {
            text: "gpt-4.1-nano".into(),
            tools: Some("gpt-4.1-nano".into()),
            parallel_tools: Some("gpt-4.1-mini".into()),
            structured: Some("gpt-4.1-nano".into()),
            vision: Some("gpt-4.1-mini".into()),
            reasoning: Some("gpt-5-nano".into()),
        },
    }
}

#[tokio::test]
async fn responses_api_conforms() {
    assert_conformant(&run(&suite("responses")).await);
}

#[tokio::test]
async fn chat_completions_api_conforms() {
    assert_conformant(&run(&suite("chat")).await);
}
