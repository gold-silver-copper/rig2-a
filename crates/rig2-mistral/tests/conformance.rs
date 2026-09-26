//! The conformance suite against Mistral. Replays cassettes; `RIG2_LIVE=1`
//! records missing ones.

use std::path::Path;
use std::sync::Arc;

use rig2_testkit::conformance::{Models, Suite, assert_conformant, run};

#[tokio::test]
async fn chat_conforms() {
    let suite = Suite {
        fixtures: Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/conformance/chat"),
        key_var: Some("MISTRAL_API_KEY"),
        build: Box::new(|setup| {
            Arc::new(
                rig2_mistral::config(setup.api_key)
                    .connect(setup.http)
                    .chat(setup.model),
            )
        }),
        models: Models {
            text: "ministral-3b-latest".into(),
            tools: Some("ministral-8b-latest".into()),
            parallel_tools: Some("ministral-14b-latest".into()),
            structured: Some("ministral-8b-latest".into()),
            vision: Some("ministral-8b-latest".into()),
            // This account has no quota for any Mistral reasoning model
            // (their rate limit is zero), so reasoning is not covered.
            reasoning: None,
        },
        pause: std::time::Duration::from_secs(3),
    };
    assert_conformant(&run(&suite).await);
}
