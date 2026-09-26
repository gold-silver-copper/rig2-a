//! The conformance suite against Cohere. Replays cassettes; `RIG2_LIVE=1`
//! records missing ones.

use std::path::Path;
use std::sync::Arc;

use rig2_cohere::CohereConfig;
use rig2_testkit::conformance::{Models, Suite, assert_conformant, run};

#[tokio::test]
async fn chat_conforms() {
    let suite = Suite {
        fixtures: Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/conformance/chat"),
        key_var: Some("COHERE_API_KEY"),
        build: Box::new(|setup| {
            Arc::new(
                CohereConfig::new(setup.api_key)
                    .connect(setup.http)
                    .chat(setup.model),
            )
        }),
        models: Models {
            text: "command-r7b-12-2024".into(),
            tools: Some("command-r7b-12-2024".into()),
            parallel_tools: Some("command-a-03-2025".into()),
            structured: Some("command-r7b-12-2024".into()),
            vision: Some("command-a-vision-07-2025".into()),
            reasoning: Some("command-a-reasoning-08-2025".into()),
        },
        pause: std::time::Duration::from_secs(2),
    };
    assert_conformant(&run(&suite).await);
}
