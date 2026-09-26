//! The conformance suite against Anthropic. Replays cassettes;
//! `RIG2_LIVE=1` records missing ones.

use std::path::Path;
use std::sync::Arc;

use rig2_anthropic::AnthropicConfig;
use rig2_testkit::conformance::{Models, Suite, assert_conformant, run};

#[tokio::test]
async fn messages_api_conforms() {
    let suite = Suite {
        fixtures: Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/conformance/messages"),
        key_var: Some("ANTHROPIC_API_KEY"),
        build: Box::new(|setup| {
            Arc::new(
                AnthropicConfig::new(setup.api_key)
                    .connect(setup.http)
                    .messages(setup.model),
            )
        }),
        models: Models {
            text: "claude-haiku-4-5-20251001".into(),
            tools: Some("claude-haiku-4-5-20251001".into()),
            parallel_tools: Some("claude-haiku-4-5-20251001".into()),
            structured: Some("claude-haiku-4-5-20251001".into()),
            vision: Some("claude-haiku-4-5-20251001".into()),
            reasoning: Some("claude-haiku-4-5-20251001".into()),
        },
        pause: std::time::Duration::ZERO,
    };
    assert_conformant(&run(&suite).await);
}
