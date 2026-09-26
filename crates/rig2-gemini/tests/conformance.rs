//! The conformance suite against Gemini. Replays cassettes; `RIG2_LIVE=1`
//! records missing ones.

use std::path::Path;
use std::sync::Arc;

use rig2_gemini::GeminiConfig;
use rig2_testkit::conformance::{Models, Suite, assert_conformant, run};

#[tokio::test]
async fn generate_content_conforms() {
    let suite = Suite {
        fixtures: Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/conformance/generate"),
        key_var: Some("GEMINI_API_KEY"),
        build: Box::new(|setup| {
            Arc::new(
                GeminiConfig::new(setup.api_key)
                    .connect(setup.http)
                    .generate(setup.model),
            )
        }),
        models: Models {
            text: "gemini-2.5-flash-lite".into(),
            tools: Some("gemini-2.5-flash-lite".into()),
            parallel_tools: Some("gemini-2.5-flash-lite".into()),
            structured: Some("gemini-2.5-flash-lite".into()),
            vision: Some("gemini-2.5-flash-lite".into()),
            reasoning: Some("gemini-2.5-flash".into()),
        },
        pause: std::time::Duration::ZERO,
    };
    assert_conformant(&run(&suite).await);
}
