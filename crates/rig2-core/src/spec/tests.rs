use std::sync::Arc;

use super::*;

#[test]
fn a_model_id_may_contain_colons() {
    let spec: ModelSpec = "ollama:qwen3:4b".parse().unwrap();
    assert_eq!(spec.provider, "ollama");
    assert_eq!(spec.model, "qwen3:4b");
}

#[test]
fn a_spec_without_a_provider_is_a_config_error() {
    for bad in ["gpt-5", ":m", "p:"] {
        assert_eq!(
            bad.parse::<ModelSpec>().unwrap_err().kind(),
            ErrorKind::Config,
            "{bad}"
        );
    }
}

#[test]
fn factories_are_keyed_by_provider_and_model_type() {
    let mut registry = Registry::default();
    registry.register::<str>("p", |spec| Ok(Arc::from(spec.model.as_str())));
    registry.register::<[u8]>("p", |spec| Ok(Arc::from(spec.model.as_bytes())));
    let spec = ModelSpec::new("p", "m");
    assert_eq!(&*registry.build::<str>(&spec).unwrap(), "m");
    assert_eq!(&*registry.build::<[u8]>(&spec).unwrap(), b"m");
    let missing = registry
        .build::<str>(&ModelSpec::new("q", "m"))
        .unwrap_err();
    assert_eq!(missing.kind(), ErrorKind::Config);
}

#[test]
fn a_spec_round_trips_through_json_without_empty_fields() {
    let spec = ModelSpec::new("openai", "gpt-5-mini").with_option("api", "chat");
    let json = serde_json::to_string(&spec).unwrap();
    assert_eq!(
        json,
        r#"{"provider":"openai","model":"gpt-5-mini","options":{"api":"chat"}}"#
    );
    assert_eq!(serde_json::from_str::<ModelSpec>(&json).unwrap(), spec);
}
