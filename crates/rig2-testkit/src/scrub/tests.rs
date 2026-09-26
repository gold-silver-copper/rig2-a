use std::path::Path;

use super::*;

/// Fake secrets are assembled at runtime so that no key-shaped literal is
/// ever committed, even in tests.
fn fake(prefix: &str, len: usize) -> String {
    format!(
        "{prefix}{}",
        "Ab1".chars().cycle().take(len).collect::<String>()
    )
}

#[test]
fn scrubbing_removes_keys_ids_emails_and_home_paths() {
    let key = fake("sk-", 40);
    let google = fake("AIza", 35);
    let text = format!(
        "key={key} g={google} mail=ada{at}lovelace.dev arn=arn:aws:bedrock:us-east-1:123456789012:model path=/Users/ada/src ok=user{at}example.com",
        at = '@'
    );
    let scrubbed = scrub_text(&text);
    assert!(!scrubbed.contains(&key) && !scrubbed.contains(&google));
    assert!(!scrubbed.contains("lovelace.dev"));
    assert!(!scrubbed.contains("123456789012"));
    assert!(scrubbed.contains("/Users/[REDACTED]/src"));
    assert!(scrubbed.contains("user@example.com"), "{scrubbed}");
    assert!(
        scan_text(Path::new("x"), &scrubbed).is_empty(),
        "{:?}",
        scan_text(Path::new("x"), &scrubbed)
    );
}

#[test]
fn scanning_reports_what_it_found_without_the_secret() {
    let findings = scan_text(
        Path::new("f.json"),
        &format!("line one\n\"auth\": \"{}\"", fake("gsk_", 30)),
    );
    assert_eq!(findings.len(), 1);
    assert_eq!(findings[0].line, 2);
    assert_eq!(findings[0].what, "Groq key");
}

#[test]
fn sensitive_headers_are_recognised_case_insensitively() {
    for header in [
        "Authorization",
        "x-api-key",
        "OpenAI-Organization",
        "x-goog-api-key",
        "Set-Cookie",
    ] {
        assert!(is_sensitive_header(header), "{header}");
    }
    assert!(!is_sensitive_header("content-type"));
}

/// Every fixture checked into this workspace is free of secrets.
#[test]
fn the_workspace_fixtures_are_clean() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let findings = scan_fixtures(&root).unwrap();
    assert!(findings.is_empty(), "{findings:#?}");
}
