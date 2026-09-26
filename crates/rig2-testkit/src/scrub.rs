//! Keep secrets out of fixtures: scrub them as they are written, and scan
//! checked-in files for anything that slipped through.
//!
//! What counts as a secret: the values of exported credential variables
//! (names ending in `_API_KEY`, `_KEY`, `_TOKEN` or `_SECRET`, and the AWS
//! credentials), known key shapes (OpenAI, Anthropic, Google, Groq, xAI,
//! Perplexity, AWS access keys, bearer tokens), organization and project
//! ids, AWS account ids in ARNs, email addresses, and home-directory paths.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use regex::Regex;

/// Header names that never reach a fixture.
const SENSITIVE_HEADERS: &[&str] = &[
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "api-key",
    "x-goog-api-key",
    "cookie",
    "set-cookie",
    "openai-organization",
    "openai-project",
    "anthropic-organization-id",
    "x-amz-security-token",
    "x-amz-content-sha256",
    "x-amz-date",
    "x-amz-user-agent",
];

/// Whether a header must be dropped from fixtures.
pub fn is_sensitive_header(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    SENSITIVE_HEADERS.contains(&name.as_str()) || name.contains("token") || name.contains("secret")
}

fn patterns() -> &'static [(&'static str, Regex)] {
    static PATTERNS: OnceLock<Vec<(&'static str, Regex)>> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        [
            ("OpenAI-style key", r"sk-[A-Za-z0-9_\-]{20,}"),
            ("Google key", r"AIza[0-9A-Za-z_\-]{30,}"),
            ("Groq key", r"gsk_[A-Za-z0-9]{20,}"),
            ("xAI key", r"xai-[A-Za-z0-9]{20,}"),
            ("Perplexity key", r"pplx-[A-Za-z0-9]{20,}"),
            ("AWS access key", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
            ("Google OAuth token", r"ya29\.[0-9A-Za-z_\-]+"),
            ("bearer token", r"Bearer [A-Za-z0-9._\-]{16,}"),
            ("OpenAI organization", r"\borg-[A-Za-z0-9]{20,}"),
            ("OpenAI project", r"\bproj_[A-Za-z0-9]{20,}"),
            (
                "AWS account in ARN",
                r"arn:aws[a-z\-]*:[a-z0-9\-]*:[a-z0-9\-]*:\d{12}:",
            ),
            (
                "email address",
                r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}\b",
            ),
            ("home directory", r"(?:/Users|/home)/[A-Za-z0-9._\-]+"),
        ]
        .into_iter()
        .filter_map(|(name, pattern)| Regex::new(pattern).ok().map(|re| (name, re)))
        .collect()
    })
}

/// Addresses that are safe to keep: documentation and test domains.
fn is_placeholder_email(email: &str) -> bool {
    [
        "@example.com",
        "@example.org",
        "@example.net",
        "@test.invalid",
    ]
    .iter()
    .any(|d| email.ends_with(d))
}

/// Values of exported credential variables, longest first.
pub fn secret_values() -> Vec<String> {
    let mut values: Vec<String> = std::env::vars()
        .filter(|(name, _)| {
            name.ends_with("_API_KEY")
                || name.ends_with("_KEY")
                || name.ends_with("_TOKEN")
                || name.ends_with("_SECRET")
                || name == "AWS_ACCESS_KEY_ID"
                || name == "AWS_SECRET_ACCESS_KEY"
        })
        .map(|(_, value)| value)
        .filter(|value| value.len() >= 8)
        .collect();
    values.sort_by_key(|v| std::cmp::Reverse(v.len()));
    values
}

/// Replace every secret in `text` with `[REDACTED]`.
pub fn scrub_text(text: &str) -> String {
    let mut out = text.to_owned();
    for value in secret_values() {
        out = out.replace(&value, "[REDACTED]");
    }
    for (name, pattern) in patterns() {
        out = pattern
            .replace_all(&out, |caps: &regex::Captures<'_>| {
                let found = &caps[0];
                if *name == "email address" && is_placeholder_email(found) {
                    found.to_owned()
                } else if *name == "home directory" {
                    let root = if found.starts_with("/Users") {
                        "/Users"
                    } else {
                        "/home"
                    };
                    format!("{root}/[REDACTED]")
                } else {
                    "[REDACTED]".to_owned()
                }
            })
            .into_owned();
    }
    out
}

/// A secret found in a file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    /// The file.
    pub path: PathBuf,
    /// The 1-based line.
    pub line: usize,
    /// What was found (never the secret itself).
    pub what: String,
}

/// Every secret in `text`, by line.
pub fn scan_text(path: &Path, text: &str) -> Vec<Finding> {
    let secrets = secret_values();
    let mut findings = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if secrets.iter().any(|s| line.contains(s.as_str())) {
            findings.push(Finding {
                path: path.to_owned(),
                line: index + 1,
                what: "an exported credential value".into(),
            });
        }
        for (name, pattern) in patterns() {
            let found = pattern
                .find_iter(line)
                .any(|m| !(*name == "email address" && is_placeholder_email(m.as_str())));
            if found {
                findings.push(Finding {
                    path: path.to_owned(),
                    line: index + 1,
                    what: (*name).to_owned(),
                });
            }
        }
    }
    findings
}

/// Scan every fixture file under `root`: files in any `fixtures` directory,
/// and any `*.cassette.json` or `*.recording.json` file. `target`, `.git`
/// and hidden directories are skipped.
pub fn scan_fixtures(root: &Path) -> std::io::Result<Vec<Finding>> {
    let mut findings = Vec::new();
    let mut stack = vec![(root.to_owned(), false)];
    while let Some((dir, in_fixtures)) = stack.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().into_owned();
            if entry.file_type()?.is_dir() {
                if name == "target" || name.starts_with('.') {
                    continue;
                }
                stack.push((path, in_fixtures || name == "fixtures"));
            } else if in_fixtures
                || name.ends_with(".cassette.json")
                || name.ends_with(".recording.json")
            {
                let bytes = std::fs::read(&path)?;
                findings.extend(scan_text(&path, &String::from_utf8_lossy(&bytes)));
            }
        }
    }
    Ok(findings)
}

#[cfg(test)]
mod tests;
