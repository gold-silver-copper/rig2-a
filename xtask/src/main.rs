//! Repository tasks.
//!
//! - `cargo xtask scan`: fail if any fixture holds a secret.

use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("scan") => scan(),
        _ => {
            eprintln!("usage: cargo xtask scan");
            ExitCode::FAILURE
        }
    }
}

fn scan() -> ExitCode {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    match rig2_testkit::scrub::scan_fixtures(&root) {
        Ok(findings) if findings.is_empty() => {
            eprintln!("fixture scan: clean");
            ExitCode::SUCCESS
        }
        Ok(findings) => {
            for finding in findings {
                eprintln!(
                    "{}:{}: {}",
                    finding.path.display(),
                    finding.line,
                    finding.what
                );
            }
            ExitCode::FAILURE
        }
        Err(error) => {
            eprintln!("fixture scan failed: {error}");
            ExitCode::FAILURE
        }
    }
}
