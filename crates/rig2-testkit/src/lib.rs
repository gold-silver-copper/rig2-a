//! Test support for rig2.
//!
//! - [`ScriptedModel`]: a completion model that replies from a script and
//!   records what it was asked, for testing agents and anything built on
//!   models.
//! - [`cassette`]: an HTTP client that records exchanges and replays them,
//!   scrubbed of secrets, for testing provider mappings byte for byte.
//! - [`scrub`]: the secret scrubber and the fixture scanner.
//! - [`conformance`]: the provider conformance suite.

mod scripted;

pub use scripted::{Reply, ScriptedModel};
