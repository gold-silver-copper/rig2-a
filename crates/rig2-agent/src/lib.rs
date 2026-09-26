//! Agents for rig2.
//!
//! An agent is a sans-IO state machine, [`run::AgentRun`], and a small async
//! driver, [`Agent`], that executes its steps with a model and a set of
//! tools. Streaming, hooks, approval, turn and cost limits, context trimming,
//! memory and retrieval are features of those two, not a separate runtime.
//! Checkpointing is serializing the run.
//!
//! ```no_run
//! # async fn demo(model: std::sync::Arc<dyn rig2_core::StreamingModel<rig2_core::completion::Completion>>) -> rig2_core::Result<()> {
//! use rig2_agent::{Agent, tool};
//!
//! /// Multiply two numbers.
//! #[tool]
//! fn multiply(a: f64, b: f64) -> f64 {
//!     a * b
//! }
//!
//! let agent = Agent::builder(model).preamble("Use tools for arithmetic.").tool(Multiply).build();
//! println!("{}", agent.prompt("What is 6 times 7?").await?);
//! # Ok(()) }
//! ```

#[cfg_attr(
    not(test),
    expect(
        unused_extern_crates,
        reason = "lets macro-generated paths name this crate from inside it and its tests"
    )
)]
extern crate self as rig2_agent;

mod agent;
pub mod memory;
pub mod retrieval;
pub mod run;
pub mod tool;

pub use agent::{Agent, AgentBuilder, AgentEvent, Approver, Control, Hook};
pub use rig2_macros::tool;

#[doc(hidden)]
pub mod __private {
    pub use rig2_core::completion::ToolDefinition;
    pub use rig2_core::content::ToolOutput;
    pub use rig2_core::{BoxFuture, Result};
    pub use {schemars, serde, serde_json};

    pub use crate::tool::{parse_arguments, schema_of, to_output};

    /// A tool function's error, reported to the model.
    pub fn tool_failure(error: impl std::fmt::Display) -> rig2_core::Error {
        rig2_core::Error::new(rig2_core::ErrorKind::Other, error.to_string())
    }
}
