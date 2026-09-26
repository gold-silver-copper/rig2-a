//! The agent as a sans-IO state machine.
//!
//! An [`AgentRun`] holds a conversation and where it stands. [`AgentRun::next`]
//! says what must happen next, as a [`Step`]; the caller does it (calls the
//! model, runs the tools, asks a person) and feeds the result back. The run
//! does no I/O and is plain data: serialize it to checkpoint, deserialize it
//! to resume. `next` is pure, so a resumed run re-issues the step that was in
//! flight when it was saved.
//!
//! ```
//! use rig2_agent::run::{AgentRun, RunSettings, Step};
//! use rig2_core::content::Message;
//!
//! let run = AgentRun::new(RunSettings::default(), Vec::new(), Message::user("hi"));
//! assert!(matches!(run.next(), Step::CallModel(_)));
//! let saved = serde_json::to_string(&run).unwrap();
//! let resumed: AgentRun = serde_json::from_str(&saved).unwrap();
//! assert_eq!(resumed, run);
//! ```

use rig2_core::completion::{CompletionRequest, CompletionResponse, ToolDefinition, Usage};
use rig2_core::content::{Message, ToolCall, ToolResult, UserContent};
use rig2_core::{Error, ErrorKind};
use serde::{Deserialize, Serialize};

/// What a run is allowed to do.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunSettings {
    /// The system prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preamble: Option<String>,
    /// The tools the model may call.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    /// Names of tools whose calls need approval.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub needs_approval: Vec<String>,
    /// The most model calls in one run.
    pub max_turns: u32,
    /// The model's context window in tokens; older turns are dropped to fit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u32>,
    /// Stop once the run has cost more than this many US dollars.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_cost: Option<f64>,
    /// Everything else to send with each request: sampling, output limits,
    /// reasoning, provider extensions. Its messages, system prompt and tools
    /// are ignored.
    #[serde(default)]
    pub request: CompletionRequest,
}

impl Default for RunSettings {
    fn default() -> Self {
        Self {
            preamble: None,
            tools: Vec::new(),
            needs_approval: Vec::new(),
            max_turns: 16,
            context_window: None,
            max_cost: None,
            request: CompletionRequest::default(),
        }
    }
}

/// What must happen next.
#[derive(Debug, Clone, PartialEq)]
pub enum Step {
    /// Call the model with this request, then [`AgentRun::model_replied`].
    CallModel(Box<CompletionRequest>),
    /// Ask whether these calls may run, then [`AgentRun::decide`].
    Approve(Vec<ToolCall>),
    /// Run these tools, then [`AgentRun::tools_returned`].
    RunTools(Vec<ToolCall>),
    /// The run is over.
    Done(Outcome),
}

/// How a run ended.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum Outcome {
    /// The model gave its final answer.
    Answered {
        /// The answer.
        response: Box<CompletionResponse>,
    },
    /// The run hit its turn limit.
    MaxTurns,
    /// The run hit its cost limit.
    MaxCost,
    /// A hook or the caller stopped the run.
    Stopped {
        /// Why.
        reason: String,
    },
    /// A model call failed.
    Failed {
        /// The error.
        error: Error,
    },
}

impl Outcome {
    /// The answer, or the reason there is none as an error.
    pub fn into_response(self) -> Result<CompletionResponse, Error> {
        match self {
            Self::Answered { response } => Ok(*response),
            Self::MaxTurns => Err(Error::new(
                ErrorKind::Limit,
                "the run reached its turn limit",
            )),
            Self::MaxCost => Err(Error::new(
                ErrorKind::Limit,
                "the run reached its cost limit",
            )),
            Self::Stopped { reason } => Err(Error::new(ErrorKind::Cancelled, reason)),
            Self::Failed { error } => Err(error),
        }
    }
}

/// A decision on a tool call awaiting approval.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum Decision {
    /// Run it.
    Approve,
    /// Do not run it; the model is told why.
    Deny {
        /// Why.
        reason: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
enum State {
    CallModel,
    Approve {
        calls: Vec<ToolCall>,
    },
    RunTools {
        calls: Vec<ToolCall>,
        denied: Vec<ToolResult>,
    },
    Done {
        outcome: Outcome,
    },
}

/// An agent run: settings, conversation, totals and position.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentRun {
    settings: RunSettings,
    messages: Vec<Message>,
    turns: u32,
    usage: Usage,
    cost: f64,
    state: State,
}

impl AgentRun {
    /// A run that continues `history` with `prompt`.
    pub fn new(settings: RunSettings, history: Vec<Message>, prompt: Message) -> Self {
        let mut messages = history;
        messages.push(prompt);
        Self {
            settings,
            messages,
            turns: 0,
            usage: Usage::default(),
            cost: 0.0,
            state: State::CallModel,
        }
    }

    /// The settings.
    pub fn settings(&self) -> &RunSettings {
        &self.settings
    }

    /// The whole conversation, including history.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Model calls so far.
    pub fn turns(&self) -> u32 {
        self.turns
    }

    /// Token counts so far.
    pub fn usage(&self) -> Usage {
        self.usage
    }

    /// Cost so far in US dollars, for models whose pricing is known.
    pub fn cost(&self) -> f64 {
        self.cost
    }

    /// What must happen next. Pure: calling it twice gives the same step.
    pub fn next(&self) -> Step {
        match &self.state {
            State::CallModel => Step::CallModel(Box::new(self.request())),
            State::Approve { calls } => Step::Approve(calls.clone()),
            State::RunTools { calls, .. } => Step::RunTools(calls.clone()),
            State::Done { outcome } => Step::Done(outcome.clone()),
        }
    }

    /// Whether the run is over.
    pub fn is_done(&self) -> bool {
        matches!(self.state, State::Done { .. })
    }

    fn request(&self) -> CompletionRequest {
        let mut request = self.settings.request.clone();
        request.system.clone_from(&self.settings.preamble);
        request.tools.clone_from(&self.settings.tools);
        let budget = self.settings.context_window.map(|window| {
            let reserved = request.max_tokens.unwrap_or(window / 4);
            let fixed =
                request.system.as_deref().map_or(0, estimate_text) + estimate_json(&request.tools);
            window.saturating_sub(reserved).saturating_sub(fixed)
        });
        request.messages = match budget {
            Some(budget) => trim(&self.messages, budget),
            None => self.messages.clone(),
        };
        request
    }

    /// Feed the model's reply to a [`Step::CallModel`].
    ///
    /// Fails with [`ErrorKind::InvalidRequest`] when the run was not waiting
    /// for the model.
    pub fn model_replied(&mut self, response: CompletionResponse) -> Result<(), Error> {
        self.expect(matches!(self.state, State::CallModel), "a model reply")?;
        self.turns += 1;
        self.usage += response.usage;
        self.cost += response.cost.unwrap_or(0.0);
        self.messages.push(response.message());
        let calls: Vec<ToolCall> = response.tool_calls().into_iter().cloned().collect();
        self.state = if self.settings.max_cost.is_some_and(|max| self.cost > max) {
            State::Done {
                outcome: Outcome::MaxCost,
            }
        } else if calls.is_empty() {
            State::Done {
                outcome: Outcome::Answered {
                    response: Box::new(response),
                },
            }
        } else if self.turns >= self.settings.max_turns {
            State::Done {
                outcome: Outcome::MaxTurns,
            }
        } else if calls
            .iter()
            .any(|c| self.settings.needs_approval.contains(&c.name))
        {
            State::Approve { calls }
        } else {
            State::RunTools {
                calls,
                denied: Vec::new(),
            }
        };
        Ok(())
    }

    /// Feed decisions to a [`Step::Approve`], by call id. Calls without a
    /// decision are denied.
    pub fn decide(&mut self, decisions: &[(String, Decision)]) -> Result<(), Error> {
        let State::Approve { calls } = &self.state else {
            return self.expect(false, "approval decisions");
        };
        let (mut approved, mut denied) = (Vec::new(), Vec::new());
        for call in calls {
            let decision = decisions
                .iter()
                .find(|(id, _)| *id == call.id)
                .map(|(_, d)| d);
            let needs = self.settings.needs_approval.contains(&call.name);
            match decision {
                Some(Decision::Approve) => approved.push(call.clone()),
                None if !needs => approved.push(call.clone()),
                Some(Decision::Deny { reason }) => {
                    denied.push(ToolResult::error(
                        &call.id,
                        &call.name,
                        format!("the call was denied: {reason}"),
                    ));
                }
                None => denied.push(ToolResult::error(
                    &call.id,
                    &call.name,
                    "the call was denied",
                )),
            }
        }
        self.state = State::RunTools {
            calls: approved,
            denied,
        };
        Ok(())
    }

    /// Feed tool results to a [`Step::RunTools`].
    pub fn tools_returned(&mut self, results: Vec<ToolResult>) -> Result<(), Error> {
        let State::RunTools { denied, .. } = &mut self.state else {
            return self.expect(false, "tool results");
        };
        let mut all = std::mem::take(denied);
        all.extend(results);
        self.messages.push(Message::tool_results(all));
        self.state = State::CallModel;
        Ok(())
    }

    /// End the run: a model call failed, or a hook stopped it.
    pub fn end(&mut self, outcome: Outcome) {
        self.state = State::Done { outcome };
    }

    fn expect(&self, ok: bool, what: &str) -> Result<(), Error> {
        if ok {
            Ok(())
        } else {
            Err(Error::new(
                ErrorKind::InvalidRequest,
                format!("the run did not expect {what} now: {:?}", self.next()),
            ))
        }
    }
}

/// A rough token count: four characters a token.
fn estimate_text(text: &str) -> u32 {
    (text.len() / 4 + 1) as u32
}

fn estimate_json(value: &impl Serialize) -> u32 {
    serde_json::to_string(value).map_or(0, |s| estimate_text(&s))
}

/// Drop the oldest turns until the rest fit in `budget` tokens.
///
/// The kept conversation always starts with a user turn that is not a tool
/// result, so no tool result loses its call. The newest such turn is kept
/// even if it alone exceeds the budget.
fn trim(messages: &[Message], budget: u32) -> Vec<Message> {
    let starts: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter(|(_, m)| match m {
            Message::User { content } => !content
                .iter()
                .any(|c| matches!(c, UserContent::ToolResult(_))),
            Message::Assistant { .. } => false,
        })
        .map(|(i, _)| i)
        .collect();
    let cost_from = |start: usize| messages[start..].iter().map(estimate_json).sum::<u32>();
    let start = starts
        .iter()
        .copied()
        .find(|&s| cost_from(s) <= budget)
        .or_else(|| starts.last().copied())
        .unwrap_or(0);
    messages[start..].to_vec()
}

#[cfg(test)]
mod tests;
