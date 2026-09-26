//! The agent: a model, tools and settings, and the async driver that runs an
//! [`AgentRun`] to its end.

use std::sync::Arc;

use futures::StreamExt;
use rig2_core::catalog::ModelCard;
use rig2_core::completion::{
    Collect, Completion, CompletionRequest, CompletionResponse, StreamEvent,
};
use rig2_core::content::{Message, ToolCall, ToolResult};
use rig2_core::{
    BoxFuture, BoxStream, Error, ErrorKind, MaybeSend, MaybeSync, Result, StreamingModel,
};

use crate::memory::Memory;
use crate::run::{AgentRun, Decision, Outcome, RunSettings, Step};
use crate::tool::{Tool, ToolContext, ToolSet};

/// Whether a hook lets the run go on.
#[derive(Debug, Clone, PartialEq)]
pub enum Control<T> {
    /// Go on, with this value.
    Continue(T),
    /// Stop the run, with a reason.
    Stop(String),
}

/// Observes a run, and may rewrite requests or stop it.
///
/// Every method has a default that does nothing, so a hook implements only
/// what it needs.
pub trait Hook: MaybeSend + MaybeSync + 'static {
    /// Before each model call: return the request to send, or stop.
    fn before_model(
        &self,
        request: CompletionRequest,
    ) -> BoxFuture<'static, Result<Control<CompletionRequest>>> {
        Box::pin(std::future::ready(Ok(Control::Continue(request))))
    }

    /// Before each tool call: allow it, or veto it with a reason the model
    /// sees.
    fn before_tool(&self, _call: &ToolCall) -> BoxFuture<'static, Result<Control<()>>> {
        Box::pin(std::future::ready(Ok(Control::Continue(()))))
    }

    /// Every event of the run, as it happens.
    fn on_event(&self, _event: &AgentEvent) {}
}

/// Decides on tool calls that need approval.
pub trait Approver: MaybeSend + MaybeSync + 'static {
    /// One decision per call, by call id.
    fn decide(&self, calls: Vec<ToolCall>) -> BoxFuture<'static, Result<Vec<(String, Decision)>>>;
}

/// Something that happened in a run.
#[derive(Debug, Clone, PartialEq)]
pub enum AgentEvent {
    /// A streamed model event.
    Model(StreamEvent),
    /// A model call finished.
    Reply(Box<CompletionResponse>),
    /// A tool is about to run.
    ToolCall(ToolCall),
    /// A tool finished.
    ToolResult(ToolResult),
    /// The run is waiting for approval, and no approver is set.
    NeedsApproval(Vec<ToolCall>),
    /// The run ended.
    Done(Outcome),
}

/// An agent: a model, a preamble, tools and limits.
///
/// ```no_run
/// # async fn demo(model: std::sync::Arc<dyn rig2_core::StreamingModel<rig2_core::completion::Completion>>) -> rig2_core::Result<()> {
/// use rig2_agent::Agent;
///
/// let agent = Agent::builder(model).preamble("You are terse.").max_turns(4).build();
/// let answer = agent.prompt("What is 2 + 2?").await?;
/// println!("{answer}");
/// # Ok(()) }
/// ```
#[derive(Clone)]
pub struct Agent {
    model: Arc<dyn StreamingModel<Completion>>,
    tools: ToolSet,
    settings: RunSettings,
    hooks: Vec<Arc<dyn Hook>>,
    approver: Option<Arc<dyn Approver>>,
    memory: Option<(Arc<dyn Memory>, String)>,
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("model", self.model.info())
            .field("tools", &self.tools)
            .finish_non_exhaustive()
    }
}

/// Builds an [`Agent`].
pub struct AgentBuilder {
    agent: Agent,
    context_window: Option<u32>,
}

impl std::fmt::Debug for AgentBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.agent.fmt(f)
    }
}

impl Agent {
    /// Start building an agent over `model`.
    ///
    /// The context window is taken from the model's catalog card when it
    /// has one.
    pub fn builder(model: impl StreamingModel<Completion> + 'static) -> AgentBuilder {
        let model: Arc<dyn StreamingModel<Completion>> = Arc::new(model);
        let card: ModelCard = model.capabilities();
        AgentBuilder {
            agent: Agent {
                model,
                tools: ToolSet::new(),
                settings: RunSettings::default(),
                hooks: Vec::new(),
                approver: None,
                memory: None,
            },
            context_window: card.context_window,
        }
    }

    /// The model.
    pub fn model(&self) -> &Arc<dyn StreamingModel<Completion>> {
        &self.model
    }

    /// The tools.
    pub fn tools(&self) -> &ToolSet {
        &self.tools
    }

    /// A new run of this agent from `history` and `prompt`.
    pub fn start(&self, history: Vec<Message>, prompt: impl Into<Message>) -> AgentRun {
        AgentRun::new(self.settings.clone(), history, prompt.into())
    }

    /// Answer `prompt`, running tools as needed, and return the final text.
    ///
    /// With memory, the conversation is loaded before and saved after.
    /// Fails with the model's error, with [`ErrorKind::Limit`] at a turn or
    /// cost limit, with [`ErrorKind::Cancelled`] when a hook stops the run,
    /// or with [`ErrorKind::Config`] when a call needs approval and no
    /// approver is set (drive the run with [`Agent::drive`] instead).
    pub async fn prompt(&self, prompt: impl Into<Message>) -> Result<String> {
        let prompt = prompt.into();
        let history = match &self.memory {
            Some((memory, conversation)) => memory.load(conversation).await?,
            None => Vec::new(),
        };
        let known = history.len();
        let run = self.drive(self.start(history, prompt), |_| {}).await?;
        if let Some((memory, conversation)) = &self.memory {
            memory
                .append(conversation, run.messages()[known..].to_vec())
                .await?;
        }
        match run.next() {
            Step::Done(outcome) => outcome.into_response().map(|r| r.text()),
            Step::Approve(calls) => Err(Error::new(
                ErrorKind::Config,
                format!(
                    "`{}` needs approval and the agent has no approver",
                    calls.first().map_or("", |c| c.name.as_str())
                ),
            )),
            Step::CallModel(_) | Step::RunTools(_) => {
                Err(Error::new(ErrorKind::Other, "the run stopped early"))
            }
        }
    }

    /// Stream the events of answering `prompt`.
    pub fn stream(&self, prompt: impl Into<Message>) -> BoxStream<'static, Result<AgentEvent>> {
        let (sender, receiver) = futures::channel::mpsc::unbounded();
        let agent = self.clone();
        let run = self.start(Vec::new(), prompt);
        let drive = async move {
            let sink = sender.clone();
            let result = agent
                .drive(run, move |event| {
                    let _ = sink.unbounded_send(Ok(event.clone()));
                })
                .await;
            if let Err(error) = result {
                let _ = sender.unbounded_send(Err(error));
            }
        };
        // The driver and the event channel make one stream: polling the
        // stream polls the driver.
        let driven = futures::stream::once(drive).filter_map(|()| std::future::ready(None));
        Box::pin(futures::stream::select(driven, receiver))
    }

    /// Advance `run` until it is done or waits for an approval that no
    /// approver can give. Every event is passed to `on_event` and the hooks.
    ///
    /// A failed model call ends the run with [`Outcome::Failed`]; the
    /// returned error is reserved for misuse of the run.
    pub async fn drive(
        &self,
        mut run: AgentRun,
        mut on_event: impl FnMut(&AgentEvent) + MaybeSend,
    ) -> Result<AgentRun> {
        let mut emit = |event: AgentEvent| {
            for hook in &self.hooks {
                hook.on_event(&event);
            }
            on_event(&event);
        };
        loop {
            match run.next() {
                Step::Done(outcome) => {
                    emit(AgentEvent::Done(outcome));
                    return Ok(run);
                }
                Step::CallModel(request) => match self.call_model(*request, &mut emit).await {
                    Ok(Control::Continue(response)) => {
                        emit(AgentEvent::Reply(Box::new(response.clone())));
                        run.model_replied(response)?;
                    }
                    Ok(Control::Stop(reason)) => run.end(Outcome::Stopped { reason }),
                    Err(error) => run.end(Outcome::Failed { error }),
                },
                Step::Approve(calls) => match &self.approver {
                    Some(approver) => {
                        let decisions = approver.decide(calls).await?;
                        run.decide(&decisions)?;
                    }
                    None => {
                        emit(AgentEvent::NeedsApproval(calls));
                        return Ok(run);
                    }
                },
                Step::RunTools(calls) => {
                    let context = ToolContext::new(run.messages().into());
                    for call in &calls {
                        emit(AgentEvent::ToolCall(call.clone()));
                    }
                    let results = futures::future::join_all(
                        calls
                            .into_iter()
                            .map(|call| self.call_tool(call, context.clone())),
                    )
                    .await;
                    for result in &results {
                        emit(AgentEvent::ToolResult(result.clone()));
                    }
                    run.tools_returned(results)?;
                }
            }
        }
    }

    async fn call_model(
        &self,
        mut request: CompletionRequest,
        emit: &mut impl FnMut(AgentEvent),
    ) -> Result<Control<CompletionResponse>> {
        for hook in &self.hooks {
            match hook.before_model(request).await? {
                Control::Continue(next) => request = next,
                Control::Stop(reason) => return Ok(Control::Stop(reason)),
            }
        }
        let mut stream = self.model.invoke_stream(request).await?;
        let mut collect = Collect::default();
        while let Some(event) = stream.next().await {
            let event = event?;
            emit(AgentEvent::Model(event.clone()));
            collect.push(event);
        }
        collect.finish().map(Control::Continue)
    }

    async fn call_tool(&self, call: ToolCall, context: ToolContext) -> ToolResult {
        for hook in &self.hooks {
            match hook.before_tool(&call).await {
                Ok(Control::Continue(())) => {}
                Ok(Control::Stop(reason)) => {
                    return ToolResult::error(
                        &call.id,
                        &call.name,
                        format!("the call was vetoed: {reason}"),
                    );
                }
                Err(error) => return ToolResult::error(&call.id, &call.name, error.to_string()),
            }
        }
        let Some(tool) = self.tools.get(&call.name) else {
            return ToolResult::error(
                &call.id,
                &call.name,
                format!("there is no tool named `{}`", call.name),
            );
        };
        match tool.call(call.arguments.clone(), context).await {
            Ok(output) => ToolResult {
                call_id: call.id,
                name: call.name,
                output,
                is_error: false,
            },
            Err(error) => ToolResult::error(&call.id, &call.name, error.message()),
        }
    }
}

impl AgentBuilder {
    /// Set the system prompt.
    pub fn preamble(mut self, preamble: impl Into<String>) -> Self {
        self.agent.settings.preamble = Some(preamble.into());
        self
    }

    /// Add a tool. Pass an `Arc<dyn Tool>` to share one between agents.
    pub fn tool(mut self, tool: impl Tool) -> Self {
        self.agent.tools.insert(tool);
        self
    }

    /// Limit model calls per run (default 16).
    pub fn max_turns(mut self, max_turns: u32) -> Self {
        self.agent.settings.max_turns = max_turns.max(1);
        self
    }

    /// Stop a run once it costs more than `usd`.
    pub fn max_cost(mut self, usd: f64) -> Self {
        self.agent.settings.max_cost = Some(usd);
        self
    }

    /// Override the context window used to trim old turns.
    pub fn context_window(mut self, tokens: u32) -> Self {
        self.context_window = Some(tokens);
        self
    }

    /// Set what every request carries besides messages, preamble and tools.
    pub fn request(mut self, request: CompletionRequest) -> Self {
        self.agent.settings.request = request;
        self
    }

    /// Add a hook.
    pub fn hook(mut self, hook: impl Hook) -> Self {
        self.agent.hooks.push(Arc::new(hook));
        self
    }

    /// Decide approvals with `approver` instead of pausing the run.
    pub fn approver(mut self, approver: impl Approver) -> Self {
        self.agent.approver = Some(Arc::new(approver));
        self
    }

    /// Load and save the conversation `conversation` in `memory`.
    pub fn memory(mut self, memory: impl Memory, conversation: impl Into<String>) -> Self {
        self.agent.memory = Some((Arc::new(memory), conversation.into()));
        self
    }

    /// The agent.
    pub fn build(mut self) -> Agent {
        self.agent.settings.tools = self.agent.tools.definitions();
        self.agent.settings.needs_approval = self.agent.tools.needing_approval();
        self.agent.settings.context_window = self.context_window;
        self.agent
    }
}

#[cfg(test)]
mod tests;
