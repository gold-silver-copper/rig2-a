use std::collections::{BTreeMap, VecDeque};

use futures::StreamExt;
use serde::{Deserialize, Serialize};

use super::{CompletionResponse, FinishReason, Usage};
use crate::content::{AssistantContent, Citation, Extensions, Reasoning, Text, ToolCall};
use crate::{BoxStream, Error, ErrorKind, MaybeSend, Result};

/// One event of a streamed completion.
///
/// A stream is canonical: every block starts once, its deltas come between
/// its start and its end, every block ends exactly once with the finished
/// content, and exactly one [`Finish`] comes last. [`StreamWriter`] cannot
/// produce anything else, and [`check_canonical`] verifies it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum StreamEvent {
    /// A block opened.
    BlockStart {
        /// The block's position in the reply.
        index: u32,
        /// What the block holds.
        kind: BlockKind,
    },
    /// More of an open block.
    Delta {
        /// The block's position in the reply.
        index: u32,
        /// The new fragment.
        delta: Delta,
    },
    /// A block closed, carrying its finished content.
    BlockEnd {
        /// The block's position in the reply.
        index: u32,
        /// The finished content.
        content: AssistantContent,
    },
    /// The reply is complete.
    Finish(Finish),
}

/// What an open block holds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BlockKind {
    /// Text.
    Text,
    /// Reasoning.
    Reasoning,
    /// A tool call.
    ToolCall {
        /// The call id.
        id: String,
        /// The tool name.
        name: String,
    },
    /// A generated image.
    Image,
}

/// A fragment of an open block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum Delta {
    /// Text.
    Text(String),
    /// Reasoning text.
    Reasoning(String),
    /// A fragment of a tool call's JSON arguments.
    Arguments(String),
}

/// The end of a reply: usage, reason, ids and the provider's document.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Finish {
    /// Token counts.
    pub usage: Usage,
    /// Why generation stopped.
    pub finish_reason: FinishReason,
    /// The provider's response id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// The model that answered.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// The cost in US dollars, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<f64>,
    /// The provider's reply document, the same shape as a unary reply.
    #[serde(default)]
    pub raw: serde_json::Value,
}

/// A handle to an open text block. Only [`StreamWriter::text`] makes one.
#[derive(Debug)]
pub struct TextBlock(u32);
/// A handle to an open reasoning block. Only [`StreamWriter::reasoning`] makes one.
#[derive(Debug)]
pub struct ReasoningBlock(u32);
/// A handle to an open tool-call block. Only [`StreamWriter::tool_call`] makes one.
#[derive(Debug)]
pub struct ToolCallBlock(u32);

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::TextBlock {}
    impl Sealed for super::ReasoningBlock {}
    impl Sealed for super::ToolCallBlock {}
}

/// A handle to an open block that takes fragments.
pub trait OpenBlock: sealed::Sealed {
    #[doc(hidden)]
    fn index(&self) -> u32;
}

impl OpenBlock for TextBlock {
    fn index(&self) -> u32 {
        self.0
    }
}
impl OpenBlock for ReasoningBlock {
    fn index(&self) -> u32 {
        self.0
    }
}
impl OpenBlock for ToolCallBlock {
    fn index(&self) -> u32 {
        self.0
    }
}

#[derive(Debug)]
enum Open {
    Text(Text),
    Reasoning(Reasoning),
    ToolCall {
        id: String,
        name: String,
        arguments: String,
        extensions: Extensions,
    },
}

/// Writes a canonical completion stream.
///
/// Opening a block returns a handle; the handle is the only way to add to or
/// end the block, and ending consumes it, so a block cannot be ended twice or
/// written after its end. A handle dropped without ending leaves its block to
/// be ended by [`finish`](Self::finish), which consumes the writer, so there
/// is exactly one terminal event.
///
/// ```
/// use rig2_core::completion::{Finish, StreamWriter};
///
/// let mut out = StreamWriter::default();
/// let text = out.text();
/// out.push(&text, "Hel");
/// out.push(&text, "lo");
/// out.end_text(text);
/// let events = out.finish(Finish::default()).unwrap();
/// assert_eq!(events.len(), 5);
/// ```
#[derive(Debug, Default)]
pub struct StreamWriter {
    events: VecDeque<StreamEvent>,
    next_index: u32,
    open: BTreeMap<u32, Open>,
}

impl StreamWriter {
    fn open(&mut self, kind: BlockKind, open: Open) -> u32 {
        let index = self.next_index;
        self.next_index += 1;
        self.events
            .push_back(StreamEvent::BlockStart { index, kind });
        self.open.insert(index, open);
        index
    }

    /// Open a text block.
    pub fn text(&mut self) -> TextBlock {
        TextBlock(self.open(BlockKind::Text, Open::Text(Text::default())))
    }

    /// Open a reasoning block.
    pub fn reasoning(&mut self) -> ReasoningBlock {
        ReasoningBlock(self.open(BlockKind::Reasoning, Open::Reasoning(Reasoning::default())))
    }

    /// Open a tool-call block.
    pub fn tool_call(&mut self, id: impl Into<String>, name: impl Into<String>) -> ToolCallBlock {
        let (id, name) = (id.into(), name.into());
        let kind = BlockKind::ToolCall {
            id: id.clone(),
            name: name.clone(),
        };
        ToolCallBlock(self.open(
            kind,
            Open::ToolCall {
                id,
                name,
                arguments: String::new(),
                extensions: Extensions::default(),
            },
        ))
    }

    /// Append a fragment to an open block. Empty fragments are dropped.
    pub fn push(&mut self, block: &impl OpenBlock, fragment: &str) {
        if fragment.is_empty() {
            return;
        }
        let index = block.index();
        let delta = match self.open.get_mut(&index) {
            Some(Open::Text(text)) => {
                text.text.push_str(fragment);
                Delta::Text(fragment.to_owned())
            }
            Some(Open::Reasoning(reasoning)) => {
                reasoning.text.push_str(fragment);
                Delta::Reasoning(fragment.to_owned())
            }
            Some(Open::ToolCall { arguments, .. }) => {
                arguments.push_str(fragment);
                Delta::Arguments(fragment.to_owned())
            }
            None => return,
        };
        self.events.push_back(StreamEvent::Delta { index, delta });
    }

    /// Add a citation to an open text block.
    pub fn cite(&mut self, block: &TextBlock, citation: Citation) {
        if let Some(Open::Text(text)) = self.open.get_mut(&block.0) {
            text.citations.push(citation);
        }
    }

    /// Set the signature of an open reasoning block.
    pub fn sign(&mut self, block: &ReasoningBlock, signature: impl Into<String>) {
        if let Some(Open::Reasoning(reasoning)) = self.open.get_mut(&block.0) {
            reasoning.signature = Some(signature.into());
        }
    }

    /// Set the encrypted form of an open reasoning block.
    pub fn encrypt(&mut self, block: &ReasoningBlock, data: impl Into<String>) {
        if let Some(Open::Reasoning(reasoning)) = self.open.get_mut(&block.0) {
            reasoning.encrypted = Some(data.into());
        }
    }

    /// The extensions of an open block, to attach provider data to its content.
    pub fn extensions(&mut self, block: &impl OpenBlock) -> Option<&mut Extensions> {
        match self.open.get_mut(&block.index())? {
            Open::Text(text) => Some(&mut text.extensions),
            Open::Reasoning(reasoning) => Some(&mut reasoning.extensions),
            Open::ToolCall { extensions, .. } => Some(extensions),
        }
    }

    /// End a text block.
    #[allow(clippy::needless_pass_by_value)] // consuming the handle is what ends the block
    pub fn end_text(&mut self, block: TextBlock) {
        self.close(block.0).ok();
    }

    /// End a reasoning block.
    #[allow(clippy::needless_pass_by_value)] // consuming the handle is what ends the block
    pub fn end_reasoning(&mut self, block: ReasoningBlock) {
        self.close(block.0).ok();
    }

    /// End a tool-call block, parsing its arguments.
    ///
    /// Empty arguments parse as `{}`. Fails with
    /// [`ErrorKind::MalformedToolInput`] when the arguments are not JSON.
    #[allow(clippy::needless_pass_by_value)] // consuming the handle is what ends the block
    pub fn end_tool_call(&mut self, block: ToolCallBlock) -> Result<()> {
        self.close(block.0)
    }

    /// Write a whole part: a start, one fragment for text-like parts, and an
    /// end. Unary decoders use this to write a finished reply.
    pub fn part(&mut self, content: AssistantContent) {
        let (kind, delta) = match &content {
            AssistantContent::Text(t) => (BlockKind::Text, Some(Delta::Text(t.text.clone()))),
            AssistantContent::Reasoning(r) => {
                (BlockKind::Reasoning, Some(Delta::Reasoning(r.text.clone())))
            }
            AssistantContent::ToolCall(c) => (
                BlockKind::ToolCall {
                    id: c.id.clone(),
                    name: c.name.clone(),
                },
                Some(Delta::Arguments(c.arguments.to_string())),
            ),
            AssistantContent::Image(_) => (BlockKind::Image, None),
        };
        let index = self.next_index;
        self.next_index += 1;
        self.events
            .push_back(StreamEvent::BlockStart { index, kind });
        let delta = delta.filter(|d| !matches!(d, Delta::Text(s) | Delta::Reasoning(s) | Delta::Arguments(s) if s.is_empty()));
        if let Some(delta) = delta {
            self.events.push_back(StreamEvent::Delta { index, delta });
        }
        self.events
            .push_back(StreamEvent::BlockEnd { index, content });
    }

    fn close(&mut self, index: u32) -> Result<()> {
        let Some(open) = self.open.remove(&index) else {
            return Ok(());
        };
        let content = match open {
            Open::Text(text) => AssistantContent::Text(text),
            Open::Reasoning(reasoning) => AssistantContent::Reasoning(reasoning),
            Open::ToolCall {
                id,
                name,
                arguments,
                extensions,
            } => {
                let parsed = if arguments.trim().is_empty() {
                    serde_json::Value::Object(serde_json::Map::new())
                } else {
                    serde_json::from_str(&arguments).map_err(|e| {
                        Error::new(
                            ErrorKind::MalformedToolInput,
                            format!("tool `{name}` arguments are not JSON: {e}"),
                        )
                    })?
                };
                AssistantContent::ToolCall(ToolCall {
                    id,
                    name,
                    arguments: parsed,
                    extensions,
                })
            }
        };
        self.events
            .push_back(StreamEvent::BlockEnd { index, content });
        Ok(())
    }

    /// Take the events written so far.
    pub fn drain(&mut self) -> impl Iterator<Item = StreamEvent> + '_ {
        self.events.drain(..)
    }

    /// End every open block, in order, then write the terminal event.
    ///
    /// Returns the events not yet drained. Fails when an open tool call's
    /// arguments are not JSON.
    pub fn finish(mut self, finish: Finish) -> Result<Vec<StreamEvent>> {
        let open: Vec<u32> = self.open.keys().copied().collect();
        for index in open {
            self.close(index)?;
        }
        self.events.push_back(StreamEvent::Finish(finish));
        Ok(self.events.into())
    }
}

/// Turns a provider's stream frames into canonical events.
///
/// A decoder is sans-IO: it sees one frame at a time and writes through the
/// [`StreamWriter`]. It returns the [`Finish`] instead of writing it, so the
/// driver ([`decode_stream`]) emits exactly one terminal event.
pub trait StreamDecoder<F>: MaybeSend + 'static {
    /// Handle one frame. Return `Some` when the frame ends the reply.
    fn feed(&mut self, frame: F, out: &mut StreamWriter) -> Result<Option<Finish>>;

    /// The frames ran out before a frame ended the reply.
    ///
    /// Return the finish if the reply is complete anyway, or an error if the
    /// stream was cut short.
    fn end(&mut self, out: &mut StreamWriter) -> Result<Finish>;
}

struct Driver<F, D> {
    frames: BoxStream<'static, Result<F>>,
    decoder: D,
    writer: Option<StreamWriter>,
    pending: VecDeque<Result<StreamEvent>>,
}

/// Run `decoder` over `frames`, yielding canonical events.
///
/// The stream ends after the finish or after the first error.
pub fn decode_stream<F, D>(
    frames: BoxStream<'static, Result<F>>,
    decoder: D,
) -> BoxStream<'static, Result<StreamEvent>>
where
    F: MaybeSend + 'static,
    D: StreamDecoder<F>,
{
    let driver = Driver {
        frames,
        decoder,
        writer: Some(StreamWriter::default()),
        pending: VecDeque::new(),
    };
    Box::pin(futures::stream::unfold(driver, |mut driver| async move {
        loop {
            if let Some(event) = driver.pending.pop_front() {
                return Some((event, driver));
            }
            let mut writer = driver.writer.take()?;
            let step = match driver.frames.next().await {
                Some(Ok(frame)) => driver.decoder.feed(frame, &mut writer),
                Some(Err(error)) => Err(error),
                None => driver.decoder.end(&mut writer).map(Some),
            };
            driver.pending.extend(writer.drain().map(Ok));
            match step {
                Ok(None) => driver.writer = Some(writer),
                Ok(Some(finish)) => match writer.finish(finish) {
                    Ok(events) => driver.pending.extend(events.into_iter().map(Ok)),
                    Err(error) => driver.pending.push_back(Err(error)),
                },
                Err(error) => driver.pending.push_back(Err(error)),
            }
        }
    }))
}

/// Build a finished response by writing it: the unary counterpart of
/// [`decode_stream`], so both paths produce content the same way.
pub fn respond(
    write: impl FnOnce(&mut StreamWriter) -> Result<Finish>,
) -> Result<CompletionResponse> {
    let mut writer = StreamWriter::default();
    let finish = write(&mut writer)?;
    let mut collect = Collect::default();
    for event in writer.finish(finish)? {
        collect.push(event);
    }
    collect.finish()
}

/// Folds a canonical stream into a [`CompletionResponse`].
#[derive(Debug, Default)]
pub struct Collect {
    parts: BTreeMap<u32, AssistantContent>,
    finish: Option<Finish>,
}

impl Collect {
    /// Take one event.
    pub fn push(&mut self, event: StreamEvent) {
        match event {
            StreamEvent::BlockEnd { index, content } => {
                self.parts.insert(index, content);
            }
            StreamEvent::Finish(finish) => self.finish = Some(finish),
            StreamEvent::BlockStart { .. } | StreamEvent::Delta { .. } => {}
        }
    }

    /// The response, or an error if the stream never finished.
    pub fn finish(self) -> Result<CompletionResponse> {
        let finish = self.finish.ok_or_else(|| {
            Error::new(
                ErrorKind::Decode,
                "the stream ended before the reply finished",
            )
        })?;
        Ok(CompletionResponse {
            content: self.parts.into_values().collect(),
            usage: finish.usage,
            finish_reason: finish.finish_reason,
            id: finish.id,
            model: finish.model,
            cost: finish.cost,
            raw: finish.raw,
        })
    }
}

/// Drain a stream into a [`CompletionResponse`].
pub async fn collect(
    mut stream: BoxStream<'static, Result<StreamEvent>>,
) -> Result<CompletionResponse> {
    let mut collect = Collect::default();
    while let Some(event) = stream.next().await {
        collect.push(event?);
    }
    collect.finish()
}

/// Check that `events` form a canonical stream.
///
/// Useful for testing a decoder. Fails with [`ErrorKind::Decode`] naming the
/// first violation.
pub fn check_canonical(events: &[StreamEvent]) -> Result<()> {
    let fail =
        |at: usize, what: &str| Err(Error::new(ErrorKind::Decode, format!("event {at}: {what}")));
    let mut open: BTreeMap<u32, BlockKind> = BTreeMap::new();
    let mut ended: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();
    let mut finished = false;
    for (at, event) in events.iter().enumerate() {
        if finished {
            return fail(at, "event after finish");
        }
        match event {
            StreamEvent::BlockStart { index, kind } => {
                if open.contains_key(index) || ended.contains(index) {
                    return fail(at, "block started twice");
                }
                open.insert(*index, kind.clone());
            }
            StreamEvent::Delta { index, delta } => {
                let Some(kind) = open.get(index) else {
                    return fail(at, "delta outside an open block");
                };
                let fits = matches!(
                    (kind, delta),
                    (BlockKind::Text, Delta::Text(_))
                        | (BlockKind::Reasoning, Delta::Reasoning(_))
                        | (BlockKind::ToolCall { .. }, Delta::Arguments(_))
                );
                if !fits {
                    return fail(at, "delta kind does not match its block");
                }
            }
            StreamEvent::BlockEnd { index, content } => {
                let Some(kind) = open.remove(index) else {
                    return fail(at, "end of a block that is not open");
                };
                let fits = matches!(
                    (&kind, content),
                    (BlockKind::Text, AssistantContent::Text(_))
                        | (BlockKind::Reasoning, AssistantContent::Reasoning(_))
                        | (BlockKind::ToolCall { .. }, AssistantContent::ToolCall(_))
                        | (BlockKind::Image, AssistantContent::Image(_))
                );
                if !fits {
                    return fail(at, "content does not match its block");
                }
                ended.insert(*index);
            }
            StreamEvent::Finish(_) => {
                if !open.is_empty() {
                    return fail(at, "finish with open blocks");
                }
                finished = true;
            }
        }
    }
    if finished {
        Ok(())
    } else {
        fail(events.len(), "no finish")
    }
}
