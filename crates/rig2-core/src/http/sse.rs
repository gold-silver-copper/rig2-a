use std::collections::VecDeque;

use futures::StreamExt;

use super::{ByteStream, LineSplitter};
use crate::{BoxStream, Result};

/// One server-sent event.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SseEvent {
    /// The `event:` field, when present.
    pub event: Option<String>,
    /// The `data:` lines, joined with newlines.
    pub data: String,
    /// The `id:` field, when present.
    pub id: Option<String>,
}

/// An incremental server-sent-events parser, following the WHATWG
/// specification: `data` lines accumulate, a blank line dispatches, and
/// comment lines are ignored.
#[derive(Debug, Default)]
pub struct SseParser {
    lines: LineSplitter,
    current: SseEvent,
    has_data: bool,
}

impl SseParser {
    /// Events completed by `chunk`.
    pub fn push(&mut self, chunk: &[u8]) -> Vec<SseEvent> {
        let mut events = Vec::new();
        for line in self.lines.push(chunk) {
            if let Some(event) = self.line(&line) {
                events.push(event);
            }
        }
        events
    }

    /// The event left pending when the input ends without a blank line.
    pub fn finish(&mut self) -> Option<SseEvent> {
        if let Some(line) = self.lines.finish()
            && let Some(event) = self.line(&line)
        {
            return Some(event);
        }
        self.dispatch()
    }

    fn line(&mut self, line: &[u8]) -> Option<SseEvent> {
        if line.is_empty() {
            return self.dispatch();
        }
        let line = String::from_utf8_lossy(line);
        if line.starts_with(':') {
            return None;
        }
        let (field, value) = match line.split_once(':') {
            Some((field, value)) => (field, value.strip_prefix(' ').unwrap_or(value)),
            None => (line.as_ref(), ""),
        };
        match field {
            "data" => {
                if self.has_data {
                    self.current.data.push('\n');
                }
                self.current.data.push_str(value);
                self.has_data = true;
            }
            "event" => self.current.event = Some(value.to_owned()),
            "id" => self.current.id = Some(value.to_owned()),
            _ => {}
        }
        None
    }

    fn dispatch(&mut self) -> Option<SseEvent> {
        let event = std::mem::take(&mut self.current);
        let had_data = std::mem::take(&mut self.has_data);
        (had_data || event.event.is_some()).then_some(event)
    }
}

/// Parse a response body as server-sent events.
pub fn sse(body: ByteStream) -> BoxStream<'static, Result<SseEvent>> {
    Box::pin(futures::stream::unfold(
        (body, SseParser::default(), VecDeque::new(), false),
        |(mut body, mut parser, mut ready, mut done)| async move {
            loop {
                if let Some(event) = ready.pop_front() {
                    return Some((Ok(event), (body, parser, ready, done)));
                }
                if done {
                    return None;
                }
                match body.next().await {
                    Some(Ok(chunk)) => ready.extend(parser.push(&chunk)),
                    Some(Err(error)) => {
                        done = true;
                        return Some((Err(error), (body, parser, ready, done)));
                    }
                    None => {
                        done = true;
                        ready.extend(parser.finish());
                    }
                }
            }
        },
    ))
}
