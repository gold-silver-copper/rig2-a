//! The Messages API: request mapping, the unary reply and the streaming
//! decoder.

use std::collections::BTreeMap;
use std::sync::Arc;

use bytes::Bytes;
use rig2_core::catalog::{ModelCard, Pricing};
use rig2_core::completion::{
    CacheHint, Completion, CompletionRequest, CompletionResponse, Finish, FinishReason,
    StreamDecoder, StreamEvent, StreamWriter, TextBlock, ToolCallBlock, ToolChoice, Usage,
    decode_stream, respond,
};
use rig2_core::content::{
    AssistantContent, Citation, Extension, Image, Message, Reasoning, Source, Text, ToolOutput,
    UserContent,
};
use rig2_core::http::{SseEvent, base64, read_json, send, set, sse};
use rig2_core::{BoxFuture, BoxStream, Error, ErrorKind, Model, ModelInfo, Result, StreamingModel};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::Anthropic;

/// Beta features to switch on with the `anthropic-beta` header.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Beta(pub Vec<String>);

impl Extension for Beta {
    const KEY: &'static str = "anthropic.beta";
}

const STRUCTURED_OUTPUTS: &str = "structured-outputs-2025-11-13";

fn unsupported(what: &str) -> Error {
    Error::new(
        ErrorKind::Unsupported,
        format!("{what} is not supported by Anthropic"),
    )
}

fn source(source: &Source, media_type: &str) -> Value {
    match source {
        Source::Bytes(bytes) => {
            json!({ "type": "base64", "media_type": media_type, "data": base64(bytes) })
        }
        Source::Url(url) => json!({ "type": "url", "url": url }),
    }
}

fn image(image: &Image) -> Result<Value> {
    let media_type = match (&image.source, &image.media_type) {
        (Source::Bytes(_), None) => return Err(unsupported("an image without a media type")),
        (_, media_type) => media_type.clone().unwrap_or_default(),
    };
    Ok(json!({ "type": "image", "source": source(&image.source, &media_type) }))
}

fn user_blocks(content: &[UserContent]) -> Result<Vec<Value>> {
    content
        .iter()
        .map(|part| match part {
            UserContent::Text(text) => Ok(json!({ "type": "text", "text": text.text })),
            UserContent::Image(i) => image(i),
            UserContent::File(file) => {
                let mut block = json!({ "type": "document", "source": source(&file.source, &file.media_type), "citations": { "enabled": true } });
                if let Some(name) = &file.name {
                    set(&mut block, "title", name.clone());
                }
                Ok(block)
            }
            UserContent::Audio(_) => Err(unsupported("audio input")),
            UserContent::ToolResult(result) => {
                let content = result
                    .output
                    .iter()
                    .map(|o| match o {
                        ToolOutput::Text(text) => Ok(json!({ "type": "text", "text": text })),
                        ToolOutput::Json(value) => Ok(json!({ "type": "text", "text": value.to_string() })),
                        ToolOutput::Image(i) => image(i),
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(json!({ "type": "tool_result", "tool_use_id": result.call_id, "content": content, "is_error": result.is_error }))
            }
        })
        .collect()
}

fn assistant_blocks(content: &[AssistantContent]) -> Vec<Value> {
    content
        .iter()
        .filter_map(|part| match part {
            AssistantContent::Text(text) if !text.text.is_empty() => Some(json!({ "type": "text", "text": text.text })),
            AssistantContent::ToolCall(call) => {
                Some(json!({ "type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments }))
            }
            AssistantContent::Reasoning(r) => match (&r.encrypted, &r.signature) {
                (Some(data), _) => Some(json!({ "type": "redacted_thinking", "data": data })),
                (None, Some(signature)) => Some(json!({ "type": "thinking", "thinking": r.text, "signature": signature })),
                (None, None) => None,
            },
            AssistantContent::Text(_) | AssistantContent::Image(_) => None,
        })
        .collect()
}

fn mark_cache(block: Option<&mut Value>) {
    if let Some(block) = block {
        set(block, "cache_control", json!({ "type": "ephemeral" }));
    }
}

/// The Messages request body and beta headers for `request`.
pub(crate) fn request_body(
    model: &str,
    card: &ModelCard,
    request: &CompletionRequest,
    stream: bool,
) -> Result<(Value, Vec<String>)> {
    let mut messages: Vec<Value> = Vec::new();
    for message in &request.messages {
        let (role, blocks) = match message {
            Message::User { content } => ("user", user_blocks(content)?),
            Message::Assistant { content } => ("assistant", assistant_blocks(content)),
        };
        if !blocks.is_empty() {
            messages.push(json!({ "role": role, "content": blocks }));
        }
    }
    let mut system: Vec<Value> = request
        .system
        .iter()
        .map(|s| json!({ "type": "text", "text": s }))
        .collect();
    let mut tools: Vec<Value> = request
        .tools
        .iter()
        .map(|t| json!({ "name": t.name, "description": t.description, "input_schema": t.parameters }))
        .collect();
    if request.cache == CacheHint::Prefix {
        mark_cache(system.last_mut());
        mark_cache(tools.last_mut());
        mark_cache(
            messages
                .last_mut()
                .and_then(|m| m.get_mut("content"))
                .and_then(Value::as_array_mut)
                .and_then(|c| c.last_mut()),
        );
    }
    let budget = request
        .reasoning
        .map(rig2_core::completion::ReasoningEffort::budget_tokens);
    let default_max = card.max_output.unwrap_or(4096).min(8192);
    let max_tokens = match budget {
        Some(budget) => request
            .max_tokens
            .unwrap_or(default_max)
            .max(budget.saturating_add(1024)),
        None => request.max_tokens.unwrap_or(default_max),
    };
    let mut body = json!({ "model": model, "max_tokens": max_tokens, "messages": messages });
    if !system.is_empty() {
        set(&mut body, "system", system);
    }
    if !tools.is_empty() {
        set(&mut body, "tools", tools);
        set(
            &mut body,
            "tool_choice",
            match &request.tool_choice {
                ToolChoice::Auto => json!({ "type": "auto" }),
                ToolChoice::None => json!({ "type": "none" }),
                ToolChoice::Required => json!({ "type": "any" }),
                ToolChoice::Tool(name) => json!({ "type": "tool", "name": name }),
            },
        );
    }
    let mut betas = request
        .extensions
        .get::<Beta>()?
        .map(|b| b.0)
        .unwrap_or_default();
    if let Some(output) = &request.output {
        set(
            &mut body,
            "output_format",
            json!({ "type": "json_schema", "schema": output.schema }),
        );
        betas.push(STRUCTURED_OUTPUTS.to_owned());
    }
    if let Some(budget) = budget {
        set(
            &mut body,
            "thinking",
            json!({ "type": "enabled", "budget_tokens": budget }),
        );
    }
    if let Some(t) = request.temperature {
        set(&mut body, "temperature", t);
    }
    if let Some(p) = request.top_p {
        set(&mut body, "top_p", p);
    }
    if !request.stop.is_empty() {
        set(&mut body, "stop_sequences", request.stop.clone());
    }
    if stream {
        set(&mut body, "stream", true);
    }
    Ok((body, betas))
}

fn u64_at(value: &Value, key: &str) -> u64 {
    value.get(key).and_then(Value::as_u64).unwrap_or(0)
}

fn usage(usage: &Value) -> Usage {
    let (read, write) = (
        u64_at(usage, "cache_read_input_tokens"),
        u64_at(usage, "cache_creation_input_tokens"),
    );
    Usage {
        input_tokens: u64_at(usage, "input_tokens") + read + write,
        output_tokens: u64_at(usage, "output_tokens"),
        cached_input_tokens: read,
        cache_write_tokens: write,
        reasoning_tokens: 0,
    }
}

fn finish_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("end_turn" | "stop_sequence") | None => FinishReason::Stop,
        Some("max_tokens") => FinishReason::Length,
        Some("tool_use") => FinishReason::ToolCalls,
        Some("refusal") => FinishReason::Refusal,
        Some(other) => FinishReason::Other(other.to_owned()),
    }
}

fn citation(value: &Value) -> Citation {
    let text = |key: &str| value.get(key).and_then(Value::as_str).map(str::to_owned);
    Citation {
        cited_text: text("cited_text"),
        title: text("document_title").or_else(|| text("title")),
        url: text("url"),
        document_index: value
            .get("document_index")
            .and_then(Value::as_u64)
            .and_then(|i| u32::try_from(i).ok()),
        ..Citation::default()
    }
}

fn error(provider: &str, value: &Value) -> Error {
    let error = value.get("error").unwrap_or(value);
    let kind = match error.get("type").and_then(Value::as_str) {
        Some("overloaded_error" | "api_error") => ErrorKind::Unavailable,
        Some("rate_limit_error") => ErrorKind::RateLimited,
        Some("invalid_request_error") => ErrorKind::InvalidRequest,
        Some("authentication_error") => ErrorKind::Auth,
        Some("permission_error") => ErrorKind::PermissionDenied,
        Some("not_found_error") => ErrorKind::NotFound,
        _ => ErrorKind::Other,
    };
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("Anthropic reported an error");
    Error::new(kind, message).with_provider(provider)
}

fn finish(doc: &Value, pricing: Option<&Pricing>) -> Finish {
    let usage = doc.get("usage").map(usage).unwrap_or_default();
    Finish {
        usage,
        finish_reason: finish_reason(doc.get("stop_reason").and_then(Value::as_str)),
        id: doc.get("id").and_then(Value::as_str).map(str::to_owned),
        model: doc.get("model").and_then(Value::as_str).map(str::to_owned),
        cost: pricing.map(|p| usage.cost(p)),
        raw: doc.clone(),
    }
}

/// Write a unary Messages reply.
pub(crate) fn write_reply(
    provider: &str,
    doc: &Value,
    pricing: Option<&Pricing>,
    out: &mut StreamWriter,
) -> Result<Finish> {
    if doc.get("type").and_then(Value::as_str) == Some("error") {
        return Err(error(provider, doc));
    }
    for block in doc
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let text = |key: &str| {
            block
                .get(key)
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned()
        };
        match block.get("type").and_then(Value::as_str) {
            Some("text") => out.part(AssistantContent::Text(Text {
                text: text("text"),
                citations: block
                    .get("citations")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .map(citation)
                    .collect(),
                ..Text::default()
            })),
            Some("tool_use") => out.part(AssistantContent::ToolCall(
                rig2_core::content::ToolCall::new(
                    text("id"),
                    text("name"),
                    block.get("input").cloned().unwrap_or_else(|| json!({})),
                ),
            )),
            Some("thinking") => out.part(AssistantContent::Reasoning(Reasoning {
                text: text("thinking"),
                signature: Some(text("signature")),
                ..Reasoning::default()
            })),
            Some("redacted_thinking") => {
                out.part(AssistantContent::Reasoning(Reasoning {
                    encrypted: Some(text("data")),
                    ..Reasoning::default()
                }));
            }
            _ => {}
        }
    }
    Ok(finish(doc, pricing))
}

enum Open {
    Text(TextBlock),
    Tool(ToolCallBlock),
    Thinking(rig2_core::completion::ReasoningBlock),
}

/// Decodes a Messages event stream, and assembles the message document for
/// `raw` from `message_start`, the blocks and `message_delta`.
pub(crate) struct MessagesDecoder {
    provider: &'static str,
    pricing: Option<Pricing>,
    open: BTreeMap<u64, Open>,
    doc: Value,
    blocks: BTreeMap<u64, Value>,
}

impl MessagesDecoder {
    pub(crate) fn new(provider: &'static str, pricing: Option<Pricing>) -> Self {
        Self {
            provider,
            pricing,
            open: BTreeMap::new(),
            doc: json!({}),
            blocks: BTreeMap::new(),
        }
    }

    fn block_mut(&mut self, index: u64) -> &mut Value {
        self.blocks.entry(index).or_insert_with(|| json!({}))
    }

    fn append(&mut self, index: u64, key: &str, fragment: &str) {
        let block = self.block_mut(index);
        let current = block
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        set(block, key, current + fragment);
    }
}

impl StreamDecoder<SseEvent> for MessagesDecoder {
    fn feed(&mut self, frame: SseEvent, out: &mut StreamWriter) -> Result<Option<Finish>> {
        if frame.data.trim().is_empty() {
            return Ok(None);
        }
        let event: Value = serde_json::from_str(&frame.data)?;
        let index = event.get("index").and_then(Value::as_u64).unwrap_or(0);
        match event
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
        {
            "message_start" => {
                self.doc = event.get("message").cloned().unwrap_or_else(|| json!({}));
            }
            "content_block_start" => {
                let block = event
                    .get("content_block")
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                let text = |key: &str| {
                    block
                        .get(key)
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned()
                };
                match block.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        let handle = out.text();
                        out.push(&handle, &text("text"));
                        self.open.insert(index, Open::Text(handle));
                    }
                    Some("tool_use") => {
                        let handle = out.tool_call(text("id"), text("name"));
                        self.open.insert(index, Open::Tool(handle));
                    }
                    Some("thinking") => {
                        self.open.insert(index, Open::Thinking(out.reasoning()));
                    }
                    Some("redacted_thinking") => {
                        let handle = out.reasoning();
                        out.encrypt(&handle, text("data"));
                        self.open.insert(index, Open::Thinking(handle));
                    }
                    _ => {}
                }
                let mut stored = block.clone();
                if stored.get("type").and_then(Value::as_str) == Some("tool_use") {
                    set(&mut stored, "input", "");
                }
                self.blocks.insert(index, stored);
            }
            "content_block_delta" => {
                let delta = event.get("delta").cloned().unwrap_or_else(|| json!({}));
                let text = |key: &str| {
                    delta
                        .get(key)
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned()
                };
                match (
                    delta.get("type").and_then(Value::as_str),
                    self.open.get(&index),
                ) {
                    (Some("text_delta"), Some(Open::Text(handle))) => {
                        out.push(handle, &text("text"));
                        self.append(index, "text", &text("text"));
                    }
                    (Some("input_json_delta"), Some(Open::Tool(handle))) => {
                        out.push(handle, &text("partial_json"));
                        self.append(index, "input", &text("partial_json"));
                    }
                    (Some("thinking_delta"), Some(Open::Thinking(handle))) => {
                        out.push(handle, &text("thinking"));
                        self.append(index, "thinking", &text("thinking"));
                    }
                    (Some("signature_delta"), Some(Open::Thinking(handle))) => {
                        out.sign(handle, text("signature"));
                        self.append(index, "signature", &text("signature"));
                    }
                    (Some("citations_delta"), Some(Open::Text(handle))) => {
                        let cited = delta.get("citation").cloned().unwrap_or(Value::Null);
                        out.cite(handle, citation(&cited));
                        let block = self.block_mut(index);
                        let mut all = block
                            .get("citations")
                            .and_then(Value::as_array)
                            .cloned()
                            .unwrap_or_default();
                        all.push(cited);
                        set(block, "citations", all);
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                match self.open.remove(&index) {
                    Some(Open::Text(handle)) => out.end_text(handle),
                    Some(Open::Tool(handle)) => out.end_tool_call(handle)?,
                    Some(Open::Thinking(handle)) => out.end_reasoning(handle),
                    None => {}
                }
                let block = self.block_mut(index);
                if let Some(input) = block.get("input").and_then(Value::as_str) {
                    let parsed = if input.trim().is_empty() {
                        json!({})
                    } else {
                        serde_json::from_str(input)?
                    };
                    set(block, "input", parsed);
                }
            }
            "message_delta" => {
                if let Some(reason) = event.pointer("/delta/stop_reason") {
                    set(&mut self.doc, "stop_reason", reason.clone());
                }
                let mut usage = self.doc.get("usage").cloned().unwrap_or_else(|| json!({}));
                for (key, value) in event
                    .get("usage")
                    .and_then(Value::as_object)
                    .into_iter()
                    .flatten()
                {
                    if !value.is_null() {
                        set(&mut usage, key, value.clone());
                    }
                }
                set(&mut self.doc, "usage", usage);
            }
            "message_stop" => {
                let blocks: Vec<Value> = std::mem::take(&mut self.blocks).into_values().collect();
                set(&mut self.doc, "content", blocks);
                return Ok(Some(finish(&self.doc, self.pricing.as_ref())));
            }
            "error" => return Err(error(self.provider, &event)),
            _ => {}
        }
        Ok(None)
    }

    fn end(&mut self, _out: &mut StreamWriter) -> Result<Finish> {
        Err(
            Error::new(ErrorKind::Decode, "the stream ended before `message_stop`")
                .with_provider(self.provider),
        )
    }
}

/// A model on the Messages API.
#[derive(Debug, Clone)]
pub struct MessagesModel {
    client: Anthropic,
    info: ModelInfo,
    card: ModelCard,
}

impl MessagesModel {
    pub(crate) fn new(client: Anthropic, model: &str) -> Self {
        let (info, card) = (client.info(model), client.card(model));
        Self { client, info, card }
    }

    fn http_request(
        &self,
        request: &CompletionRequest,
        stream: bool,
    ) -> Result<http::Request<rig2_core::http::Body>> {
        let (body, betas) = request_body(&self.info.model, &self.card, request, stream)?;
        let betas: Vec<&str> = betas.iter().map(String::as_str).collect();
        self.client.request(
            http::Method::POST,
            "/messages",
            &betas,
            Bytes::from(serde_json::to_vec(&body)?),
        )
    }
}

impl Model<Completion> for MessagesModel {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn capabilities(&self) -> ModelCard {
        self.card.clone()
    }

    fn invoke(&self, request: CompletionRequest) -> BoxFuture<'static, Result<CompletionResponse>> {
        let (http, provider, pricing) = (
            Arc::clone(self.client.http()),
            self.client.provider(),
            self.card.pricing,
        );
        let request = self.http_request(&request, false);
        Box::pin(async move {
            let doc = read_json(send(&*http, provider, request?).await?, provider).await?;
            respond(|out| write_reply(provider, &doc, pricing.as_ref(), out))
        })
    }
}

impl StreamingModel<Completion> for MessagesModel {
    fn invoke_stream(
        &self,
        request: CompletionRequest,
    ) -> BoxFuture<'static, Result<BoxStream<'static, Result<StreamEvent>>>> {
        let (http, provider, pricing) = (
            Arc::clone(self.client.http()),
            self.client.provider(),
            self.card.pricing,
        );
        let request = self.http_request(&request, true);
        Box::pin(async move {
            let response = send(&*http, provider, request?).await?;
            Ok(decode_stream(
                sse(response.into_body()),
                MessagesDecoder::new(provider, pricing),
            ))
        })
    }
}

#[cfg(test)]
mod tests;
