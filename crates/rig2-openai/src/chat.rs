//! The Chat Completions API: request mapping, the unary reply, and the
//! streaming decoder. Every OpenAI-compatible provider uses this mapping,
//! adjusted by its [`Quirks`].

use std::collections::BTreeMap;

use rig2_core::catalog::{ModelCard, Pricing};
use rig2_core::completion::{
    Completion, CompletionRequest, CompletionResponse, Finish, FinishReason, ReasoningBlock,
    ReasoningEffort, StreamDecoder, StreamEvent, StreamWriter, TextBlock, ToolCallBlock,
    ToolChoice, Usage, decode_stream, respond,
};
use rig2_core::content::{
    AssistantContent, Citation, Extension, Image, Message, Source, Text, ToolOutput, UserContent,
};
use rig2_core::http::{SseEvent, base64, data_url, read_json, send, set, sse};
use rig2_core::{BoxFuture, BoxStream, Error, ErrorKind, Model, ModelInfo, Result, StreamingModel};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::OpenAI;
use crate::dialect::{Dialect, MaxTokens};

/// Extra top-level fields for the request body.
///
/// For options rig2 has no portable name for: `seed`, `user`,
/// `logit_bias`, provider-specific fields. They are merged last, so they can
/// override mapped fields.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ExtraBody(pub Map<String, Value>);

impl Extension for ExtraBody {
    const KEY: &'static str = "openai.extra_body";
}

/// An image's `detail` level: `low`, `high` or `auto`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageDetail(pub String);

impl Extension for ImageDetail {
    const KEY: &'static str = "openai.detail";
}

pub(crate) fn unsupported(what: &str) -> Error {
    Error::new(
        ErrorKind::Unsupported,
        format!("{what} is not supported by this API"),
    )
}

pub(crate) fn image_url(image: &Image) -> Result<String> {
    match &image.source {
        Source::Url(url) => Ok(url.clone()),
        Source::Bytes(bytes) => {
            let media_type = image
                .media_type
                .as_deref()
                .ok_or_else(|| unsupported("an image without a media type"))?;
            Ok(data_url(media_type, bytes))
        }
    }
}

fn tool_output_text(output: &[ToolOutput]) -> Result<String> {
    let mut parts = Vec::new();
    for part in output {
        match part {
            ToolOutput::Text(text) => parts.push(text.clone()),
            ToolOutput::Json(value) => parts.push(value.to_string()),
            ToolOutput::Image(_) => return Err(unsupported("an image in a tool result")),
        }
    }
    Ok(parts.join("\n"))
}

fn user_parts(content: &[UserContent], messages: &mut Vec<Value>) -> Result<()> {
    let mut parts = Vec::new();
    for part in content {
        match part {
            UserContent::Text(text) => parts.push(json!({ "type": "text", "text": text.text })),
            UserContent::Image(image) => {
                let mut url = json!({ "url": image_url(image)? });
                if let Some(ImageDetail(detail)) = image.extensions.get::<ImageDetail>()? {
                    set(&mut url, "detail", Value::String(detail));
                }
                parts.push(json!({ "type": "image_url", "image_url": url }));
            }
            UserContent::Audio(audio) => {
                let Source::Bytes(bytes) = &audio.source else {
                    return Err(unsupported("audio by URL"));
                };
                let format = audio
                    .media_type
                    .rsplit('/')
                    .next()
                    .unwrap_or("wav")
                    .replace("mpeg", "mp3");
                let data = base64(bytes);
                parts.push(json!({ "type": "input_audio", "input_audio": { "data": data, "format": format } }));
            }
            UserContent::File(file) => {
                let Source::Bytes(bytes) = &file.source else {
                    return Err(unsupported("a file by URL"));
                };
                parts.push(json!({
                    "type": "file",
                    "file": { "file_data": data_url(&file.media_type, bytes), "filename": file.name.clone().unwrap_or_else(|| "document".into()) },
                }));
            }
            UserContent::ToolResult(result) => {
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": result.call_id,
                    "content": tool_output_text(&result.output)?,
                }));
            }
        }
    }
    if !parts.is_empty() {
        messages.push(json!({ "role": "user", "content": parts }));
    }
    Ok(())
}

fn assistant_message(content: &[AssistantContent]) -> Value {
    let text: String = content
        .iter()
        .filter_map(|c| match c {
            AssistantContent::Text(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect();
    let calls: Vec<Value> = content
        .iter()
        .filter_map(|c| match c {
            AssistantContent::ToolCall(call) => Some(json!({
                "id": call.id,
                "type": "function",
                "function": { "name": call.name, "arguments": call.arguments.to_string() },
            })),
            _ => None,
        })
        .collect();
    let mut message = json!({ "role": "assistant", "content": if text.is_empty() { Value::Null } else { Value::String(text) } });
    if !calls.is_empty() {
        set(&mut message, "tool_calls", Value::Array(calls));
    }
    message
}

fn tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::None => json!("none"),
        ToolChoice::Required => json!("required"),
        ToolChoice::Tool(name) => json!({ "type": "function", "function": { "name": name } }),
    }
}

pub(crate) fn effort(effort: ReasoningEffort, minimal: bool) -> &'static str {
    match effort {
        ReasoningEffort::Minimal if minimal => "minimal",
        ReasoningEffort::Minimal | ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
    }
}

/// The Chat Completions request body for `request`.
///
/// Fails with [`ErrorKind::Unsupported`] for content the API cannot carry,
/// such as images in tool results.
pub fn chat_request_body(
    dialect: &Dialect,
    model: &str,
    request: &CompletionRequest,
    stream: bool,
) -> Result<Value> {
    let quirks = &dialect.quirks;
    let mut messages = Vec::new();
    if let Some(system) = &request.system {
        messages.push(json!({ "role": "system", "content": system }));
    }
    for message in &request.messages {
        match message {
            Message::User { content } => user_parts(content, &mut messages)?,
            Message::Assistant { content } => messages.push(assistant_message(content)),
        }
    }
    let mut body = json!({ "model": model, "messages": messages });
    if !request.tools.is_empty() {
        set(&mut body, "tools", request
            .tools
            .iter()
            .map(|t| json!({ "type": "function", "function": { "name": t.name, "description": t.description, "parameters": t.parameters } }))
            .collect::<Vec<Value>>());
        set(&mut body, "tool_choice", tool_choice(&request.tool_choice));
    }
    if let Some(output) = &request.output {
        set(
            &mut body,
            "response_format",
            if quirks.json_schema {
                json!({ "type": "json_schema", "json_schema": { "name": output.name, "schema": output.schema } })
            } else {
                json!({ "type": "json_object" })
            },
        );
    }
    if let Some(max) = request.max_tokens {
        let field = match quirks.max_tokens {
            MaxTokens::MaxCompletionTokens => "max_completion_tokens",
            MaxTokens::MaxTokens => "max_tokens",
        };
        set(&mut body, field, max);
    }
    if let Some(t) = request.temperature {
        set(&mut body, "temperature", json!(t));
    }
    if let Some(p) = request.top_p {
        set(&mut body, "top_p", json!(p));
    }
    if !request.stop.is_empty() {
        set(&mut body, "stop", json!(request.stop));
    }
    if let (Some(e), true) = (request.reasoning, quirks.reasoning_effort) {
        set(
            &mut body,
            "reasoning_effort",
            json!(effort(e, dialect.name == "openai")),
        );
    }
    if stream {
        set(&mut body, "stream", json!(true));
        if quirks.stream_usage {
            set(
                &mut body,
                "stream_options",
                json!({ "include_usage": true }),
            );
        }
    }
    if let (Some(ExtraBody(extra)), Some(object)) =
        (request.extensions.get::<ExtraBody>()?, body.as_object_mut())
    {
        object.extend(extra);
    }
    Ok(body)
}

pub(crate) fn finish_reason(reason: &str) -> FinishReason {
    match reason {
        "stop" | "end_turn" => FinishReason::Stop,
        "length" | "max_tokens" => FinishReason::Length,
        "tool_calls" | "function_call" => FinishReason::ToolCalls,
        "content_filter" => FinishReason::ContentFilter,
        other => FinishReason::Other(other.to_owned()),
    }
}

fn u64_at(value: &Value, pointer: &str) -> u64 {
    value.pointer(pointer).and_then(Value::as_u64).unwrap_or(0)
}

pub(crate) fn chat_usage(usage: &Value) -> Usage {
    Usage {
        input_tokens: u64_at(usage, "/prompt_tokens"),
        output_tokens: u64_at(usage, "/completion_tokens"),
        cached_input_tokens: u64_at(usage, "/prompt_tokens_details/cached_tokens")
            .max(u64_at(usage, "/prompt_cache_hit_tokens")),
        cache_write_tokens: 0,
        reasoning_tokens: u64_at(usage, "/completion_tokens_details/reasoning_tokens"),
    }
}

/// An error reported inside a stream or a 200 body.
pub(crate) fn body_error(provider: &str, error: &Value) -> Error {
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("the provider reported an error");
    let code = [error.get("code"), error.get("type")]
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>()
        .join(" ");
    let kind = if code.contains("rate_limit") {
        ErrorKind::RateLimited
    } else if code.contains("server") || code.contains("overloaded") || code.contains("unavailable")
    {
        ErrorKind::Unavailable
    } else if code.contains("invalid") {
        ErrorKind::InvalidRequest
    } else if code.contains("auth") || code.contains("key") {
        ErrorKind::Auth
    } else {
        ErrorKind::Other
    };
    Error::new(kind, message).with_provider(provider)
}

fn citations(annotations: Option<&Value>) -> Vec<Citation> {
    annotations
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|a| {
            let cite = a.get("url_citation").unwrap_or(a);
            let url = cite.get("url")?.as_str()?.to_owned();
            Some(Citation {
                url: Some(url),
                title: cite.get("title").and_then(Value::as_str).map(str::to_owned),
                ..Citation::default()
            })
        })
        .collect()
}

/// The reasoning and text of a message or delta.
///
/// `content` is usually a string. Some providers (Mistral's reasoning
/// models) send an array of chunks instead, with `thinking` chunks for the
/// reasoning; others send reasoning as `reasoning_content` or `reasoning`.
fn pieces(message: &Value) -> (String, String) {
    let mut reasoning = message
        .get("reasoning_content")
        .or_else(|| message.get("reasoning"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let mut text = String::new();
    match message.get("content") {
        Some(Value::String(content)) => text.push_str(content),
        Some(Value::Array(chunks)) => {
            for chunk in chunks {
                match chunk.get("type").and_then(Value::as_str) {
                    Some("text") => text.push_str(
                        chunk
                            .get("text")
                            .and_then(Value::as_str)
                            .unwrap_or_default(),
                    ),
                    Some("thinking") => match chunk.get("thinking") {
                        Some(Value::String(thinking)) => reasoning.push_str(thinking),
                        Some(Value::Array(parts)) => {
                            for part in parts {
                                reasoning.push_str(
                                    part.get("text").and_then(Value::as_str).unwrap_or_default(),
                                );
                            }
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }
        }
        _ => {}
    }
    (reasoning, text)
}

/// Write a unary Chat Completions reply.
pub(crate) fn write_reply(
    provider: &str,
    doc: &Value,
    pricing: Option<&Pricing>,
    out: &mut StreamWriter,
) -> Result<Finish> {
    if let Some(error) = doc.get("error") {
        return Err(body_error(provider, error));
    }
    let choice = doc
        .pointer("/choices/0")
        .ok_or_else(|| Error::new(ErrorKind::Decode, "the reply has no choices"))?;
    let message = choice.get("message").unwrap_or(&Value::Null);
    let (reasoning, content) = pieces(message);
    if !reasoning.is_empty() {
        out.part(AssistantContent::Reasoning(rig2_core::content::Reasoning {
            text: reasoning,
            ..Default::default()
        }));
    }
    let refusal = message
        .get("refusal")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let text = if content.is_empty() {
        refusal
    } else {
        content.as_str()
    };
    if !text.is_empty() {
        out.part(AssistantContent::Text(Text {
            text: text.to_owned(),
            citations: citations(message.get("annotations")),
            ..Text::default()
        }));
    }
    for call in message
        .get("tool_calls")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let id = call.get("id").and_then(Value::as_str).unwrap_or_default();
        let name = call
            .pointer("/function/name")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let block = out.tool_call(id, name);
        out.push(
            &block,
            call.pointer("/function/arguments")
                .and_then(Value::as_str)
                .unwrap_or_default(),
        );
        out.end_tool_call(block)?;
    }
    let usage = doc.get("usage").map(chat_usage).unwrap_or_default();
    let refused = message
        .get("refusal")
        .and_then(Value::as_str)
        .is_some_and(|r| !r.is_empty());
    let reason = if refused {
        FinishReason::Refusal
    } else {
        choice
            .get("finish_reason")
            .and_then(Value::as_str)
            .map_or(FinishReason::Stop, finish_reason)
    };
    Ok(Finish {
        usage,
        finish_reason: reason,
        id: doc.get("id").and_then(Value::as_str).map(str::to_owned),
        model: doc.get("model").and_then(Value::as_str).map(str::to_owned),
        cost: pricing.map(|p| usage.cost(p)),
        raw: doc.clone(),
    })
}

struct PendingCall {
    id: String,
    name: String,
    arguments: String,
    block: Option<ToolCallBlock>,
}

/// Decodes a Chat Completions stream, and assembles the equivalent unary
/// reply document for `raw`.
pub(crate) struct ChatDecoder {
    provider: &'static str,
    pricing: Option<Pricing>,
    stream_usage: bool,
    text: Option<TextBlock>,
    reasoning: Option<ReasoningBlock>,
    calls: BTreeMap<u64, PendingCall>,
    all_text: String,
    all_reasoning: String,
    usage: Option<Value>,
    reason: Option<String>,
    id: Option<String>,
    model: Option<String>,
    created: Option<Value>,
}

impl ChatDecoder {
    pub(crate) fn new(
        provider: &'static str,
        pricing: Option<Pricing>,
        stream_usage: bool,
    ) -> Self {
        Self {
            provider,
            pricing,
            stream_usage,
            text: None,
            reasoning: None,
            calls: BTreeMap::new(),
            all_text: String::new(),
            all_reasoning: String::new(),
            usage: None,
            reason: None,
            id: None,
            model: None,
            created: None,
        }
    }

    fn finish(&mut self, out: &mut StreamWriter) -> Result<Finish> {
        if let Some(block) = self.reasoning.take() {
            out.end_reasoning(block);
        }
        if let Some(block) = self.text.take() {
            out.end_text(block);
        }
        let mut tool_calls = Vec::new();
        for call in std::mem::take(&mut self.calls).into_values() {
            let block = match call.block {
                Some(block) => block,
                None => {
                    let block = out.tool_call(&call.id, &call.name);
                    out.push(&block, &call.arguments);
                    block
                }
            };
            out.end_tool_call(block)?;
            tool_calls.push(json!({
                "id": call.id,
                "type": "function",
                "function": { "name": call.name, "arguments": call.arguments },
            }));
        }
        let usage = self.usage.as_ref().map(chat_usage).unwrap_or_default();
        let reason = self.reason.clone().unwrap_or_else(|| "stop".into());
        let mut message = json!({ "role": "assistant", "content": if self.all_text.is_empty() { Value::Null } else { Value::String(self.all_text.clone()) } });
        if !self.all_reasoning.is_empty() {
            set(
                &mut message,
                "reasoning_content",
                Value::String(self.all_reasoning.clone()),
            );
        }
        if !tool_calls.is_empty() {
            set(&mut message, "tool_calls", Value::Array(tool_calls));
        }
        let raw = json!({
            "id": self.id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [{ "index": 0, "message": message, "finish_reason": reason }],
            "usage": self.usage,
        });
        Ok(Finish {
            usage,
            finish_reason: finish_reason(&reason),
            id: self.id.clone(),
            model: self.model.clone(),
            cost: self.pricing.as_ref().map(|p| usage.cost(p)),
            raw,
        })
    }
}

impl StreamDecoder<SseEvent> for ChatDecoder {
    fn feed(&mut self, frame: SseEvent, out: &mut StreamWriter) -> Result<Option<Finish>> {
        let data = frame.data.trim();
        if data == "[DONE]" {
            return self.finish(out).map(Some);
        }
        if data.is_empty() {
            return Ok(None);
        }
        let chunk: Value = serde_json::from_str(data)?;
        if let Some(error) = chunk.get("error") {
            return Err(body_error(self.provider, error));
        }
        if self.id.is_none() {
            self.id = chunk.get("id").and_then(Value::as_str).map(str::to_owned);
            self.model = chunk
                .get("model")
                .and_then(Value::as_str)
                .map(str::to_owned);
            self.created = chunk.get("created").cloned();
        }
        if let Some(usage) = chunk.get("usage").filter(|u| !u.is_null()) {
            self.usage = Some(usage.clone());
        }
        let Some(choice) = chunk.pointer("/choices/0") else {
            return Ok(None);
        };
        let delta = choice.get("delta").unwrap_or(&Value::Null);
        let (reasoning, text) = pieces(delta);
        if !reasoning.is_empty() {
            let block = self.reasoning.get_or_insert_with(|| out.reasoning());
            out.push(block, &reasoning);
            self.all_reasoning.push_str(&reasoning);
        }
        if !text.is_empty() {
            if let Some(block) = self.reasoning.take() {
                out.end_reasoning(block);
            }
            let block = self.text.get_or_insert_with(|| out.text());
            out.push(block, &text);
            self.all_text.push_str(&text);
        }
        for call in delta
            .get("tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let index = call.get("index").and_then(Value::as_u64).unwrap_or(0);
            let pending = self.calls.entry(index).or_insert_with(|| PendingCall {
                id: String::new(),
                name: String::new(),
                arguments: String::new(),
                block: None,
            });
            if let Some(id) = call
                .get("id")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            {
                id.clone_into(&mut pending.id);
            }
            if let Some(name) = call
                .pointer("/function/name")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            {
                name.clone_into(&mut pending.name);
            }
            let fragment = call
                .pointer("/function/arguments")
                .and_then(Value::as_str)
                .unwrap_or_default();
            pending.arguments.push_str(fragment);
            match &pending.block {
                Some(block) => out.push(block, fragment),
                None if !pending.name.is_empty() && !pending.id.is_empty() => {
                    let block = out.tool_call(&pending.id, &pending.name);
                    out.push(&block, &pending.arguments);
                    pending.block = Some(block);
                }
                None => {}
            }
        }
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            self.reason = Some(reason.to_owned());
            if !self.stream_usage {
                return self.finish(out).map(Some);
            }
        }
        Ok(None)
    }

    fn end(&mut self, out: &mut StreamWriter) -> Result<Finish> {
        if self.reason.is_some() {
            self.finish(out)
        } else {
            Err(Error::new(
                ErrorKind::Decode,
                "the stream ended before the reply finished",
            )
            .with_provider(self.provider))
        }
    }
}

/// A model on the Chat Completions API.
#[derive(Debug, Clone)]
pub struct ChatModel {
    client: OpenAI,
    info: ModelInfo,
    card: ModelCard,
}

impl ChatModel {
    pub(crate) fn new(client: OpenAI, model: &str) -> Self {
        let (info, card) = (client.info(model), client.card(model));
        Self { client, info, card }
    }

    fn body(
        &self,
        request: &CompletionRequest,
        stream: bool,
    ) -> Result<http::Request<rig2_core::http::Body>> {
        let body = chat_request_body(self.client.dialect(), &self.info.model, request, stream)?;
        self.client.post_json("/chat/completions", &body)
    }
}

impl Model<Completion> for ChatModel {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn capabilities(&self) -> ModelCard {
        self.card.clone()
    }

    fn invoke(&self, request: CompletionRequest) -> BoxFuture<'static, Result<CompletionResponse>> {
        let http = std::sync::Arc::clone(self.client.http());
        let (provider, pricing) = (self.client.provider(), self.card.pricing);
        let request = self.body(&request, false);
        Box::pin(async move {
            let response = send(&*http, provider, request?).await?;
            let doc = read_json(response, provider).await?;
            respond(|out| write_reply(provider, &doc, pricing.as_ref(), out))
        })
    }
}

impl StreamingModel<Completion> for ChatModel {
    fn invoke_stream(
        &self,
        request: CompletionRequest,
    ) -> BoxFuture<'static, Result<BoxStream<'static, Result<StreamEvent>>>> {
        let http = std::sync::Arc::clone(self.client.http());
        let (provider, pricing) = (self.client.provider(), self.card.pricing);
        let stream_usage = self.client.dialect().quirks.stream_usage;
        let request = self.body(&request, true);
        Box::pin(async move {
            let response = send(&*http, provider, request?).await?;
            Ok(decode_stream(
                sse(response.into_body()),
                ChatDecoder::new(provider, pricing, stream_usage),
            ))
        })
    }
}

#[cfg(test)]
mod tests;
