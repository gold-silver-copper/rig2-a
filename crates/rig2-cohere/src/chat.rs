//! Cohere chat (API v2): request mapping, the unary reply and the streaming
//! decoder. Cohere's tool plan, the text it writes before calling tools, is
//! reasoning.

use std::collections::BTreeMap;
use std::sync::Arc;

use rig2_core::catalog::{ModelCard, Pricing};
use rig2_core::completion::{
    Completion, CompletionRequest, CompletionResponse, Finish, FinishReason, ReasoningBlock,
    StreamDecoder, StreamEvent, StreamWriter, TextBlock, ToolCallBlock, ToolChoice, Usage,
    decode_stream, respond,
};
use rig2_core::content::{
    AssistantContent, Citation, Message, Reasoning, Text, ToolOutput, UserContent,
};
use rig2_core::http::{SseEvent, data_url, read_json, send, set, sse};
use rig2_core::{BoxFuture, BoxStream, Error, ErrorKind, Model, ModelInfo, Result, StreamingModel};
use serde_json::{Value, json};

use crate::{Cohere, PROVIDER};

fn unsupported(what: &str) -> Error {
    Error::new(
        ErrorKind::Unsupported,
        format!("{what} is not supported by Cohere"),
    )
}

fn user_messages(content: &[UserContent], messages: &mut Vec<Value>) -> Result<()> {
    let mut parts = Vec::new();
    for part in content {
        match part {
            UserContent::Text(text) => parts.push(json!({ "type": "text", "text": text.text })),
            UserContent::Image(image) => {
                let url = match image.bytes() {
                    Some((media_type, bytes)) => data_url(&media_type, &bytes),
                    None => match &image.source {
                        rig2_core::content::Source::Url(url) => url.clone(),
                        rig2_core::content::Source::Bytes(_) => {
                            return Err(unsupported("this image"));
                        }
                    },
                };
                parts.push(json!({ "type": "image_url", "image_url": { "url": url } }));
            }
            UserContent::Audio(_) => return Err(unsupported("audio input")),
            UserContent::File(_) => return Err(unsupported("file input")),
            UserContent::ToolResult(result) => {
                let documents: Vec<Value> = result
                    .output
                    .iter()
                    .map(|o| match o {
                        ToolOutput::Text(text) => Ok(json!({ "type": "document", "document": { "data": text } })),
                        ToolOutput::Json(value) => {
                            Ok(json!({ "type": "document", "document": { "data": value.to_string() } }))
                        }
                        ToolOutput::Image(_) => Err(unsupported("an image in a tool result")),
                    })
                    .collect::<Result<_>>()?;
                messages.push(
                    json!({ "role": "tool", "tool_call_id": result.call_id, "content": documents }),
                );
            }
        }
    }
    if !parts.is_empty() {
        messages.push(json!({ "role": "user", "content": parts }));
    }
    Ok(())
}

fn assistant_message(content: &[AssistantContent]) -> Value {
    let mut message = json!({ "role": "assistant" });
    let text: Vec<Value> = content
        .iter()
        .filter_map(|c| match c {
            AssistantContent::Text(t) if !t.text.is_empty() => {
                Some(json!({ "type": "text", "text": t.text }))
            }
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
    let plan: String = content
        .iter()
        .filter_map(|c| match c {
            AssistantContent::Reasoning(r) => Some(r.text.as_str()),
            _ => None,
        })
        .collect();
    if !text.is_empty() {
        set(&mut message, "content", text);
    }
    if !calls.is_empty() {
        set(&mut message, "tool_calls", calls);
        if !plan.is_empty() {
            set(&mut message, "tool_plan", plan);
        }
    }
    message
}

/// The chat request body for `request`.
pub(crate) fn request_body(
    model: &str,
    request: &CompletionRequest,
    stream: bool,
) -> Result<Value> {
    let mut messages = Vec::new();
    if let Some(system) = &request.system {
        messages.push(json!({ "role": "system", "content": system }));
    }
    for message in &request.messages {
        match message {
            Message::User { content } => user_messages(content, &mut messages)?,
            Message::Assistant { content } => messages.push(assistant_message(content)),
        }
    }
    let mut body = json!({ "model": model, "messages": messages, "stream": stream });
    if !request.tools.is_empty() {
        let tools: Vec<Value> = request
            .tools
            .iter()
            .map(|t| json!({ "type": "function", "function": { "name": t.name, "description": t.description, "parameters": t.parameters } }))
            .collect();
        set(&mut body, "tools", tools);
        match &request.tool_choice {
            ToolChoice::Auto => {}
            ToolChoice::None => set(&mut body, "tool_choice", "NONE"),
            ToolChoice::Required | ToolChoice::Tool(_) => set(&mut body, "tool_choice", "REQUIRED"),
        }
    }
    if let Some(output) = &request.output {
        set(
            &mut body,
            "response_format",
            json!({ "type": "json_object", "json_schema": output.schema }),
        );
    }
    if let Some(max) = request.max_tokens {
        set(&mut body, "max_tokens", max);
    }
    if let Some(t) = request.temperature {
        set(&mut body, "temperature", t);
    }
    if let Some(p) = request.top_p {
        set(&mut body, "p", p);
    }
    if !request.stop.is_empty() {
        set(&mut body, "stop_sequences", request.stop.clone());
    }
    if let Some(effort) = request.reasoning {
        set(
            &mut body,
            "thinking",
            json!({ "type": "enabled", "token_budget": effort.budget_tokens() }),
        );
    }
    Ok(body)
}

fn usage(usage: &Value) -> Usage {
    let tokens = usage
        .get("tokens")
        .or_else(|| usage.get("billed_units"))
        .unwrap_or(&Value::Null);
    let at = |key: &str| tokens.get(key).and_then(Value::as_u64).unwrap_or(0);
    Usage {
        input_tokens: at("input_tokens"),
        output_tokens: at("output_tokens"),
        ..Usage::default()
    }
}

fn finish_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("COMPLETE" | "STOP_SEQUENCE") | None => FinishReason::Stop,
        Some("MAX_TOKENS") => FinishReason::Length,
        Some("TOOL_CALL") => FinishReason::ToolCalls,
        Some(other) => FinishReason::Other(other.to_owned()),
    }
}

fn finish(doc: &Value, pricing: Option<&Pricing>) -> Result<Finish> {
    if doc.get("finish_reason").and_then(Value::as_str) == Some("ERROR") {
        let message = doc
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("Cohere reported an error");
        return Err(Error::new(ErrorKind::Other, message).with_provider(PROVIDER));
    }
    let usage = doc.get("usage").map(usage).unwrap_or_default();
    Ok(Finish {
        usage,
        finish_reason: finish_reason(doc.get("finish_reason").and_then(Value::as_str)),
        id: doc.get("id").and_then(Value::as_str).map(str::to_owned),
        model: None,
        cost: pricing.map(|p| usage.cost(p)),
        raw: doc.clone(),
    })
}

fn citations(message: &Value) -> Vec<Citation> {
    message
        .get("citations")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|c| Citation {
            cited_text: c.get("text").and_then(Value::as_str).map(str::to_owned),
            ..Citation::default()
        })
        .collect()
}

/// Write a unary chat reply.
pub(crate) fn write_reply(
    doc: &Value,
    pricing: Option<&Pricing>,
    out: &mut StreamWriter,
) -> Result<Finish> {
    let message = doc.get("message").unwrap_or(&Value::Null);
    let plan = message
        .get("tool_plan")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if !plan.is_empty() {
        out.part(AssistantContent::Reasoning(Reasoning {
            text: plan.to_owned(),
            ..Reasoning::default()
        }));
    }
    let mut cited = false;
    for content in message
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        match content.get("type").and_then(Value::as_str) {
            Some("thinking") => out.part(AssistantContent::Reasoning(Reasoning {
                text: content
                    .get("thinking")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                ..Reasoning::default()
            })),
            Some("text") => {
                let citations = if cited {
                    Vec::new()
                } else {
                    citations(message)
                };
                cited = true;
                out.part(AssistantContent::Text(Text {
                    text: content
                        .get("text")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    citations,
                    ..Text::default()
                }));
            }
            _ => {}
        }
    }
    for call in message
        .get("tool_calls")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let block = out.tool_call(
            call.get("id").and_then(Value::as_str).unwrap_or_default(),
            call.pointer("/function/name")
                .and_then(Value::as_str)
                .unwrap_or_default(),
        );
        out.push(
            &block,
            call.pointer("/function/arguments")
                .and_then(Value::as_str)
                .unwrap_or_default(),
        );
        out.end_tool_call(block)?;
    }
    finish(doc, pricing)
}

enum Open {
    Text(TextBlock),
    Thinking(ReasoningBlock),
}

/// Decodes a chat stream, and assembles the unary reply document for `raw`.
pub(crate) struct ChatDecoder {
    pricing: Option<Pricing>,
    content: BTreeMap<u64, Open>,
    calls: BTreeMap<u64, ToolCallBlock>,
    plan: Option<ReasoningBlock>,
    id: Option<String>,
    texts: BTreeMap<u64, (String, String)>,
    plan_text: String,
    tool_calls: BTreeMap<u64, (String, String, String)>,
}

impl ChatDecoder {
    pub(crate) fn new(pricing: Option<Pricing>) -> Self {
        Self {
            pricing,
            content: BTreeMap::new(),
            calls: BTreeMap::new(),
            plan: None,
            id: None,
            texts: BTreeMap::new(),
            plan_text: String::new(),
            tool_calls: BTreeMap::new(),
        }
    }

    fn document(&self, end: &Value) -> Value {
        let content: Vec<Value> = self
            .texts
            .values()
            .map(|(kind, text)| {
                if kind == "thinking" {
                    json!({ "type": "thinking", "thinking": text })
                } else {
                    json!({ "type": "text", "text": text })
                }
            })
            .collect();
        let mut message = json!({ "role": "assistant", "content": content });
        if !self.plan_text.is_empty() {
            set(&mut message, "tool_plan", self.plan_text.clone());
        }
        if !self.tool_calls.is_empty() {
            let calls: Vec<Value> = self
                .tool_calls
                .values()
                .map(|(id, name, arguments)| json!({ "id": id, "type": "function", "function": { "name": name, "arguments": arguments } }))
                .collect();
            set(&mut message, "tool_calls", calls);
        }
        json!({
            "id": self.id,
            "finish_reason": end.pointer("/delta/finish_reason"),
            "message": message,
            "usage": end.pointer("/delta/usage"),
        })
    }
}

impl StreamDecoder<SseEvent> for ChatDecoder {
    fn feed(&mut self, frame: SseEvent, out: &mut StreamWriter) -> Result<Option<Finish>> {
        if frame.data.trim().is_empty() {
            return Ok(None);
        }
        let event: Value = serde_json::from_str(&frame.data)?;
        let index = event.get("index").and_then(Value::as_u64).unwrap_or(0);
        let message = event.pointer("/delta/message").unwrap_or(&Value::Null);
        match event
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
        {
            "message-start" => self.id = event.get("id").and_then(Value::as_str).map(str::to_owned),
            "content-start" => {
                let kind = message
                    .pointer("/content/type")
                    .and_then(Value::as_str)
                    .unwrap_or("text")
                    .to_owned();
                if let Some(block) = self.plan.take() {
                    out.end_reasoning(block);
                }
                let open = if kind == "thinking" {
                    Open::Thinking(out.reasoning())
                } else {
                    Open::Text(out.text())
                };
                self.content.insert(index, open);
                self.texts.insert(index, (kind, String::new()));
            }
            "content-delta" => {
                let fragment = message
                    .pointer("/content/text")
                    .or_else(|| message.pointer("/content/thinking"))
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                match self.content.get(&index) {
                    Some(Open::Text(block)) => out.push(block, fragment),
                    Some(Open::Thinking(block)) => out.push(block, fragment),
                    None => {}
                }
                if let Some((_, text)) = self.texts.get_mut(&index) {
                    text.push_str(fragment);
                }
            }
            "content-end" => match self.content.remove(&index) {
                Some(Open::Text(block)) => out.end_text(block),
                Some(Open::Thinking(block)) => out.end_reasoning(block),
                None => {}
            },
            "tool-plan-delta" => {
                let fragment = message
                    .get("tool_plan")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let block = self.plan.get_or_insert_with(|| out.reasoning());
                out.push(block, fragment);
                self.plan_text.push_str(fragment);
            }
            "tool-call-start" => {
                if let Some(block) = self.plan.take() {
                    out.end_reasoning(block);
                }
                let call = message.get("tool_calls").unwrap_or(&Value::Null);
                let id = call
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                let name = call
                    .pointer("/function/name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                let arguments = call
                    .pointer("/function/arguments")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                let block = out.tool_call(&id, &name);
                out.push(&block, &arguments);
                self.calls.insert(index, block);
                self.tool_calls.insert(index, (id, name, arguments));
            }
            "tool-call-delta" => {
                let fragment = message
                    .pointer("/tool_calls/function/arguments")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if let Some(block) = self.calls.get(&index) {
                    out.push(block, fragment);
                }
                if let Some((_, _, arguments)) = self.tool_calls.get_mut(&index) {
                    arguments.push_str(fragment);
                }
            }
            "tool-call-end" => {
                if let Some(block) = self.calls.remove(&index) {
                    out.end_tool_call(block)?;
                }
            }
            "message-end" => {
                return finish(&self.document(&event), self.pricing.as_ref()).map(Some);
            }
            _ => {}
        }
        Ok(None)
    }

    fn end(&mut self, _out: &mut StreamWriter) -> Result<Finish> {
        Err(
            Error::new(ErrorKind::Decode, "the stream ended before `message-end`")
                .with_provider(PROVIDER),
        )
    }
}

/// A Cohere chat model.
#[derive(Debug, Clone)]
pub struct ChatModel {
    client: Cohere,
    info: ModelInfo,
    card: ModelCard,
}

impl ChatModel {
    pub(crate) fn new(client: Cohere, model: &str) -> Self {
        let card = client.card(model);
        Self {
            client,
            info: Cohere::info(model),
            card,
        }
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
        let (http, pricing) = (Arc::clone(self.client.http()), self.card.pricing);
        let request = request_body(&self.info.model, &request, false)
            .and_then(|b| self.client.post_json("/chat", &b));
        Box::pin(async move {
            let doc = read_json(send(&*http, PROVIDER, request?).await?, PROVIDER).await?;
            respond(|out| write_reply(&doc, pricing.as_ref(), out))
        })
    }
}

impl StreamingModel<Completion> for ChatModel {
    fn invoke_stream(
        &self,
        request: CompletionRequest,
    ) -> BoxFuture<'static, Result<BoxStream<'static, Result<StreamEvent>>>> {
        let (http, pricing) = (Arc::clone(self.client.http()), self.card.pricing);
        let request = request_body(&self.info.model, &request, true)
            .and_then(|b| self.client.post_json("/chat", &b));
        Box::pin(async move {
            let response = send(&*http, PROVIDER, request?).await?;
            Ok(decode_stream(
                sse(response.into_body()),
                ChatDecoder::new(pricing),
            ))
        })
    }
}

#[cfg(test)]
mod tests;
