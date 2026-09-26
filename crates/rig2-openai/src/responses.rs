//! The Responses API: request mapping, the unary reply, and the streaming
//! decoder. Requests are stateless (`store: false`); reasoning is carried
//! between turns as encrypted content.

use std::collections::BTreeMap;

use rig2_core::catalog::{ModelCard, Pricing};
use rig2_core::completion::{
    Completion, CompletionRequest, CompletionResponse, Finish, FinishReason, ReasoningBlock,
    StreamDecoder, StreamEvent, StreamWriter, TextBlock, ToolCallBlock, ToolChoice, Usage,
    decode_stream, respond,
};
use rig2_core::content::{
    AssistantContent, Citation, Extension, Extensions, Message, Reasoning, Source, Text,
    ToolOutput, UserContent,
};
use rig2_core::http::{SseEvent, data_url, read_json, send, set, sse};
use rig2_core::{BoxFuture, BoxStream, Error, ErrorKind, Model, ModelInfo, Result, StreamingModel};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::OpenAI;
use crate::chat::{ExtraBody, ImageDetail, body_error, effort, image_url, unsupported};

/// The id of a Responses output item, kept so it can be sent back.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ItemId(pub String);

impl Extension for ItemId {
    const KEY: &'static str = "openai.item_id";
}

/// Ask for reasoning summaries (`auto`, `concise` or `detailed`). Some
/// accounts need verification before OpenAI returns them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReasoningSummary(pub String);

impl Extension for ReasoningSummary {
    const KEY: &'static str = "openai.reasoning_summary";
}

fn tool_output(output: &[ToolOutput]) -> Result<String> {
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

fn user_items(content: &[UserContent], input: &mut Vec<Value>) -> Result<()> {
    let mut parts = Vec::new();
    for part in content {
        match part {
            UserContent::Text(text) => {
                parts.push(json!({ "type": "input_text", "text": text.text }));
            }
            UserContent::Image(image) => {
                let detail = image
                    .extensions
                    .get::<ImageDetail>()?
                    .map_or_else(|| "auto".to_owned(), |d| d.0);
                parts.push(json!({ "type": "input_image", "image_url": image_url(image)?, "detail": detail }));
            }
            UserContent::File(file) => match &file.source {
                Source::Bytes(bytes) => parts.push(json!({
                    "type": "input_file",
                    "file_data": data_url(&file.media_type, bytes),
                    "filename": file.name.clone().unwrap_or_else(|| "document.pdf".into()),
                })),
                Source::Url(url) => parts.push(json!({ "type": "input_file", "file_url": url })),
            },
            UserContent::Audio(_) => return Err(unsupported("audio input")),
            UserContent::ToolResult(result) => input.push(json!({
                "type": "function_call_output",
                "call_id": result.call_id,
                "output": tool_output(&result.output)?,
            })),
        }
    }
    if !parts.is_empty() {
        input.push(json!({ "role": "user", "content": parts }));
    }
    Ok(())
}

fn assistant_items(content: &[AssistantContent], input: &mut Vec<Value>) -> Result<()> {
    for part in content {
        match part {
            AssistantContent::Text(text) => input.push(json!({
                "role": "assistant",
                "content": [{ "type": "output_text", "text": text.text }],
            })),
            AssistantContent::ToolCall(call) => input.push(json!({
                "type": "function_call",
                "call_id": call.id,
                "name": call.name,
                "arguments": call.arguments.to_string(),
            })),
            AssistantContent::Reasoning(reasoning) => {
                if let Some(encrypted) = &reasoning.encrypted {
                    let mut item = json!({
                        "type": "reasoning",
                        "summary": if reasoning.text.is_empty() { json!([]) } else { json!([{ "type": "summary_text", "text": reasoning.text }]) },
                        "encrypted_content": encrypted,
                    });
                    if let Some(ItemId(id)) = reasoning.extensions.get::<ItemId>()? {
                        set(&mut item, "id", Value::String(id));
                    }
                    input.push(item);
                }
            }
            AssistantContent::Image(_) => {}
        }
    }
    Ok(())
}

/// The Responses request body for `request`.
pub(crate) fn request_body(
    model: &str,
    request: &CompletionRequest,
    stream: bool,
) -> Result<Value> {
    if !request.stop.is_empty() {
        return Err(unsupported("stop sequences"));
    }
    let mut input = Vec::new();
    for message in &request.messages {
        match message {
            Message::User { content } => user_items(content, &mut input)?,
            Message::Assistant { content } => assistant_items(content, &mut input)?,
        }
    }
    let mut body = json!({
        "model": model,
        "input": input,
        "store": false,
        "include": ["reasoning.encrypted_content"],
    });
    if let Some(system) = &request.system {
        set(&mut body, "instructions", json!(system));
    }
    if !request.tools.is_empty() {
        set(&mut body, "tools", request
            .tools
            .iter()
            .map(|t| json!({ "type": "function", "name": t.name, "description": t.description, "parameters": t.parameters, "strict": false }))
            .collect::<Vec<Value>>());
        set(
            &mut body,
            "tool_choice",
            match &request.tool_choice {
                ToolChoice::Auto => json!("auto"),
                ToolChoice::None => json!("none"),
                ToolChoice::Required => json!("required"),
                ToolChoice::Tool(name) => json!({ "type": "function", "name": name }),
            },
        );
    }
    if let Some(output) = &request.output {
        set(
            &mut body,
            "text",
            json!({ "format": { "type": "json_schema", "name": output.name, "schema": output.schema, "strict": false } }),
        );
    }
    if let Some(max) = request.max_tokens {
        set(&mut body, "max_output_tokens", json!(max));
    }
    if let Some(t) = request.temperature {
        set(&mut body, "temperature", json!(t));
    }
    if let Some(p) = request.top_p {
        set(&mut body, "top_p", json!(p));
    }
    let summary = request.extensions.get::<ReasoningSummary>()?;
    if request.reasoning.is_some() || summary.is_some() {
        let mut reasoning = json!({});
        if let Some(e) = request.reasoning {
            set(&mut reasoning, "effort", json!(effort(e, true)));
        }
        if let Some(ReasoningSummary(summary)) = summary {
            set(&mut reasoning, "summary", json!(summary));
        }
        set(&mut body, "reasoning", reasoning);
    }
    if stream {
        set(&mut body, "stream", json!(true));
    }
    if let (Some(ExtraBody(extra)), Some(object)) =
        (request.extensions.get::<ExtraBody>()?, body.as_object_mut())
    {
        object.extend(extra);
    }
    Ok(body)
}

fn u64_at(value: &Value, pointer: &str) -> u64 {
    value.pointer(pointer).and_then(Value::as_u64).unwrap_or(0)
}

fn usage(response: &Value) -> Usage {
    let usage = response.get("usage").unwrap_or(&Value::Null);
    Usage {
        input_tokens: u64_at(usage, "/input_tokens"),
        output_tokens: u64_at(usage, "/output_tokens"),
        cached_input_tokens: u64_at(usage, "/input_tokens_details/cached_tokens"),
        cache_write_tokens: 0,
        reasoning_tokens: u64_at(usage, "/output_tokens_details/reasoning_tokens"),
    }
}

fn citations(content: &Value) -> Vec<Citation> {
    content
        .get("annotations")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|a| {
            Some(Citation {
                url: Some(a.get("url")?.as_str()?.to_owned()),
                title: a.get("title").and_then(Value::as_str).map(str::to_owned),
                ..Citation::default()
            })
        })
        .collect()
}

fn reasoning_part(item: &Value) -> Result<Reasoning> {
    let summary: String = item
        .get("summary")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .chain(
            item.get("content")
                .and_then(Value::as_array)
                .into_iter()
                .flatten(),
        )
        .filter_map(|s| s.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("\n\n");
    let mut extensions = Extensions::default();
    if let Some(id) = item.get("id").and_then(Value::as_str) {
        extensions.insert(&ItemId(id.to_owned()))?;
    }
    Ok(Reasoning {
        text: summary,
        signature: None,
        encrypted: item
            .get("encrypted_content")
            .and_then(Value::as_str)
            .map(str::to_owned),
        extensions,
    })
}

fn finish(
    provider: &str,
    response: &Value,
    pricing: Option<&Pricing>,
    refused: bool,
) -> Result<Finish> {
    if let Some(error) = response.get("error").filter(|e| !e.is_null()) {
        return Err(body_error(provider, error));
    }
    let has_calls = response
        .get("output")
        .and_then(Value::as_array)
        .is_some_and(|items| {
            items
                .iter()
                .any(|i| i.get("type").and_then(Value::as_str) == Some("function_call"))
        });
    let reason = match response.get("status").and_then(Value::as_str) {
        Some("incomplete") => match response
            .pointer("/incomplete_details/reason")
            .and_then(Value::as_str)
        {
            Some("content_filter") => FinishReason::ContentFilter,
            _ => FinishReason::Length,
        },
        _ if refused => FinishReason::Refusal,
        _ if has_calls => FinishReason::ToolCalls,
        _ => FinishReason::Stop,
    };
    let usage = usage(response);
    Ok(Finish {
        usage,
        finish_reason: reason,
        id: response
            .get("id")
            .and_then(Value::as_str)
            .map(str::to_owned),
        model: response
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_owned),
        cost: pricing.map(|p| usage.cost(p)),
        raw: response.clone(),
    })
}

/// Write a unary Responses reply.
pub(crate) fn write_reply(
    provider: &str,
    response: &Value,
    pricing: Option<&Pricing>,
    out: &mut StreamWriter,
) -> Result<Finish> {
    let mut refused = false;
    for item in response
        .get("output")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        match item.get("type").and_then(Value::as_str) {
            Some("message") => {
                for content in item
                    .get("content")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    let text = match content.get("type").and_then(Value::as_str) {
                        Some("output_text") => content.get("text").and_then(Value::as_str),
                        Some("refusal") => {
                            refused = true;
                            content.get("refusal").and_then(Value::as_str)
                        }
                        _ => None,
                    };
                    if let Some(text) = text {
                        out.part(AssistantContent::Text(Text {
                            text: text.to_owned(),
                            citations: citations(content),
                            ..Text::default()
                        }));
                    }
                }
            }
            Some("function_call") => {
                let block = out.tool_call(
                    item.get("call_id")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                    item.get("name").and_then(Value::as_str).unwrap_or_default(),
                );
                out.push(
                    &block,
                    item.get("arguments")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                );
                out.end_tool_call(block)?;
            }
            Some("reasoning") => out.part(AssistantContent::Reasoning(reasoning_part(item)?)),
            _ => {}
        }
    }
    finish(provider, response, pricing, refused)
}

/// Decodes a Responses event stream. The `response.completed` event carries
/// the whole reply document, which becomes `raw`.
pub(crate) struct ResponsesDecoder {
    provider: &'static str,
    pricing: Option<Pricing>,
    texts: BTreeMap<(u64, u64), TextBlock>,
    calls: BTreeMap<u64, ToolCallBlock>,
    reasoning: BTreeMap<u64, ReasoningBlock>,
    refused: bool,
}

impl ResponsesDecoder {
    pub(crate) fn new(provider: &'static str, pricing: Option<Pricing>) -> Self {
        Self {
            provider,
            pricing,
            texts: BTreeMap::new(),
            calls: BTreeMap::new(),
            reasoning: BTreeMap::new(),
            refused: false,
        }
    }

    fn close_item(&mut self, index: u64, item: &Value, out: &mut StreamWriter) -> Result<()> {
        let keys: Vec<(u64, u64)> = self
            .texts
            .range((index, 0)..=(index, u64::MAX))
            .map(|(k, _)| *k)
            .collect();
        for key in keys {
            if let Some(block) = self.texts.remove(&key) {
                let content = item
                    .pointer(&format!("/content/{}", key.1))
                    .unwrap_or(&Value::Null);
                for citation in citations(content) {
                    out.cite(&block, citation);
                }
                out.end_text(block);
            }
        }
        if let Some(block) = self.calls.remove(&index) {
            out.end_tool_call(block)?;
        }
        if let Some(block) = self.reasoning.remove(&index) {
            let part = reasoning_part(item)?;
            if let Some(encrypted) = part.encrypted {
                out.encrypt(&block, encrypted);
            }
            if let Some(extensions) = out.extensions(&block) {
                *extensions = part.extensions;
            }
            out.end_reasoning(block);
        }
        Ok(())
    }
}

impl StreamDecoder<SseEvent> for ResponsesDecoder {
    fn feed(&mut self, frame: SseEvent, out: &mut StreamWriter) -> Result<Option<Finish>> {
        if frame.data.trim().is_empty() {
            return Ok(None);
        }
        let event: Value = serde_json::from_str(&frame.data)?;
        let index = event
            .get("output_index")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let delta = event
            .get("delta")
            .and_then(Value::as_str)
            .unwrap_or_default();
        match event
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
        {
            "response.output_item.added" => {
                let item = event.get("item").unwrap_or(&Value::Null);
                match item.get("type").and_then(Value::as_str) {
                    Some("function_call") => {
                        let block = out.tool_call(
                            item.get("call_id")
                                .and_then(Value::as_str)
                                .unwrap_or_default(),
                            item.get("name").and_then(Value::as_str).unwrap_or_default(),
                        );
                        self.calls.insert(index, block);
                    }
                    Some("reasoning") => {
                        self.reasoning.insert(index, out.reasoning());
                    }
                    _ => {}
                }
            }
            "response.output_text.delta" | "response.refusal.delta" => {
                if event.get("type").and_then(Value::as_str) == Some("response.refusal.delta") {
                    self.refused = true;
                }
                let content = event
                    .get("content_index")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                let block = self
                    .texts
                    .entry((index, content))
                    .or_insert_with(|| out.text());
                out.push(block, delta);
            }
            "response.function_call_arguments.delta" => {
                if let Some(block) = self.calls.get(&index) {
                    out.push(block, delta);
                }
            }
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                let block = self
                    .reasoning
                    .entry(index)
                    .or_insert_with(|| out.reasoning());
                out.push(block, delta);
            }
            "response.reasoning_summary_part.done" => {
                if let Some(block) = self.reasoning.get(&index) {
                    out.push(block, "\n\n");
                }
            }
            "response.output_item.done" => {
                let item = event.get("item").cloned().unwrap_or(Value::Null);
                self.close_item(index, &item, out)?;
            }
            "response.completed" | "response.incomplete" => {
                let response = event.get("response").unwrap_or(&Value::Null);
                return finish(self.provider, response, self.pricing.as_ref(), self.refused)
                    .map(Some);
            }
            "response.failed" => {
                let error = event
                    .pointer("/response/error")
                    .cloned()
                    .unwrap_or(Value::Null);
                return Err(body_error(self.provider, &error));
            }
            "error" => return Err(body_error(self.provider, &event)),
            _ => {}
        }
        Ok(None)
    }

    fn end(&mut self, _out: &mut StreamWriter) -> Result<Finish> {
        Err(Error::new(
            ErrorKind::Decode,
            "the stream ended before `response.completed`",
        )
        .with_provider(self.provider))
    }
}

/// A model on the Responses API.
#[derive(Debug, Clone)]
pub struct ResponsesModel {
    client: OpenAI,
    info: ModelInfo,
    card: ModelCard,
}

impl ResponsesModel {
    pub(crate) fn new(client: OpenAI, model: &str) -> Self {
        let (info, card) = (client.info(model), client.card(model));
        Self { client, info, card }
    }
}

impl Model<Completion> for ResponsesModel {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn capabilities(&self) -> ModelCard {
        self.card.clone()
    }

    fn invoke(&self, request: CompletionRequest) -> BoxFuture<'static, Result<CompletionResponse>> {
        let http = std::sync::Arc::clone(self.client.http());
        let (provider, pricing) = (self.client.provider(), self.card.pricing);
        let request = request_body(&self.info.model, &request, false)
            .and_then(|b| self.client.post_json("/responses", &b));
        Box::pin(async move {
            let response = send(&*http, provider, request?).await?;
            let doc = read_json(response, provider).await?;
            respond(|out| write_reply(provider, &doc, pricing.as_ref(), out))
        })
    }
}

impl StreamingModel<Completion> for ResponsesModel {
    fn invoke_stream(
        &self,
        request: CompletionRequest,
    ) -> BoxFuture<'static, Result<BoxStream<'static, Result<StreamEvent>>>> {
        let http = std::sync::Arc::clone(self.client.http());
        let (provider, pricing) = (self.client.provider(), self.card.pricing);
        let request = request_body(&self.info.model, &request, true)
            .and_then(|b| self.client.post_json("/responses", &b));
        Box::pin(async move {
            let response = send(&*http, provider, request?).await?;
            Ok(decode_stream(
                sse(response.into_body()),
                ResponsesDecoder::new(provider, pricing),
            ))
        })
    }
}

#[cfg(test)]
mod tests;
