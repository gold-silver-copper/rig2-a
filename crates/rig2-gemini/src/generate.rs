//! GenerateContent: request mapping, the unary reply and the streaming
//! decoder.

use std::sync::Arc;

use rig2_core::catalog::{ModelCard, Pricing};
use rig2_core::completion::{
    Completion, CompletionRequest, CompletionResponse, Finish, FinishReason, ReasoningBlock,
    ReasoningEffort, StreamDecoder, StreamEvent, StreamWriter, TextBlock, ToolChoice, Usage,
    decode_stream, respond,
};
use rig2_core::content::{
    AssistantContent, Extension, Extensions, Image, Message, Reasoning, Source, Text, ToolCall,
    ToolOutput, UserContent,
};
use rig2_core::http::{SseEvent, base64, parse_data_url, read_json, send, set, sse};
use rig2_core::{BoxFuture, BoxStream, Error, ErrorKind, Model, ModelInfo, Result, StreamingModel};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::Gemini;

/// A Gemini thought signature, returned with a part and sent back with it
/// on the next turn so the model can resume its reasoning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThoughtSignature(pub String);

impl Extension for ThoughtSignature {
    const KEY: &'static str = "gemini.thought_signature";
}

fn unsupported(what: &str) -> Error {
    Error::new(
        ErrorKind::Unsupported,
        format!("{what} is not supported by Gemini"),
    )
}

fn media(source: &Source, media_type: &str) -> Value {
    match source {
        Source::Bytes(bytes) => {
            json!({ "inlineData": { "mimeType": media_type, "data": base64(bytes) } })
        }
        Source::Url(url) => json!({ "fileData": { "mimeType": media_type, "fileUri": url } }),
    }
}

fn image(image: &Image) -> Value {
    media(
        &image.source,
        image.media_type.as_deref().unwrap_or("image/png"),
    )
}

fn with_signature(mut part: Value, extensions: &Extensions) -> Result<Value> {
    if let Some(ThoughtSignature(signature)) = extensions.get::<ThoughtSignature>()? {
        set(&mut part, "thoughtSignature", signature);
    }
    Ok(part)
}

fn user_parts(content: &[UserContent]) -> Result<Vec<Value>> {
    content
        .iter()
        .map(|part| match part {
            UserContent::Text(text) => Ok(json!({ "text": text.text })),
            UserContent::Image(i) => Ok(image(i)),
            UserContent::Audio(audio) => Ok(media(&audio.source, &audio.media_type)),
            UserContent::File(file) => Ok(media(&file.source, &file.media_type)),
            UserContent::ToolResult(result) => {
                let output: Vec<Value> = result
                    .output
                    .iter()
                    .map(|o| match o {
                        ToolOutput::Text(text) => Ok(Value::String(text.clone())),
                        ToolOutput::Json(value) => Ok(value.clone()),
                        ToolOutput::Image(_) => Err(unsupported("an image in a tool result")),
                    })
                    .collect::<Result<_>>()?;
                let output = match <[Value; 1]>::try_from(output) {
                    Ok([single]) => single,
                    Err(many) => Value::Array(many),
                };
                let key = if result.is_error { "error" } else { "output" };
                Ok(json!({ "functionResponse": { "name": result.name, "response": { key: output } } }))
            }
        })
        .collect()
}

fn model_parts(content: &[AssistantContent]) -> Result<Vec<Value>> {
    let mut parts = Vec::new();
    for part in content {
        match part {
            AssistantContent::Text(text) => parts.push(with_signature(
                json!({ "text": text.text }),
                &text.extensions,
            )?),
            AssistantContent::ToolCall(call) => parts.push(with_signature(
                json!({ "functionCall": { "name": call.name, "args": call.arguments } }),
                &call.extensions,
            )?),
            AssistantContent::Image(i) => parts.push(image(i)),
            AssistantContent::Reasoning(_) => {}
        }
    }
    Ok(parts)
}

fn thinking_budget(effort: ReasoningEffort) -> u32 {
    match effort {
        ReasoningEffort::Minimal => 512,
        other => other.budget_tokens(),
    }
}

/// The GenerateContent request body for `request`.
pub(crate) fn request_body(request: &CompletionRequest, image_output: bool) -> Result<Value> {
    let mut contents = Vec::new();
    for message in &request.messages {
        let (role, parts) = match message {
            Message::User { content } => ("user", user_parts(content)?),
            Message::Assistant { content } => ("model", model_parts(content)?),
        };
        if !parts.is_empty() {
            contents.push(json!({ "role": role, "parts": parts }));
        }
    }
    let mut body = json!({ "contents": contents });
    if let Some(system) = &request.system {
        set(
            &mut body,
            "systemInstruction",
            json!({ "parts": [{ "text": system }] }),
        );
    }
    if !request.tools.is_empty() {
        let declarations: Vec<Value> = request
            .tools
            .iter()
            .map(|t| json!({ "name": t.name, "description": t.description, "parametersJsonSchema": t.parameters }))
            .collect();
        set(
            &mut body,
            "tools",
            json!([{ "functionDeclarations": declarations }]),
        );
        let config = match &request.tool_choice {
            ToolChoice::Auto => json!({ "mode": "AUTO" }),
            ToolChoice::None => json!({ "mode": "NONE" }),
            ToolChoice::Required => json!({ "mode": "ANY" }),
            ToolChoice::Tool(name) => json!({ "mode": "ANY", "allowedFunctionNames": [name] }),
        };
        set(
            &mut body,
            "toolConfig",
            json!({ "functionCallingConfig": config }),
        );
    }
    let mut generation = json!({});
    if let Some(max) = request.max_tokens {
        set(&mut generation, "maxOutputTokens", max);
    }
    if let Some(t) = request.temperature {
        set(&mut generation, "temperature", t);
    }
    if let Some(p) = request.top_p {
        set(&mut generation, "topP", p);
    }
    if !request.stop.is_empty() {
        set(&mut generation, "stopSequences", request.stop.clone());
    }
    if let Some(output) = &request.output {
        set(&mut generation, "responseMimeType", "application/json");
        set(&mut generation, "responseJsonSchema", output.schema.clone());
    }
    if let Some(effort) = request.reasoning {
        set(
            &mut generation,
            "thinkingConfig",
            json!({ "thinkingBudget": thinking_budget(effort), "includeThoughts": true }),
        );
    }
    if image_output {
        set(
            &mut generation,
            "responseModalities",
            json!(["TEXT", "IMAGE"]),
        );
    }
    if generation.as_object().is_some_and(|g| !g.is_empty()) {
        set(&mut body, "generationConfig", generation);
    }
    Ok(body)
}

fn u64_at(value: &Value, key: &str) -> u64 {
    value.get(key).and_then(Value::as_u64).unwrap_or(0)
}

pub(crate) fn usage(metadata: &Value) -> Usage {
    let thoughts = u64_at(metadata, "thoughtsTokenCount");
    Usage {
        input_tokens: u64_at(metadata, "promptTokenCount"),
        output_tokens: u64_at(metadata, "candidatesTokenCount") + thoughts,
        cached_input_tokens: u64_at(metadata, "cachedContentTokenCount"),
        cache_write_tokens: 0,
        reasoning_tokens: thoughts,
    }
}

fn finish_reason(reason: Option<&str>, has_calls: bool) -> FinishReason {
    match reason {
        Some("MAX_TOKENS") => FinishReason::Length,
        Some(
            "SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" | "IMAGE_SAFETY",
        ) => FinishReason::ContentFilter,
        Some("STOP") | None if has_calls => FinishReason::ToolCalls,
        Some("STOP" | "FINISH_REASON_UNSPECIFIED") | None => FinishReason::Stop,
        Some(other) => FinishReason::Other(other.to_owned()),
    }
}

fn signature_extensions(part: &Value) -> Result<Extensions> {
    let mut extensions = Extensions::default();
    if let Some(signature) = part.get("thoughtSignature").and_then(Value::as_str) {
        extensions.insert(&ThoughtSignature(signature.to_owned()))?;
    }
    Ok(extensions)
}

fn inline_image(part: &Value) -> Result<Option<Image>> {
    let Some(data) = part.get("inlineData") else {
        return Ok(None);
    };
    let media_type = data
        .get("mimeType")
        .and_then(Value::as_str)
        .unwrap_or("image/png");
    let encoded = data.get("data").and_then(Value::as_str).unwrap_or_default();
    let (_, bytes) = parse_data_url(&format!("data:{media_type};base64,{encoded}"))?;
    Ok(Some(Image::from_bytes(bytes, media_type)))
}

/// Parts of one candidate, written as content. `calls` counts tool calls
/// so far, to name those Gemini leaves without an id.
fn write_parts(parts: &[Value], calls: &mut usize, out: &mut StreamWriter) -> Result<()> {
    for part in parts {
        let extensions = signature_extensions(part)?;
        if let Some(call) = part.get("functionCall") {
            *calls += 1;
            let id = call
                .get("id")
                .and_then(Value::as_str)
                .map_or_else(|| format!("call_{calls}"), str::to_owned);
            let name = call.get("name").and_then(Value::as_str).unwrap_or_default();
            let mut tool_call = ToolCall::new(
                id,
                name,
                call.get("args").cloned().unwrap_or_else(|| json!({})),
            );
            tool_call.extensions = extensions;
            out.part(AssistantContent::ToolCall(tool_call));
        } else if let Some(image) = inline_image(part)? {
            out.part(AssistantContent::Image(image));
        } else if let Some(text) = part.get("text").and_then(Value::as_str) {
            if part.get("thought").and_then(Value::as_bool) == Some(true) {
                let signature = part
                    .get("thoughtSignature")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                out.part(AssistantContent::Reasoning(Reasoning {
                    text: text.to_owned(),
                    signature,
                    ..Reasoning::default()
                }));
            } else if !text.is_empty() || !extensions.is_empty() {
                out.part(AssistantContent::Text(Text {
                    text: text.to_owned(),
                    extensions,
                    ..Text::default()
                }));
            }
        }
    }
    Ok(())
}

fn finish(doc: &Value, has_calls: bool, pricing: Option<&Pricing>) -> Finish {
    let usage = doc.get("usageMetadata").map(usage).unwrap_or_default();
    Finish {
        usage,
        finish_reason: finish_reason(
            doc.pointer("/candidates/0/finishReason")
                .and_then(Value::as_str),
            has_calls,
        ),
        id: doc
            .get("responseId")
            .and_then(Value::as_str)
            .map(str::to_owned),
        model: doc
            .get("modelVersion")
            .and_then(Value::as_str)
            .map(str::to_owned),
        cost: pricing.map(|p| usage.cost(p)),
        raw: doc.clone(),
    }
}

/// Write a unary GenerateContent reply.
pub(crate) fn write_reply(
    doc: &Value,
    pricing: Option<&Pricing>,
    out: &mut StreamWriter,
) -> Result<Finish> {
    let parts = doc
        .pointer("/candidates/0/content/parts")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut calls = 0;
    write_parts(&parts, &mut calls, out)?;
    let mut finish = finish(doc, calls > 0, pricing);
    if doc.pointer("/promptFeedback/blockReason").is_some() {
        finish.finish_reason = FinishReason::ContentFilter;
    }
    Ok(finish)
}

/// Decodes a GenerateContent stream. Each event is a partial response;
/// the decoder merges their parts into one document for `raw`.
pub(crate) struct GenerateDecoder {
    pricing: Option<Pricing>,
    text: Option<TextBlock>,
    thought: Option<ReasoningBlock>,
    calls: usize,
    parts: Vec<Value>,
    last: Value,
    reason: Option<String>,
}

impl GenerateDecoder {
    pub(crate) fn new(pricing: Option<Pricing>) -> Self {
        Self {
            pricing,
            text: None,
            thought: None,
            calls: 0,
            parts: Vec::new(),
            last: Value::Null,
            reason: None,
        }
    }

    fn close(&mut self, out: &mut StreamWriter) {
        if let Some(block) = self.thought.take() {
            out.end_reasoning(block);
        }
        if let Some(block) = self.text.take() {
            out.end_text(block);
        }
    }

    /// Merge `part` into the document: consecutive text of the same kind
    /// joins, everything else is appended.
    fn merge(&mut self, part: &Value) {
        let is_thought = part.get("thought").and_then(Value::as_bool) == Some(true);
        let text = part.get("text").and_then(Value::as_str);
        if let (Some(text), Some(last)) = (text, self.parts.last_mut()) {
            let last_is_thought = last.get("thought").and_then(Value::as_bool) == Some(true);
            if let Some(previous) = last
                .get("text")
                .and_then(Value::as_str)
                .filter(|_| last_is_thought == is_thought)
            {
                let joined = format!("{previous}{text}");
                set(last, "text", joined);
                if let Some(signature) = part.get("thoughtSignature") {
                    set(last, "thoughtSignature", signature.clone());
                }
                return;
            }
        }
        self.parts.push(part.clone());
    }

    fn finish(&mut self, out: &mut StreamWriter) -> Finish {
        self.close(out);
        let mut doc = self.last.clone();
        let candidate = json!({
            "content": { "role": "model", "parts": std::mem::take(&mut self.parts) },
            "finishReason": self.reason,
        });
        set(&mut doc, "candidates", json!([candidate]));
        finish(&doc, self.calls > 0, self.pricing.as_ref())
    }
}

impl StreamDecoder<SseEvent> for GenerateDecoder {
    fn feed(&mut self, frame: SseEvent, out: &mut StreamWriter) -> Result<Option<Finish>> {
        if frame.data.trim().is_empty() {
            return Ok(None);
        }
        let chunk: Value = serde_json::from_str(&frame.data)?;
        if let Some(error) = chunk.get("error") {
            let message = error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("Gemini reported an error");
            return Err(Error::new(ErrorKind::Other, message).with_provider(crate::PROVIDER));
        }
        for part in chunk
            .pointer("/candidates/0/content/parts")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            self.merge(part);
            let text = part.get("text").and_then(Value::as_str);
            let signature = part.get("thoughtSignature").and_then(Value::as_str);
            if part.get("thought").and_then(Value::as_bool) == Some(true) {
                if let Some(block) = self.text.take() {
                    out.end_text(block);
                }
                let block = self.thought.get_or_insert_with(|| out.reasoning());
                out.push(block, text.unwrap_or_default());
                if let Some(signature) = signature {
                    out.sign(block, signature);
                }
            } else if part.get("functionCall").is_some() || part.get("inlineData").is_some() {
                self.close(out);
                write_parts(std::slice::from_ref(part), &mut self.calls, out)?;
            } else if let Some(text) = text {
                if let Some(block) = self.thought.take() {
                    out.end_reasoning(block);
                }
                let block = self.text.get_or_insert_with(|| out.text());
                out.push(block, text);
                if let (Some(signature), Some(extensions)) = (signature, out.extensions(block)) {
                    extensions.insert(&ThoughtSignature(signature.to_owned()))?;
                }
            }
        }
        if let Some(reason) = chunk
            .pointer("/candidates/0/finishReason")
            .and_then(Value::as_str)
        {
            self.reason = Some(reason.to_owned());
        }
        self.last = chunk;
        Ok(None)
    }

    fn end(&mut self, out: &mut StreamWriter) -> Result<Finish> {
        if self.reason.is_some() {
            Ok(self.finish(out))
        } else {
            Err(Error::new(
                ErrorKind::Decode,
                "the stream ended without a finish reason",
            )
            .with_provider(crate::PROVIDER))
        }
    }
}

/// A model on GenerateContent.
#[derive(Debug, Clone)]
pub struct GenerateModel {
    client: Gemini,
    info: ModelInfo,
    card: ModelCard,
}

impl GenerateModel {
    pub(crate) fn new(client: Gemini, model: &str) -> Self {
        let card = client.card(model);
        Self {
            client,
            info: Gemini::info(model),
            card,
        }
    }
}

impl Model<Completion> for GenerateModel {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn capabilities(&self) -> ModelCard {
        self.card.clone()
    }

    fn invoke(&self, request: CompletionRequest) -> BoxFuture<'static, Result<CompletionResponse>> {
        let (http, pricing) = (Arc::clone(self.client.http()), self.card.pricing);
        let path = format!("/models/{}:generateContent", self.info.model);
        let request = request_body(&request, false).and_then(|b| self.client.post_json(&path, &b));
        Box::pin(async move {
            let doc = read_json(
                send(&*http, crate::PROVIDER, request?).await?,
                crate::PROVIDER,
            )
            .await?;
            respond(|out| write_reply(&doc, pricing.as_ref(), out))
        })
    }
}

impl StreamingModel<Completion> for GenerateModel {
    fn invoke_stream(
        &self,
        request: CompletionRequest,
    ) -> BoxFuture<'static, Result<BoxStream<'static, Result<StreamEvent>>>> {
        let (http, pricing) = (Arc::clone(self.client.http()), self.card.pricing);
        let path = format!("/models/{}:streamGenerateContent?alt=sse", self.info.model);
        let request = request_body(&request, false).and_then(|b| self.client.post_json(&path, &b));
        Box::pin(async move {
            let response = send(&*http, crate::PROVIDER, request?).await?;
            Ok(decode_stream(
                sse(response.into_body()),
                GenerateDecoder::new(pricing),
            ))
        })
    }
}

#[cfg(test)]
mod tests;
