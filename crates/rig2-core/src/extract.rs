//! Structured output: extract a typed value from any completion model.
//!
//! [`Extract<T>`] is a task whose output is `T`. An [`Extractor`] performs it
//! over any completion model: it uses the provider's native schema mode when
//! the catalog says the model has one, and a forced tool call otherwise. When
//! the reply does not parse as `T`, it tells the model what was wrong and
//! asks again, a bounded number of times.
//!
//! ```no_run
//! # async fn demo(model: std::sync::Arc<dyn rig2_core::Model<rig2_core::completion::Completion>>) -> rig2_core::Result<()> {
//! use rig2_core::Model;
//! use rig2_core::extract::Extractor;
//!
//! #[derive(serde::Deserialize, schemars::JsonSchema)]
//! struct City { name: String, country: String }
//!
//! let city = Extractor::<City, _>::new(model).run("The capital of France, as JSON.").await?;
//! # Ok(()) }
//! ```

use std::marker::PhantomData;
use std::sync::Arc;

use schemars::JsonSchema;
use serde::de::DeserializeOwned;

use crate::catalog::{Capability, ModelCard};
use crate::completion::{
    Completion, CompletionRequest, CompletionResponse, OutputSchema, ToolChoice, ToolDefinition,
};
use crate::content::{Message, ToolResult};
use crate::{BoxFuture, Error, ErrorKind, MaybeSend, Model, ModelInfo, Result, Task};

/// Extract a `T` from a conversation.
pub struct Extract<T>(PhantomData<fn() -> T>);

impl<T: JsonSchema + DeserializeOwned + MaybeSend + 'static> Task for Extract<T> {
    const NAME: &'static str = "extract";
    type Input = CompletionRequest;
    type Output = T;
    type Capabilities = ModelCard;
}

/// Performs [`Extract<T>`] over a completion model.
pub struct Extractor<T, M> {
    inner: Arc<M>,
    max_repairs: u32,
    target: PhantomData<fn() -> T>,
}

impl<T, M> std::fmt::Debug for Extractor<T, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Extractor")
            .field("target", &std::any::type_name::<T>())
            .field("max_repairs", &self.max_repairs)
            .finish_non_exhaustive()
    }
}

impl<T, M> Extractor<T, M> {
    /// Extract a `T` with `model`, allowing 2 repair rounds.
    pub fn new(model: M) -> Self {
        Self {
            inner: Arc::new(model),
            max_repairs: 2,
            target: PhantomData,
        }
    }

    /// Allow this many repair rounds after the first attempt.
    pub fn with_max_repairs(mut self, max_repairs: u32) -> Self {
        self.max_repairs = max_repairs;
        self
    }
}

/// How the schema reaches the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Native,
    Tool,
}

/// The schema, wrapped in an object when its root is not one, because
/// providers require object roots.
struct Shape {
    schema: OutputSchema,
    wrapped: bool,
}

impl Shape {
    fn of<T: JsonSchema>() -> Self {
        let mut schema = OutputSchema::of::<T>();
        let is_object = schema.schema.get("type").and_then(|t| t.as_str()) == Some("object");
        if !is_object {
            let mut inner = schema.schema.clone();
            let defs = inner.as_object_mut().and_then(|o| o.remove("$defs"));
            let mut root = serde_json::json!({
                "type": "object",
                "properties": { "value": inner },
                "required": ["value"],
                "additionalProperties": false,
            });
            if let (Some(defs), Some(obj)) = (defs, root.as_object_mut()) {
                obj.insert("$defs".into(), defs);
            }
            schema.schema = root;
        }
        Self {
            schema,
            wrapped: !is_object,
        }
    }

    fn decode<T: DeserializeOwned>(&self, value: serde_json::Value) -> Result<T> {
        let value = if self.wrapped {
            value
                .get("value")
                .cloned()
                .unwrap_or(serde_json::Value::Null)
        } else {
            value
        };
        serde_json::from_value(value).map_err(|e| {
            Error::new(
                ErrorKind::InvalidOutput,
                format!("the reply does not match the schema: {e}"),
            )
        })
    }
}

async fn extract<T: DeserializeOwned, M: Model<Completion>>(
    model: Arc<M>,
    mut request: CompletionRequest,
    shape: Shape,
    mode: Mode,
    max_repairs: u32,
) -> Result<T> {
    let tool_name = format!("submit_{}", shape.schema.name);
    match mode {
        Mode::Native => request.output = Some(shape.schema.clone()),
        Mode::Tool => {
            request.tools.push(ToolDefinition {
                name: tool_name.clone(),
                description: "Submit the answer in the required structure.".into(),
                parameters: shape.schema.schema.clone(),
            });
            request.tool_choice = ToolChoice::Tool(tool_name.clone());
        }
    }
    let mut repairs = 0;
    loop {
        let response: CompletionResponse = model.invoke(request.clone()).await?;
        let attempt = match mode {
            Mode::Native => response
                .parse::<serde_json::Value>()
                .and_then(|value| shape.decode::<T>(value))
                .map_err(|e| (e, None)),
            Mode::Tool => match response
                .tool_calls()
                .into_iter()
                .find(|c| c.name == tool_name)
            {
                Some(call) => shape
                    .decode::<T>(call.arguments.clone())
                    .map_err(|e| (e, Some(call.id.clone()))),
                None => Err((
                    Error::new(
                        ErrorKind::InvalidOutput,
                        format!("the reply did not call `{tool_name}`"),
                    ),
                    None,
                )),
            },
        };
        match attempt {
            Ok(value) => return Ok(value),
            Err((error, _)) if repairs >= max_repairs => return Err(error),
            Err((error, call_id)) => {
                repairs += 1;
                request.messages.push(response.message());
                let complaint = format!(
                    "{} Answer again, following the schema exactly.",
                    error.message()
                );
                request.messages.push(match call_id {
                    Some(id) => {
                        Message::tool_results([ToolResult::error(id, &tool_name, complaint)])
                    }
                    None => Message::user(complaint),
                });
            }
        }
    }
}

impl<T, M> Model<Extract<T>> for Extractor<T, M>
where
    T: JsonSchema + DeserializeOwned + MaybeSend + 'static,
    M: Model<Completion> + 'static,
{
    fn info(&self) -> &ModelInfo {
        self.inner.info()
    }

    fn capabilities(&self) -> ModelCard {
        self.inner.capabilities()
    }

    fn invoke(&self, request: CompletionRequest) -> BoxFuture<'static, Result<T>> {
        let mode = if self
            .inner
            .capabilities()
            .supports(Capability::StructuredOutput)
        {
            Mode::Native
        } else {
            Mode::Tool
        };
        Box::pin(extract(
            Arc::clone(&self.inner),
            request,
            Shape::of::<T>(),
            mode,
            self.max_repairs,
        ))
    }
}

#[cfg(test)]
mod tests;
