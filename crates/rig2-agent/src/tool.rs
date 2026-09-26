//! Tools: what an agent can call.
//!
//! A [`Tool`] has a definition (name, description, argument schema) and a
//! call. Write one by hand, generate one with `#[tool]`, or adapt any
//! [`Model<T>`](rig2_core::Model) with [`ModelTool`], which is how vision and
//! other non-chat models enter an agent.
//!
//! ```
//! use rig2_agent::tool::{Tool, ToolSet};
//!
//! /// Add two numbers.
//! #[rig2_agent::tool]
//! fn add(#[describe("the first addend")] a: f64, b: f64) -> f64 {
//!     a + b
//! }
//!
//! let mut tools = ToolSet::new();
//! tools.insert(Add);
//! assert_eq!(tools.definitions()[0].name, "add");
//! assert_eq!(tools.definitions()[0].description, "Add two numbers.");
//! ```

use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::Arc;

use rig2_core::completion::ToolDefinition;
use rig2_core::content::{Image, Message, ToolOutput, UserContent};
use rig2_core::{BoxFuture, Error, ErrorKind, MaybeSend, MaybeSync, Model, Result, Task};
use schemars::JsonSchema;
use serde::Serialize;
use serde::de::DeserializeOwned;

/// Something an agent can call.
pub trait Tool: MaybeSend + MaybeSync + 'static {
    /// The name, description and argument schema the model sees.
    fn definition(&self) -> ToolDefinition;

    /// Run the tool with the model's arguments.
    ///
    /// An error is reported to the model as a failed tool call, not as a
    /// failure of the run.
    fn call(
        &self,
        arguments: serde_json::Value,
        context: ToolContext,
    ) -> BoxFuture<'static, Result<Vec<ToolOutput>>>;

    /// Whether a call needs approval before it runs.
    fn requires_approval(&self) -> bool {
        false
    }
}

impl<T: Tool + ?Sized> Tool for Arc<T> {
    fn definition(&self) -> ToolDefinition {
        (**self).definition()
    }

    fn call(
        &self,
        arguments: serde_json::Value,
        context: ToolContext,
    ) -> BoxFuture<'static, Result<Vec<ToolOutput>>> {
        (**self).call(arguments, context)
    }

    fn requires_approval(&self) -> bool {
        (**self).requires_approval()
    }
}

/// What a tool can see of the run that called it.
#[derive(Debug, Clone, Default)]
pub struct ToolContext {
    messages: Arc<[Message]>,
}

impl ToolContext {
    /// A context over `messages`.
    pub fn new(messages: Arc<[Message]>) -> Self {
        Self { messages }
    }

    /// The conversation so far.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// The images the user sent, oldest first.
    pub fn images(&self) -> Vec<&Image> {
        self.messages
            .iter()
            .filter_map(|m| match m {
                Message::User { content } => Some(content),
                Message::Assistant { .. } => None,
            })
            .flatten()
            .filter_map(|c| match c {
                UserContent::Image(image) => Some(image),
                _ => None,
            })
            .collect()
    }
}

/// The tools an agent can call, by name.
#[derive(Clone, Default)]
pub struct ToolSet {
    tools: BTreeMap<String, Arc<dyn Tool>>,
}

impl std::fmt::Debug for ToolSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(self.tools.keys()).finish()
    }
}

impl ToolSet {
    /// An empty set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a tool, replacing any with the same name.
    pub fn insert(&mut self, tool: impl Tool) {
        let tool: Arc<dyn Tool> = Arc::new(tool);
        self.tools.insert(tool.definition().name, tool);
    }

    /// The tool called `name`.
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.get(name)
    }

    /// Every definition, sorted by name.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(Tool::definition).collect()
    }

    /// The names of tools that need approval.
    pub fn needing_approval(&self) -> Vec<String> {
        self.tools
            .iter()
            .filter(|(_, t)| t.requires_approval())
            .map(|(n, _)| n.clone())
            .collect()
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

/// Makes a tool out of any model.
///
/// The model sees arguments of type `A`; `map` turns them, with the calling
/// run's context, into the model's input. The output is serialized as JSON.
///
/// ```
/// # use std::sync::Arc;
/// use rig2_agent::tool::{ModelTool, ToolContext};
/// use rig2_core::{Error, ErrorKind};
/// use rig2_core::vision::{EncodedImage, SemanticSegmentation};
/// # fn demo(segmenter: Arc<dyn rig2_core::Model<SemanticSegmentation>>) {
///
/// #[derive(serde::Deserialize, schemars::JsonSchema)]
/// struct Args { image: usize }
///
/// let tool = ModelTool::new("segment", "Segment the n-th image the user sent.", segmenter, |args: Args, ctx: &ToolContext| {
///     let image = ctx.images().get(args.image).copied().cloned().ok_or_else(|| Error::new(ErrorKind::InvalidRequest, "no such image"))?;
///     match image.source {
///         rig2_core::content::Source::Bytes(bytes) => Ok(EncodedImage { media_type: image.media_type.unwrap_or_default(), bytes }),
///         rig2_core::content::Source::Url(_) => Err(Error::new(ErrorKind::Unsupported, "images by URL")),
///     }
/// });
/// # }
/// ```
pub struct ModelTool<T: Task, A, F> {
    definition: ToolDefinition,
    model: Arc<dyn Model<T>>,
    map: Arc<F>,
    arguments: PhantomData<fn(A)>,
}

impl<T: Task, A: JsonSchema, F: Fn(A, &ToolContext) -> Result<T::Input>> ModelTool<T, A, F> {
    /// A tool named `name` that calls `model`.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        model: Arc<dyn Model<T>>,
        map: F,
    ) -> Self {
        Self {
            definition: ToolDefinition {
                name: name.into(),
                description: description.into(),
                parameters: schema_of::<A>(),
            },
            model,
            map: Arc::new(map),
            arguments: PhantomData,
        }
    }
}

impl<T, A, F> Tool for ModelTool<T, A, F>
where
    T: Task,
    T::Output: Serialize,
    A: DeserializeOwned + JsonSchema + 'static,
    F: Fn(A, &ToolContext) -> Result<T::Input> + MaybeSend + MaybeSync + 'static,
{
    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    fn call(
        &self,
        arguments: serde_json::Value,
        context: ToolContext,
    ) -> BoxFuture<'static, Result<Vec<ToolOutput>>> {
        let input = parse_arguments::<A>(&self.definition.name, arguments)
            .and_then(|args| (self.map)(args, &context));
        let call = input.map(|input| self.model.invoke(input));
        Box::pin(async move { to_output(&call?.await?) })
    }
}

/// The JSON schema of `A`, as tools send it.
pub fn schema_of<A: JsonSchema>() -> serde_json::Value {
    let mut schema = schemars::schema_for!(A).to_value();
    if let Some(obj) = schema.as_object_mut() {
        obj.remove("$schema");
        obj.remove("title");
    }
    schema
}

/// Parse a tool's arguments, reporting a mismatch as
/// [`ErrorKind::InvalidRequest`] so the model can correct itself.
pub fn parse_arguments<A: DeserializeOwned>(tool: &str, arguments: serde_json::Value) -> Result<A> {
    serde_json::from_value(arguments).map_err(|e| {
        Error::new(
            ErrorKind::InvalidRequest,
            format!("invalid arguments for `{tool}`: {e}"),
        )
    })
}

/// A tool's output: text when it serializes to a string, JSON otherwise.
pub fn to_output<O: Serialize + ?Sized>(output: &O) -> Result<Vec<ToolOutput>> {
    Ok(vec![match serde_json::to_value(output)? {
        serde_json::Value::String(text) => ToolOutput::Text(text),
        value => ToolOutput::Json(value),
    }])
}

#[cfg(test)]
mod tests;
