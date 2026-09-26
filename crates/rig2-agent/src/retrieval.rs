//! Retrieval: search a vector store from an agent.
//!
//! [`Retriever`] embeds a query and searches a store. It works two ways: as a
//! [`Tool`] the model calls when it wants to look something up, and as a
//! [`Hook`] that injects the top matches for the latest user message before
//! every model call (dynamic context).

use std::sync::Arc;

use rig2_core::completion::{CompletionRequest, ToolDefinition};
use rig2_core::content::{Message, ToolOutput};
use rig2_core::store::{Filter, Hit, Query, VectorStore};
use rig2_core::tasks::{Embedding, EmbeddingRequest, InputType};
use rig2_core::{BoxFuture, Error, ErrorKind, Model, Result};
use serde::Deserialize;

use crate::agent::{Control, Hook};
use crate::tool::{Tool, ToolContext, parse_arguments, schema_of};

/// Searches a vector store with an embedding model.
#[derive(Clone)]
pub struct Retriever {
    embedder: Arc<dyn Model<Embedding>>,
    store: Arc<dyn VectorStore>,
    top_k: u32,
    filter: Option<Filter>,
    name: String,
    description: String,
}

impl std::fmt::Debug for Retriever {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Retriever")
            .field("name", &self.name)
            .field("top_k", &self.top_k)
            .finish_non_exhaustive()
    }
}

impl Retriever {
    /// Search `store` for the 3 best matches, embedding queries with
    /// `embedder`. As a tool it is called `search`.
    pub fn new(embedder: Arc<dyn Model<Embedding>>, store: Arc<dyn VectorStore>) -> Self {
        Self {
            embedder,
            store,
            top_k: 3,
            filter: None,
            name: "search".into(),
            description: "Search the knowledge base. Returns the most relevant documents.".into(),
        }
    }

    /// Return this many matches.
    pub fn with_top_k(mut self, top_k: u32) -> Self {
        self.top_k = top_k.max(1);
        self
    }

    /// Only search records matching `filter`.
    pub fn with_filter(mut self, filter: Filter) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Name and describe the tool.
    pub fn with_tool_name(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        self.name = name.into();
        self.description = description.into();
        self
    }

    /// The best matches for `query`.
    pub fn search(&self, query: &str) -> BoxFuture<'static, Result<Vec<Hit>>> {
        let (embedder, store) = (Arc::clone(&self.embedder), Arc::clone(&self.store));
        let (top_k, filter) = (self.top_k, self.filter.clone());
        let request = EmbeddingRequest::new([query]).with_input_type(InputType::Query);
        Box::pin(async move {
            let vector = embedder
                .invoke(request)
                .await?
                .embeddings
                .into_iter()
                .next()
                .ok_or_else(|| Error::new(ErrorKind::Decode, "the embedder returned no vector"))?;
            let mut query = Query::new(vector, top_k);
            query.filter = filter;
            store.search(query).await
        })
    }
}

fn render(hits: &[Hit]) -> String {
    hits.iter()
        .map(|h| {
            format!(
                "[{}] (score {:.3}) {}",
                h.id,
                h.score,
                serde_json::Value::Object(h.document.clone())
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[derive(Deserialize, schemars::JsonSchema)]
struct SearchArguments {
    /// What to look for.
    query: String,
}

impl Tool for Retriever {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name.clone(),
            description: self.description.clone(),
            parameters: schema_of::<SearchArguments>(),
        }
    }

    fn call(
        &self,
        arguments: serde_json::Value,
        _context: ToolContext,
    ) -> BoxFuture<'static, Result<Vec<ToolOutput>>> {
        let search = parse_arguments::<SearchArguments>(&self.name, arguments)
            .map(|a| self.search(&a.query));
        Box::pin(async move { Ok(vec![ToolOutput::Text(render(&search?.await?))]) })
    }
}

impl Hook for Retriever {
    /// Add the best matches for the latest user text to the system prompt.
    fn before_model(
        &self,
        mut request: CompletionRequest,
    ) -> BoxFuture<'static, Result<Control<CompletionRequest>>> {
        let latest = request.messages.iter().rev().find_map(|m| match m {
            Message::User { .. } => Some(m.text()),
            Message::Assistant { .. } => None,
        });
        let Some(query) = latest.filter(|q| !q.trim().is_empty()) else {
            return Box::pin(std::future::ready(Ok(Control::Continue(request))));
        };
        let search = self.search(&query);
        Box::pin(async move {
            let hits = search.await?;
            if !hits.is_empty() {
                let context = format!("Relevant documents:\n{}", render(&hits));
                request.system = Some(match request.system.take() {
                    Some(system) => format!("{system}\n\n{context}"),
                    None => context,
                });
            }
            Ok(Control::Continue(request))
        })
    }
}
