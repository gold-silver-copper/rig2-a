//! Cohere for rig2: chat (API v2), embeddings and rerank.
//!
//! ```no_run
//! # async fn demo() -> rig2_core::Result<()> {
//! use rig2_cohere::Cohere;
//! use rig2_core::Model;
//! use rig2_core::tasks::RerankRequest;
//!
//! let cohere = Cohere::from_env()?;
//! let ranked = cohere.rerank("rerank-v3.5").run(RerankRequest::new("capital of France", ["Paris", "Rome"])).await?;
//! println!("{:?}", ranked.results);
//! # Ok(()) }
//! ```

mod chat;
mod media;

use std::sync::Arc;

use bytes::Bytes;
use rig2_core::catalog::{Catalog, ModelCard};
use rig2_core::completion::Completion;
use rig2_core::http::{Body, HttpClient, SharedClient};
use rig2_core::spec::{ModelSpec, Registry};
use rig2_core::tasks::{Embedding, Rerank};
use rig2_core::{Model, ModelInfo, Result, Secret, StreamingModel};

pub use chat::ChatModel;
pub use media::{EmbeddingModel, RerankModel};

const CATALOG: &str = include_str!("../catalogs/cohere.json");
const PROVIDER: &str = "cohere";

/// The configuration of the Cohere API: key and base URL.
#[derive(Debug, Clone)]
pub struct CohereConfig {
    key: Secret,
    base_url: String,
}

impl CohereConfig {
    /// Cohere with `key`.
    pub fn new(key: impl Into<Secret>) -> Self {
        Self {
            key: key.into(),
            base_url: "https://api.cohere.com/v2".into(),
        }
    }

    /// Cohere, with the key from `COHERE_API_KEY`.
    pub fn from_env() -> Result<Self> {
        Ok(Self::new(Secret::from_env("COHERE_API_KEY")?))
    }

    /// Use another base URL, such as a proxy.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        base_url
            .into()
            .trim_end_matches('/')
            .clone_into(&mut self.base_url);
        self
    }

    /// The client over `http`.
    pub fn connect(self, http: impl HttpClient + 'static) -> Cohere {
        let catalog = Catalog::parse(CATALOG).unwrap_or_default();
        Cohere {
            inner: Arc::new(Client {
                config: self,
                http: Arc::new(http),
                catalog,
            }),
        }
    }
}

#[derive(Debug)]
struct Client {
    config: CohereConfig,
    http: SharedClient,
    catalog: Catalog,
}

/// A client for Cohere.
#[derive(Debug, Clone)]
pub struct Cohere {
    inner: Arc<Client>,
}

impl Cohere {
    /// Cohere with `key`, on the shared reqwest client.
    #[cfg(feature = "reqwest")]
    pub fn new(key: impl Into<Secret>) -> Self {
        CohereConfig::new(key).connect(rig2_reqwest::ReqwestClient::default())
    }

    /// Cohere with the key from `COHERE_API_KEY`, on the shared reqwest
    /// client.
    #[cfg(feature = "reqwest")]
    pub fn from_env() -> Result<Self> {
        Ok(CohereConfig::from_env()?.connect(rig2_reqwest::ReqwestClient::default()))
    }

    /// The same configuration over another HTTP client.
    pub fn with_http(&self, http: impl HttpClient + 'static) -> Self {
        self.inner.config.clone().connect(http)
    }

    /// The catalog of Cohere's models.
    pub fn catalog(&self) -> &Catalog {
        &self.inner.catalog
    }

    fn info(model: &str) -> ModelInfo {
        ModelInfo::new(PROVIDER, model)
    }

    fn card(&self, model: &str) -> ModelCard {
        self.inner.catalog.card(model)
    }

    fn http(&self) -> &SharedClient {
        &self.inner.http
    }

    fn post_json(&self, path: &str, body: &serde_json::Value) -> Result<http::Request<Body>> {
        http::Request::post(format!("{}{path}", self.inner.config.base_url))
            .header(
                http::header::AUTHORIZATION,
                format!("Bearer {}", self.inner.config.key.expose()),
            )
            .header(http::header::CONTENT_TYPE, "application/json")
            .body(Bytes::from(serde_json::to_vec(body)?))
            .map_err(rig2_core::http::invalid_request)
    }

    /// A chat model.
    pub fn chat(&self, model: impl Into<String>) -> ChatModel {
        ChatModel::new(self.clone(), &model.into())
    }

    /// An embedding model.
    pub fn embedding(&self, model: impl Into<String>) -> EmbeddingModel {
        EmbeddingModel::new(self.clone(), &model.into())
    }

    /// A rerank model.
    pub fn rerank(&self, model: impl Into<String>) -> RerankModel {
        RerankModel::new(self.clone(), &model.into())
    }

    /// Register this client's models with `registry` as `cohere`.
    pub fn register(&self, registry: &mut Registry) {
        fn for_spec(client: &Cohere, spec: &ModelSpec) -> Cohere {
            match &spec.base_url {
                Some(url) => client
                    .inner
                    .config
                    .clone()
                    .with_base_url(url)
                    .connect(Arc::clone(&client.inner.http)),
                None => client.clone(),
            }
        }
        let client = self.clone();
        registry.register::<dyn StreamingModel<Completion>>(PROVIDER, move |spec| {
            Ok(Arc::new(for_spec(&client, spec).chat(&spec.model))
                as Arc<dyn StreamingModel<Completion>>)
        });
        let client = self.clone();
        registry.register::<dyn Model<Embedding>>(PROVIDER, move |spec| {
            Ok(Arc::new(for_spec(&client, spec).embedding(&spec.model))
                as Arc<dyn Model<Embedding>>)
        });
        let client = self.clone();
        registry.register::<dyn Model<Rerank>>(PROVIDER, move |spec| {
            Ok(Arc::new(for_spec(&client, spec).rerank(&spec.model)) as Arc<dyn Model<Rerank>>)
        });
    }
}
