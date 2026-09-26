//! Anthropic for rig2: the Messages API, with extended thinking, prompt
//! caching, citations and tools.
//!
//! ```no_run
//! # async fn demo() -> rig2_core::Result<()> {
//! use rig2_anthropic::Anthropic;
//! use rig2_core::Model;
//!
//! let anthropic = Anthropic::from_env()?;
//! let reply = anthropic.messages("claude-haiku-4-5-20251001").run("Name a prime.").await?;
//! println!("{}", reply.text());
//! # Ok(()) }
//! ```

mod messages;

use std::sync::Arc;

use bytes::Bytes;
use rig2_core::catalog::{Catalog, ModelCard};
use rig2_core::completion::Completion;
use rig2_core::http::{Body, HttpClient, SharedClient, read_json, send};
use rig2_core::spec::Registry;
use rig2_core::tasks::{ListedModel, ModelListing, ModelListingRequest};
use rig2_core::{BoxFuture, Error, ErrorKind, Model, ModelInfo, Result, Secret, StreamingModel};
use serde_json::Value;

pub use messages::{Beta, MessagesModel};

const API_VERSION: &str = "2023-06-01";
const CATALOG: &str = include_str!("../catalogs/anthropic.json");

/// The configuration of an Anthropic-compatible provider.
#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    name: &'static str,
    key: Secret,
    base_url: String,
    catalog: &'static str,
}

impl AnthropicConfig {
    /// Anthropic with `key`.
    pub fn new(key: impl Into<Secret>) -> Self {
        Self {
            name: "anthropic",
            key: key.into(),
            base_url: "https://api.anthropic.com/v1".into(),
            catalog: CATALOG,
        }
    }

    /// Anthropic, with the key from `ANTHROPIC_API_KEY`.
    pub fn from_env() -> Result<Self> {
        Ok(Self::new(Secret::from_env("ANTHROPIC_API_KEY")?))
    }

    /// An Anthropic-compatible provider: its name, base URL, key and
    /// catalog JSON.
    pub fn compatible(
        name: &'static str,
        base_url: impl Into<String>,
        key: impl Into<Secret>,
        catalog: &'static str,
    ) -> Self {
        Self {
            name,
            key: key.into(),
            base_url: base_url.into(),
            catalog,
        }
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
    pub fn connect(self, http: impl HttpClient + 'static) -> Anthropic {
        let catalog = Catalog::parse(self.catalog).unwrap_or_default();
        Anthropic {
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
    config: AnthropicConfig,
    http: SharedClient,
    catalog: Catalog,
}

/// A client for Anthropic or an Anthropic-compatible provider.
#[derive(Debug, Clone)]
pub struct Anthropic {
    inner: Arc<Client>,
}

impl Anthropic {
    /// Anthropic with `key`, on the shared reqwest client.
    #[cfg(feature = "reqwest")]
    pub fn new(key: impl Into<Secret>) -> Self {
        AnthropicConfig::new(key).connect(rig2_reqwest::ReqwestClient::default())
    }

    /// Anthropic with the key from `ANTHROPIC_API_KEY`, on the shared
    /// reqwest client.
    #[cfg(feature = "reqwest")]
    pub fn from_env() -> Result<Self> {
        Ok(AnthropicConfig::from_env()?.connect(rig2_reqwest::ReqwestClient::default()))
    }

    /// The same configuration over another HTTP client.
    pub fn with_http(&self, http: impl HttpClient + 'static) -> Self {
        self.inner.config.clone().connect(http)
    }

    /// The catalog of this provider's models.
    pub fn catalog(&self) -> &Catalog {
        &self.inner.catalog
    }

    fn provider(&self) -> &'static str {
        self.inner.config.name
    }

    fn info(&self, model: &str) -> ModelInfo {
        ModelInfo::new(self.provider(), model)
    }

    fn card(&self, model: &str) -> ModelCard {
        self.inner.catalog.card(model)
    }

    fn http(&self) -> &SharedClient {
        &self.inner.http
    }

    fn request(
        &self,
        method: http::Method,
        path: &str,
        betas: &[&str],
        body: Body,
    ) -> Result<http::Request<Body>> {
        let mut builder = http::Request::builder()
            .method(method)
            .uri(format!("{}{path}", self.inner.config.base_url))
            .header("x-api-key", self.inner.config.key.expose())
            .header("anthropic-version", API_VERSION)
            .header(http::header::CONTENT_TYPE, "application/json");
        if !betas.is_empty() {
            builder = builder.header("anthropic-beta", betas.join(","));
        }
        builder.body(body).map_err(rig2_core::http::invalid_request)
    }

    /// A model on the Messages API.
    pub fn messages(&self, model: impl Into<String>) -> MessagesModel {
        MessagesModel::new(self.clone(), &model.into())
    }

    /// The provider's model listing.
    pub fn models(&self) -> ModelList {
        ModelList {
            info: self.info(""),
            client: self.clone(),
        }
    }

    /// Register this client's models with `registry` under its provider
    /// name. A spec's base URL overrides the client's.
    pub fn register(&self, registry: &mut Registry) {
        let client = self.clone();
        registry.register::<dyn StreamingModel<Completion>>(self.provider(), move |spec| {
            let client = match &spec.base_url {
                Some(url) => client
                    .inner
                    .config
                    .clone()
                    .with_base_url(url)
                    .connect(Arc::clone(&client.inner.http)),
                None => client.clone(),
            };
            Ok(Arc::new(client.messages(&spec.model)) as Arc<dyn StreamingModel<Completion>>)
        });
        let client = self.clone();
        registry.register::<dyn Model<ModelListing>>(self.provider(), move |_| {
            Ok(Arc::new(client.models()) as Arc<dyn Model<ModelListing>>)
        });
    }
}

/// The provider's model listing.
#[derive(Debug, Clone)]
pub struct ModelList {
    client: Anthropic,
    info: ModelInfo,
}

impl Model<ModelListing> for ModelList {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn capabilities(&self) {}

    fn invoke(
        &self,
        _request: ModelListingRequest,
    ) -> BoxFuture<'static, Result<Vec<ListedModel>>> {
        let (http, provider) = (Arc::clone(self.client.http()), self.client.provider());
        let request =
            self.client
                .request(http::Method::GET, "/models?limit=1000", &[], Bytes::new());
        Box::pin(async move {
            let doc = read_json(send(&*http, provider, request?).await?, provider).await?;
            let data = doc
                .get("data")
                .and_then(Value::as_array)
                .ok_or_else(|| Error::new(ErrorKind::Decode, "no data"))?;
            Ok(data
                .iter()
                .filter_map(|m| {
                    Some(ListedModel {
                        id: m.get("id")?.as_str()?.to_owned(),
                        name: m
                            .get("display_name")
                            .and_then(Value::as_str)
                            .map(str::to_owned),
                        owned_by: None,
                    })
                })
                .collect())
        })
    }
}
