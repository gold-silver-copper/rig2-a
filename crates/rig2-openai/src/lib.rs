//! OpenAI for rig2, and every provider that speaks OpenAI's API.
//!
//! An [`OpenAI`] client is a configuration plus an HTTP client. Its methods
//! return models: [`OpenAI::responses`] and [`OpenAI::chat`] for completion,
//! and [`OpenAI::embedding`], [`OpenAI::image_generation`],
//! [`OpenAI::speech`], [`OpenAI::transcription`], [`OpenAI::moderation`]
//! and [`OpenAI::models`] for the other tasks. OpenAI-compatible providers
//! are [`Dialect`]s: data, not separate implementations.
//!
//! ```no_run
//! # async fn demo() -> rig2_core::Result<()> {
//! use rig2_core::Model;
//! use rig2_openai::OpenAI;
//!
//! let openai = OpenAI::from_env()?;
//! let reply = openai.responses("gpt-5-mini").run("Name a prime.").await?;
//! println!("{}", reply.text());
//! # Ok(()) }
//! ```

mod chat;
mod dialect;
mod media;
mod responses;

use std::sync::Arc;

use rig2_core::catalog::{Catalog, ModelCard};
use rig2_core::completion::Completion;
use rig2_core::http::{Body, HttpClient, SharedClient};
use rig2_core::spec::{ModelSpec, Registry};
use rig2_core::tasks::{
    Embedding, ImageGeneration, ModelListing, Moderation, Speech, Transcription,
};
use rig2_core::{Error, ErrorKind, Model, ModelInfo, Result, Secret, StreamingModel};

pub use chat::{ChatModel, ExtraBody, ImageDetail, chat_request_body};
pub use dialect::{Dialect, MaxTokens, OPENAI, Quirks};
pub use media::{
    EmbeddingModel, ImageModel, ModelList, ModerationModel, SpeechModel, TranscriptionModel,
};
pub use responses::{ItemId, ReasoningSummary, ResponsesModel};

/// The configuration of an OpenAI-compatible provider: dialect, key and
/// base URL. Connect it to an HTTP client to get an [`OpenAI`] client.
#[derive(Debug, Clone)]
pub struct OpenAIConfig {
    dialect: Dialect,
    key: Secret,
    base_url: String,
    headers: Vec<(String, String)>,
}

impl OpenAIConfig {
    /// OpenAI with `key`.
    pub fn new(key: impl Into<Secret>) -> Self {
        Self::for_dialect(dialect::OPENAI, key)
    }

    /// OpenAI, with the key from `OPENAI_API_KEY`.
    pub fn from_env() -> Result<Self> {
        Self::dialect_from_env(dialect::OPENAI)
    }

    /// A provider speaking `dialect`, with `key`.
    pub fn for_dialect(dialect: Dialect, key: impl Into<Secret>) -> Self {
        Self {
            base_url: dialect.base_url.to_owned(),
            dialect,
            key: key.into(),
            headers: Vec::new(),
        }
    }

    /// A provider speaking `dialect`, with the key from its variable.
    pub fn dialect_from_env(dialect: Dialect) -> Result<Self> {
        let key = Secret::from_env(dialect.key_var)?;
        Ok(Self::for_dialect(dialect, key))
    }

    /// Use another base URL, such as a proxy.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        base_url
            .into()
            .trim_end_matches('/')
            .clone_into(&mut self.base_url);
        self
    }

    /// Send an extra header with every request, such as
    /// `OpenAI-Organization`.
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// The client over `http`.
    pub fn connect(self, http: impl HttpClient + 'static) -> OpenAI {
        let catalog = Catalog::parse(self.dialect.catalog).unwrap_or_default();
        OpenAI {
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
    config: OpenAIConfig,
    http: SharedClient,
    catalog: Catalog,
}

/// A client for OpenAI or an OpenAI-compatible provider.
#[derive(Debug, Clone)]
pub struct OpenAI {
    inner: Arc<Client>,
}

impl OpenAI {
    /// OpenAI with `key`, on the shared reqwest client.
    #[cfg(feature = "reqwest")]
    pub fn new(key: impl Into<Secret>) -> Self {
        OpenAIConfig::new(key).connect(rig2_reqwest::ReqwestClient::default())
    }

    /// OpenAI with the key from `OPENAI_API_KEY`, on the shared reqwest
    /// client.
    #[cfg(feature = "reqwest")]
    pub fn from_env() -> Result<Self> {
        Ok(OpenAIConfig::from_env()?.connect(rig2_reqwest::ReqwestClient::default()))
    }

    /// A provider speaking `dialect`, with the key from its variable, on the
    /// shared reqwest client.
    #[cfg(feature = "reqwest")]
    pub fn dialect_from_env(dialect: Dialect) -> Result<Self> {
        Ok(
            OpenAIConfig::dialect_from_env(dialect)?
                .connect(rig2_reqwest::ReqwestClient::default()),
        )
    }

    /// The same configuration over another HTTP client.
    pub fn with_http(&self, http: impl HttpClient + 'static) -> Self {
        self.inner.config.clone().connect(http)
    }

    /// The dialect.
    pub fn dialect(&self) -> &Dialect {
        &self.inner.config.dialect
    }

    /// The catalog of this provider's models.
    pub fn catalog(&self) -> &Catalog {
        &self.inner.catalog
    }

    fn info(&self, model: &str) -> ModelInfo {
        ModelInfo::new(self.inner.config.dialect.name, model)
    }

    fn card(&self, model: &str) -> ModelCard {
        self.inner.catalog.card(model)
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.inner.config.base_url)
    }

    fn provider(&self) -> &'static str {
        self.inner.config.dialect.name
    }

    fn http(&self) -> &SharedClient {
        &self.inner.http
    }

    /// Build a request with the provider's auth and extra headers.
    fn request(
        &self,
        method: http::Method,
        path: &str,
        content_type: Option<&str>,
        body: Body,
    ) -> Result<http::Request<Body>> {
        let config = &self.inner.config;
        let mut builder = http::Request::builder()
            .method(method)
            .uri(self.url(path))
            .header(
                http::header::AUTHORIZATION,
                format!("Bearer {}", config.key.expose()),
            );
        if let Some(content_type) = content_type {
            builder = builder.header(http::header::CONTENT_TYPE, content_type);
        }
        for (name, value) in &config.headers {
            builder = builder.header(name, value);
        }
        builder.body(body).map_err(rig2_core::http::invalid_request)
    }

    fn post_json(&self, path: &str, body: &serde_json::Value) -> Result<http::Request<Body>> {
        let bytes = bytes::Bytes::from(serde_json::to_vec(body)?);
        self.request(http::Method::POST, path, Some("application/json"), bytes)
    }

    /// A model on the Responses API.
    pub fn responses(&self, model: impl Into<String>) -> ResponsesModel {
        ResponsesModel::new(self.clone(), &model.into())
    }

    /// A model on the Chat Completions API.
    pub fn chat(&self, model: impl Into<String>) -> ChatModel {
        ChatModel::new(self.clone(), &model.into())
    }

    /// An embedding model.
    pub fn embedding(&self, model: impl Into<String>) -> EmbeddingModel {
        EmbeddingModel::new(self.clone(), &model.into())
    }

    /// An image generation model.
    pub fn image_generation(&self, model: impl Into<String>) -> ImageModel {
        ImageModel::new(self.clone(), &model.into())
    }

    /// A text-to-speech model.
    pub fn speech(&self, model: impl Into<String>) -> SpeechModel {
        SpeechModel::new(self.clone(), &model.into())
    }

    /// A speech-to-text model.
    pub fn transcription(&self, model: impl Into<String>) -> TranscriptionModel {
        TranscriptionModel::new(self.clone(), &model.into())
    }

    /// A moderation model.
    pub fn moderation(&self, model: impl Into<String>) -> ModerationModel {
        ModerationModel::new(self.clone(), &model.into())
    }

    /// The provider's model listing.
    pub fn models(&self) -> ModelList {
        ModelList::new(self.clone())
    }

    /// Register this client's models with `registry` under the dialect's
    /// name.
    ///
    /// Completion specs take the option `api`: `responses` or `chat`. The
    /// default is the dialect's preferred API. A spec's base URL overrides
    /// the client's.
    pub fn register(&self, registry: &mut Registry) {
        let name = self.provider();
        let client = self.clone();
        registry.register::<dyn StreamingModel<Completion>>(name, move |spec| {
            let client = client.for_spec(spec);
            match spec
                .options
                .get("api")
                .map_or_else(|| client.dialect().preferred_api, String::as_str)
            {
                "responses" => {
                    Ok(Arc::new(client.responses(&spec.model))
                        as Arc<dyn StreamingModel<Completion>>)
                }
                "chat" => Ok(Arc::new(client.chat(&spec.model))),
                other => Err(Error::new(
                    ErrorKind::Config,
                    format!("unknown OpenAI API `{other}`"),
                )),
            }
        });
        macro_rules! factory {
            ($task:ty, $method:ident) => {{
                let client = self.clone();
                registry.register::<dyn Model<$task>>(name, move |spec: &ModelSpec| {
                    Ok(Arc::new(client.for_spec(spec).$method(&spec.model))
                        as Arc<dyn Model<$task>>)
                });
            }};
        }
        factory!(Embedding, embedding);
        factory!(ImageGeneration, image_generation);
        factory!(Speech, speech);
        factory!(Transcription, transcription);
        factory!(Moderation, moderation);
        let client = self.clone();
        registry.register::<dyn Model<ModelListing>>(name, move |spec: &ModelSpec| {
            Ok(Arc::new(client.for_spec(spec).models()) as Arc<dyn Model<ModelListing>>)
        });
    }

    fn for_spec(&self, spec: &ModelSpec) -> Self {
        match &spec.base_url {
            Some(base_url) => {
                let config = self.inner.config.clone().with_base_url(base_url);
                Self {
                    inner: Arc::new(Client {
                        config,
                        http: Arc::clone(&self.inner.http),
                        catalog: self.inner.catalog.clone(),
                    }),
                }
            }
            None => self.clone(),
        }
    }
}
