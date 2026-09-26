//! Google Gemini for rig2: GenerateContent (completion and image output),
//! embeddings, thinking, and the model listing.
//!
//! ```no_run
//! # async fn demo() -> rig2_core::Result<()> {
//! use rig2_core::Model;
//! use rig2_gemini::Gemini;
//!
//! let gemini = Gemini::from_env()?;
//! let reply = gemini.generate("gemini-2.5-flash-lite").run("Name a prime.").await?;
//! println!("{}", reply.text());
//! # Ok(()) }
//! ```

mod generate;
mod media;

use std::sync::Arc;

use bytes::Bytes;
use rig2_core::catalog::{Catalog, ModelCard};
use rig2_core::completion::Completion;
use rig2_core::http::{Body, HttpClient, SharedClient};
use rig2_core::spec::{ModelSpec, Registry};
use rig2_core::tasks::{Embedding, ImageGeneration, ModelListing};
use rig2_core::{Model, ModelInfo, Result, Secret, StreamingModel};

pub use generate::{GenerateModel, ThoughtSignature};
pub use media::{EmbeddingModel, ImageModel, ModelList};

const CATALOG: &str = include_str!("../catalogs/gemini.json");

/// The configuration of the Gemini API: key and base URL.
#[derive(Debug, Clone)]
pub struct GeminiConfig {
    key: Secret,
    base_url: String,
}

impl GeminiConfig {
    /// Gemini with `key`.
    pub fn new(key: impl Into<Secret>) -> Self {
        Self {
            key: key.into(),
            base_url: "https://generativelanguage.googleapis.com/v1beta".into(),
        }
    }

    /// Gemini, with the key from `GEMINI_API_KEY`.
    pub fn from_env() -> Result<Self> {
        Ok(Self::new(Secret::from_env("GEMINI_API_KEY")?))
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
    pub fn connect(self, http: impl HttpClient + 'static) -> Gemini {
        let catalog = Catalog::parse(CATALOG).unwrap_or_default();
        Gemini {
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
    config: GeminiConfig,
    http: SharedClient,
    catalog: Catalog,
}

/// A client for the Gemini API.
#[derive(Debug, Clone)]
pub struct Gemini {
    inner: Arc<Client>,
}

const PROVIDER: &str = "gemini";

impl Gemini {
    /// Gemini with `key`, on the shared reqwest client.
    #[cfg(feature = "reqwest")]
    pub fn new(key: impl Into<Secret>) -> Self {
        GeminiConfig::new(key).connect(rig2_reqwest::ReqwestClient::default())
    }

    /// Gemini with the key from `GEMINI_API_KEY`, on the shared reqwest
    /// client.
    #[cfg(feature = "reqwest")]
    pub fn from_env() -> Result<Self> {
        Ok(GeminiConfig::from_env()?.connect(rig2_reqwest::ReqwestClient::default()))
    }

    /// The same configuration over another HTTP client.
    pub fn with_http(&self, http: impl HttpClient + 'static) -> Self {
        self.inner.config.clone().connect(http)
    }

    /// The catalog of Gemini's models.
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

    fn request(&self, method: http::Method, path: &str, body: Body) -> Result<http::Request<Body>> {
        http::Request::builder()
            .method(method)
            .uri(format!("{}{path}", self.inner.config.base_url))
            .header("x-goog-api-key", self.inner.config.key.expose())
            .header(http::header::CONTENT_TYPE, "application/json")
            .body(body)
            .map_err(rig2_core::http::invalid_request)
    }

    fn post_json(&self, path: &str, body: &serde_json::Value) -> Result<http::Request<Body>> {
        self.request(
            http::Method::POST,
            path,
            Bytes::from(serde_json::to_vec(body)?),
        )
    }

    /// A model on GenerateContent, for completion.
    pub fn generate(&self, model: impl Into<String>) -> GenerateModel {
        GenerateModel::new(self.clone(), &model.into())
    }

    /// An embedding model.
    pub fn embedding(&self, model: impl Into<String>) -> EmbeddingModel {
        EmbeddingModel::new(self.clone(), &model.into())
    }

    /// An image generation model: a Gemini model with image output.
    pub fn image_generation(&self, model: impl Into<String>) -> ImageModel {
        ImageModel::new(self.clone(), &model.into())
    }

    /// Gemini's model listing.
    pub fn models(&self) -> ModelList {
        ModelList::new(self.clone())
    }

    /// Register this client's models with `registry` as `gemini`.
    pub fn register(&self, registry: &mut Registry) {
        fn for_spec(client: &Gemini, spec: &ModelSpec) -> Gemini {
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
            Ok(Arc::new(for_spec(&client, spec).generate(&spec.model))
                as Arc<dyn StreamingModel<Completion>>)
        });
        let client = self.clone();
        registry.register::<dyn Model<Embedding>>(PROVIDER, move |spec| {
            Ok(Arc::new(for_spec(&client, spec).embedding(&spec.model))
                as Arc<dyn Model<Embedding>>)
        });
        let client = self.clone();
        registry.register::<dyn Model<ImageGeneration>>(PROVIDER, move |spec| {
            Ok(
                Arc::new(for_spec(&client, spec).image_generation(&spec.model))
                    as Arc<dyn Model<ImageGeneration>>,
            )
        });
        let client = self.clone();
        registry.register::<dyn Model<ModelListing>>(PROVIDER, move |spec| {
            Ok(Arc::new(for_spec(&client, spec).models()) as Arc<dyn Model<ModelListing>>)
        });
    }
}
