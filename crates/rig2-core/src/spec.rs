//! Model specs: which model to use, as plain data, and the registry that
//! turns a spec into a live model.
//!
//! A [`ModelSpec`] is what configuration files, checkpoints and recordings
//! store. It never holds credentials or a client. A [`Registry`] maps a
//! provider name to a factory that builds a model from a spec; provider
//! crates register their factories explicitly, and the registry keys each
//! factory by the model type it builds, so one provider can register a
//! completion factory, an embedding factory and so on.
//!
//! ```
//! use rig2_core::spec::ModelSpec;
//!
//! let spec: ModelSpec = "anthropic:claude-sonnet-4-6".parse().unwrap();
//! assert_eq!(spec.provider, "anthropic");
//! assert_eq!(spec.model, "claude-sonnet-4-6");
//! assert_eq!(spec.to_string(), "anthropic:claude-sonnet-4-6");
//! ```

use std::any::{Any, TypeId};
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::str::FromStr;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{Error, ErrorKind, Result};

/// Which model to use, as plain, serializable data.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelSpec {
    /// The provider name, as registered.
    pub provider: String,
    /// The provider's model id.
    pub model: String,
    /// A base URL overriding the provider's default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    /// Provider-interpreted options, such as which API to use.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub options: BTreeMap<String, String>,
}

impl ModelSpec {
    /// A spec for `model` at `provider`.
    pub fn new(provider: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            base_url: None,
            options: BTreeMap::new(),
        }
    }

    /// Set a provider option.
    pub fn with_option(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.options.insert(key.into(), value.into());
        self
    }

    /// Override the base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }
}

impl FromStr for ModelSpec {
    type Err = Error;

    /// Parse `provider:model`. The model id may itself contain colons.
    fn from_str(s: &str) -> Result<Self> {
        match s.split_once(':') {
            Some((provider, model)) if !provider.is_empty() && !model.is_empty() => {
                Ok(Self::new(provider, model))
            }
            _ => Err(Error::new(
                ErrorKind::Config,
                format!("`{s}` is not `provider:model`"),
            )),
        }
    }
}

impl fmt::Display for ModelSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.provider, self.model)
    }
}

#[cfg(not(target_family = "wasm"))]
type Factory<M> = Arc<dyn Fn(&ModelSpec) -> Result<Arc<M>> + Send + Sync>;
#[cfg(target_family = "wasm")]
type Factory<M> = Arc<dyn Fn(&ModelSpec) -> Result<Arc<M>>>;

#[cfg(not(target_family = "wasm"))]
type AnyFactory = Box<dyn Any + Send + Sync>;
#[cfg(target_family = "wasm")]
type AnyFactory = Box<dyn Any>;

/// Builds live models from [`ModelSpec`]s.
///
/// `M` is the model type a factory builds, usually a trait object such as
/// `dyn StreamingModel<Completion>` or `dyn Model<Embedding>`.
///
/// ```
/// use std::sync::Arc;
/// use rig2_core::spec::{ModelSpec, Registry};
///
/// let mut registry = Registry::default();
/// registry.register::<str>("echo", |spec| Ok(Arc::from(spec.model.as_str())));
/// let built = registry.build::<str>(&"echo:hello".parse().unwrap()).unwrap();
/// assert_eq!(&*built, "hello");
/// ```
#[derive(Default)]
pub struct Registry {
    factories: HashMap<(String, TypeId), AnyFactory>,
}

impl fmt::Debug for Registry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut providers: Vec<&str> = self.factories.keys().map(|(p, _)| p.as_str()).collect();
        providers.sort_unstable();
        providers.dedup();
        f.debug_struct("Registry")
            .field("providers", &providers)
            .finish()
    }
}

impl Registry {
    /// Register how `provider` builds models of type `M`, replacing any
    /// earlier factory for the same pair.
    #[cfg(not(target_family = "wasm"))]
    pub fn register<M: ?Sized + 'static>(
        &mut self,
        provider: impl Into<String>,
        factory: impl Fn(&ModelSpec) -> Result<Arc<M>> + Send + Sync + 'static,
    ) {
        let factory: Factory<M> = Arc::new(factory);
        self.factories
            .insert((provider.into(), TypeId::of::<M>()), Box::new(factory));
    }

    /// Register how `provider` builds models of type `M`, replacing any
    /// earlier factory for the same pair.
    #[cfg(target_family = "wasm")]
    pub fn register<M: ?Sized + 'static>(
        &mut self,
        provider: impl Into<String>,
        factory: impl Fn(&ModelSpec) -> Result<Arc<M>> + 'static,
    ) {
        let factory: Factory<M> = Arc::new(factory);
        self.factories
            .insert((provider.into(), TypeId::of::<M>()), Box::new(factory));
    }

    /// Build the model `spec` names.
    ///
    /// Fails with [`ErrorKind::Config`] when no factory for the provider and
    /// model type is registered, or with the factory's own error.
    pub fn build<M: ?Sized + 'static>(&self, spec: &ModelSpec) -> Result<Arc<M>> {
        let factory = self
            .factories
            .get(&(spec.provider.clone(), TypeId::of::<M>()))
            .and_then(|f| f.downcast_ref::<Factory<M>>())
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::Config,
                    format!(
                        "no `{}` factory registered for provider `{}`",
                        std::any::type_name::<M>(),
                        spec.provider
                    ),
                )
            })?;
        factory(spec)
    }

    /// Whether `provider` has a factory for `M`.
    pub fn contains<M: ?Sized + 'static>(&self, provider: &str) -> bool {
        self.factories
            .contains_key(&(provider.to_owned(), TypeId::of::<M>()))
    }
}

#[cfg(test)]
mod tests;
