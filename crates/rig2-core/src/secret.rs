//! A credential that never prints.

use std::fmt;

use crate::{Error, ErrorKind, Result};

/// An API key or token. Its `Debug` output is redacted, it is not
/// serializable, and it is only readable through [`Secret::expose`].
#[derive(Clone, PartialEq, Eq)]
pub struct Secret(String);

impl Secret {
    /// Wrap a credential.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Read the credential from the environment variable `name`.
    ///
    /// Fails with [`ErrorKind::Config`] when it is unset or empty.
    pub fn from_env(name: &str) -> Result<Self> {
        match std::env::var(name) {
            Ok(value) if !value.trim().is_empty() => Ok(Self(value.trim().to_owned())),
            _ => Err(Error::new(
                ErrorKind::Config,
                format!("the environment variable `{name}` is not set"),
            )),
        }
    }

    /// The credential itself, for building a request.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Secret([REDACTED])")
    }
}

impl From<&str> for Secret {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for Secret {
    fn from(value: String) -> Self {
        Self(value)
    }
}
