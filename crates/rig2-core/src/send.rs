//! `Send` and `Sync` bounds that disappear on WebAssembly, where nothing is
//! sent across threads and many browser types are not `Send`.

use std::future::Future;
use std::pin::Pin;

use futures::Stream;

/// `Send` on native targets, and no bound on `wasm32`.
#[cfg(not(target_family = "wasm"))]
pub trait MaybeSend: Send {}
#[cfg(not(target_family = "wasm"))]
impl<T: Send + ?Sized> MaybeSend for T {}

/// `Send` on native targets, and no bound on `wasm32`.
#[cfg(target_family = "wasm")]
pub trait MaybeSend {}
#[cfg(target_family = "wasm")]
impl<T: ?Sized> MaybeSend for T {}

/// `Sync` on native targets, and no bound on `wasm32`.
#[cfg(not(target_family = "wasm"))]
pub trait MaybeSync: Sync {}
#[cfg(not(target_family = "wasm"))]
impl<T: Sync + ?Sized> MaybeSync for T {}

/// `Sync` on native targets, and no bound on `wasm32`.
#[cfg(target_family = "wasm")]
pub trait MaybeSync {}
#[cfg(target_family = "wasm")]
impl<T: ?Sized> MaybeSync for T {}

/// A boxed future that is `Send` on native targets.
#[cfg(not(target_family = "wasm"))]
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
/// A boxed future that is `Send` on native targets.
#[cfg(target_family = "wasm")]
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

/// A boxed stream that is `Send` on native targets.
#[cfg(not(target_family = "wasm"))]
pub type BoxStream<'a, T> = Pin<Box<dyn Stream<Item = T> + Send + 'a>>;
/// A boxed stream that is `Send` on native targets.
#[cfg(target_family = "wasm")]
pub type BoxStream<'a, T> = Pin<Box<dyn Stream<Item = T> + 'a>>;
