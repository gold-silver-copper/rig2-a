//! Memory: conversations that outlive a run.
//!
//! A [`Memory`] loads and appends the messages of a named conversation. The
//! agent loads before a prompt and appends what the run added after it.
//! [`InMemory`] keeps conversations in the process; store-backed memories
//! live in their store's crate.

use std::collections::HashMap;
use std::sync::{Arc, PoisonError, RwLock};

use rig2_core::content::Message;
use rig2_core::{BoxFuture, MaybeSend, MaybeSync, Result};

/// Stores conversations by id.
pub trait Memory: MaybeSend + MaybeSync + 'static {
    /// The conversation's messages, oldest first; empty if unknown.
    fn load(&self, conversation: &str) -> BoxFuture<'static, Result<Vec<Message>>>;

    /// Append messages to the conversation.
    fn append(&self, conversation: &str, messages: Vec<Message>) -> BoxFuture<'static, Result<()>>;
}

impl<M: Memory + ?Sized> Memory for Arc<M> {
    fn load(&self, conversation: &str) -> BoxFuture<'static, Result<Vec<Message>>> {
        (**self).load(conversation)
    }

    fn append(&self, conversation: &str, messages: Vec<Message>) -> BoxFuture<'static, Result<()>> {
        (**self).append(conversation, messages)
    }
}

/// Conversations kept in memory. Clones share the same conversations.
#[derive(Debug, Clone, Default)]
pub struct InMemory {
    conversations: Arc<RwLock<HashMap<String, Vec<Message>>>>,
}

impl Memory for InMemory {
    fn load(&self, conversation: &str) -> BoxFuture<'static, Result<Vec<Message>>> {
        let messages = self
            .conversations
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .get(conversation)
            .cloned()
            .unwrap_or_default();
        Box::pin(std::future::ready(Ok(messages)))
    }

    fn append(&self, conversation: &str, messages: Vec<Message>) -> BoxFuture<'static, Result<()>> {
        self.conversations
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .entry(conversation.to_owned())
            .or_default()
            .extend(messages);
        Box::pin(std::future::ready(Ok(())))
    }
}
