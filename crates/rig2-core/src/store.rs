//! Vector stores: the one trait every backend implements, its portable
//! filter, and an in-memory store.
//!
//! A [`Record`] is an id, a vector and a JSON document. Scores are
//! similarities: higher is closer. A [`Filter`] is a small expression over
//! top-level document fields; each backend translates it, or rejects what it
//! cannot express with [`ErrorKind::Unsupported`].
//!
//! ```
//! # futures::executor::block_on(async {
//! use rig2_core::store::{Filter, InMemoryStore, Query, Record, VectorStore};
//!
//! let store = InMemoryStore::default();
//! store.upsert(vec![
//!     Record::new("a", vec![1.0, 0.0], serde_json::json!({"lang": "en"})),
//!     Record::new("b", vec![0.0, 1.0], serde_json::json!({"lang": "fr"})),
//! ]).await?;
//! let hits = store.search(Query::new(vec![1.0, 0.1], 5).with_filter(Filter::eq("lang", "en"))).await?;
//! assert_eq!(hits.len(), 1);
//! assert_eq!(hits[0].id, "a");
//! # Ok::<(), rig2_core::Error>(()) }).unwrap();
//! ```

use std::collections::BTreeMap;
use std::sync::{Arc, PoisonError, RwLock};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::tasks::{Embedding, EmbeddingRequest, InputType};
use crate::{BoxFuture, Error, ErrorKind, MaybeSend, MaybeSync, Model, Result};

/// A document's JSON fields.
pub type Document = serde_json::Map<String, serde_json::Value>;

/// One stored item: an id, its vector and its document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Record {
    /// The id, unique in the store.
    pub id: String,
    /// The embedding.
    pub vector: Vec<f32>,
    /// The document, whose top-level fields filters can test.
    pub document: Document,
}

impl Record {
    /// A record. A `document` that is not a JSON object is stored under the
    /// key `value`.
    pub fn new(id: impl Into<String>, vector: Vec<f32>, document: serde_json::Value) -> Self {
        let document = match document {
            serde_json::Value::Object(map) => map,
            other => Document::from_iter([("value".to_owned(), other)]),
        };
        Self {
            id: id.into(),
            vector,
            document,
        }
    }
}

/// A similarity search.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Query {
    /// The query vector.
    pub vector: Vec<f32>,
    /// The most results to return.
    pub top_k: u32,
    /// Only consider records matching this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filter: Option<Filter>,
    /// Drop results scored below this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_score: Option<f32>,
}

impl Query {
    /// The `top_k` records closest to `vector`.
    pub fn new(vector: Vec<f32>, top_k: u32) -> Self {
        Self {
            vector,
            top_k,
            filter: None,
            min_score: None,
        }
    }

    /// Restrict the search to records matching `filter`.
    pub fn with_filter(mut self, filter: Filter) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Drop results scored below `min_score`.
    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = Some(min_score);
        self
    }
}

/// A search result with its document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hit {
    /// The record id.
    pub id: String,
    /// The similarity; higher is closer.
    pub score: f32,
    /// The document.
    pub document: Document,
}

impl Hit {
    /// The document as `D`.
    pub fn document<D: DeserializeOwned>(&self) -> Result<D> {
        Ok(serde_json::from_value(serde_json::Value::Object(
            self.document.clone(),
        ))?)
    }
}

/// A search result without its document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IdHit {
    /// The record id.
    pub id: String,
    /// The similarity; higher is closer.
    pub score: f32,
}

/// A vector store backend.
pub trait VectorStore: MaybeSend + MaybeSync {
    /// Insert records, replacing any with the same id.
    fn upsert(&self, records: Vec<Record>) -> BoxFuture<'static, Result<()>>;

    /// The closest records, best first, with their documents.
    fn search(&self, query: Query) -> BoxFuture<'static, Result<Vec<Hit>>>;

    /// The closest record ids, best first, without fetching documents.
    fn search_ids(&self, query: Query) -> BoxFuture<'static, Result<Vec<IdHit>>>;

    /// Delete records by id. Unknown ids are ignored.
    fn delete(&self, ids: Vec<String>) -> BoxFuture<'static, Result<()>>;
}

impl<S: VectorStore + ?Sized> VectorStore for Arc<S> {
    fn upsert(&self, records: Vec<Record>) -> BoxFuture<'static, Result<()>> {
        (**self).upsert(records)
    }
    fn search(&self, query: Query) -> BoxFuture<'static, Result<Vec<Hit>>> {
        (**self).search(query)
    }
    fn search_ids(&self, query: Query) -> BoxFuture<'static, Result<Vec<IdHit>>> {
        (**self).search_ids(query)
    }
    fn delete(&self, ids: Vec<String>) -> BoxFuture<'static, Result<()>> {
        (**self).delete(ids)
    }
}

/// A scalar a filter compares against.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Scalar {
    /// A boolean.
    Bool(bool),
    /// An integer.
    Int(i64),
    /// A float.
    Float(f64),
    /// A string.
    Str(String),
}

impl From<bool> for Scalar {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}
impl From<i64> for Scalar {
    fn from(v: i64) -> Self {
        Self::Int(v)
    }
}
impl From<i32> for Scalar {
    fn from(v: i32) -> Self {
        Self::Int(v.into())
    }
}
impl From<f64> for Scalar {
    fn from(v: f64) -> Self {
        Self::Float(v)
    }
}
impl From<&str> for Scalar {
    fn from(v: &str) -> Self {
        Self::Str(v.to_owned())
    }
}
impl From<String> for Scalar {
    fn from(v: String) -> Self {
        Self::Str(v)
    }
}

impl Scalar {
    /// The value as JSON.
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            Self::Bool(b) => (*b).into(),
            Self::Int(i) => (*i).into(),
            Self::Float(f) => serde_json::Number::from_f64(*f)
                .map_or(serde_json::Value::Null, serde_json::Value::Number),
            Self::Str(s) => s.clone().into(),
        }
    }
}

/// How a comparison compares.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Op {
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
    /// Greater than.
    Gt,
    /// Greater than or equal.
    Gte,
    /// Less than.
    Lt,
    /// Less than or equal.
    Lte,
}

/// A portable filter over top-level document fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Filter {
    /// Compare a field with a value.
    Cmp {
        /// The field.
        field: String,
        /// The comparison.
        op: Op,
        /// The value.
        value: Scalar,
    },
    /// The field equals one of the values.
    In {
        /// The field.
        field: String,
        /// The values.
        values: Vec<Scalar>,
    },
    /// All must match.
    And(Vec<Filter>),
    /// Any must match.
    Or(Vec<Filter>),
    /// Must not match.
    Not(Box<Filter>),
}

impl Filter {
    /// `field == value`.
    pub fn eq(field: impl Into<String>, value: impl Into<Scalar>) -> Self {
        Self::Cmp {
            field: field.into(),
            op: Op::Eq,
            value: value.into(),
        }
    }

    /// `field <op> value`.
    pub fn cmp(field: impl Into<String>, op: Op, value: impl Into<Scalar>) -> Self {
        Self::Cmp {
            field: field.into(),
            op,
            value: value.into(),
        }
    }

    /// `field` is one of `values`.
    pub fn any_of(
        field: impl Into<String>,
        values: impl IntoIterator<Item = impl Into<Scalar>>,
    ) -> Self {
        Self::In {
            field: field.into(),
            values: values.into_iter().map(Into::into).collect(),
        }
    }

    /// Both this and `other`.
    pub fn and(self, other: Self) -> Self {
        match self {
            Self::And(mut all) => {
                all.push(other);
                Self::And(all)
            }
            first => Self::And(vec![first, other]),
        }
    }

    /// This or `other`.
    pub fn or(self, other: Self) -> Self {
        match self {
            Self::Or(mut any) => {
                any.push(other);
                Self::Or(any)
            }
            first => Self::Or(vec![first, other]),
        }
    }

    /// Whether `document` matches. The in-memory store and tests use this;
    /// backends translate the filter instead.
    pub fn matches(&self, document: &Document) -> bool {
        match self {
            Self::Cmp { field, op, value } => {
                document.get(field).is_some_and(|v| compare(v, *op, value))
            }
            Self::In { field, values } => document
                .get(field)
                .is_some_and(|v| values.iter().any(|value| compare(v, Op::Eq, value))),
            Self::And(all) => all.iter().all(|f| f.matches(document)),
            Self::Or(any) => any.iter().any(|f| f.matches(document)),
            Self::Not(inner) => !inner.matches(document),
        }
    }

    /// Every field the filter names.
    pub fn fields(&self) -> Vec<&str> {
        match self {
            Self::Cmp { field, .. } | Self::In { field, .. } => vec![field.as_str()],
            Self::And(all) | Self::Or(all) => all.iter().flat_map(Self::fields).collect(),
            Self::Not(inner) => inner.fields(),
        }
    }

    /// Check that every field name is a plain identifier, so backends can
    /// embed it in their query languages. Fails with
    /// [`ErrorKind::InvalidRequest`].
    pub fn validate(&self) -> Result<()> {
        for field in self.fields() {
            let ok = !field.is_empty()
                && field.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
                && !field.starts_with(|c: char| c.is_ascii_digit());
            if !ok {
                return Err(Error::new(
                    ErrorKind::InvalidRequest,
                    format!("`{field}` is not a valid filter field"),
                ));
            }
        }
        Ok(())
    }
}

fn compare(actual: &serde_json::Value, op: Op, expected: &Scalar) -> bool {
    use std::cmp::Ordering;
    let ordering = match (actual, expected) {
        (serde_json::Value::Bool(a), Scalar::Bool(b)) => Some(a.cmp(b)),
        (serde_json::Value::String(a), Scalar::Str(b)) => Some(a.as_str().cmp(b.as_str())),
        (serde_json::Value::Number(a), Scalar::Int(b)) => {
            a.as_f64().and_then(|a| a.partial_cmp(&(*b as f64)))
        }
        (serde_json::Value::Number(a), Scalar::Float(b)) => {
            a.as_f64().and_then(|a| a.partial_cmp(b))
        }
        _ => None,
    };
    match (op, ordering) {
        (Op::Ne, None) => true,
        (_, None) => false,
        (Op::Eq, Some(o)) => o == Ordering::Equal,
        (Op::Ne, Some(o)) => o != Ordering::Equal,
        (Op::Gt, Some(o)) => o == Ordering::Greater,
        (Op::Gte, Some(o)) => o != Ordering::Less,
        (Op::Lt, Some(o)) => o == Ordering::Less,
        (Op::Lte, Some(o)) => o != Ordering::Greater,
    }
}

/// Cosine similarity; 0 when either vector is zero or they differ in length.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let (mut dot, mut na, mut nb) = (0.0_f32, 0.0_f32, 0.0_f32);
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

/// A store held in memory, searched exhaustively by cosine similarity.
#[derive(Debug, Clone, Default)]
pub struct InMemoryStore {
    records: Arc<RwLock<BTreeMap<String, Record>>>,
}

impl InMemoryStore {
    fn ranked(&self, query: &Query) -> Result<Vec<(f32, Record)>> {
        if let Some(filter) = &query.filter {
            filter.validate()?;
        }
        let records = self.records.read().unwrap_or_else(PoisonError::into_inner);
        let mut scored: Vec<(f32, Record)> = records
            .values()
            .filter(|r| query.filter.as_ref().is_none_or(|f| f.matches(&r.document)))
            .map(|r| (cosine(&query.vector, &r.vector), r.clone()))
            .filter(|(score, _)| query.min_score.is_none_or(|min| *score >= min))
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.id.cmp(&b.1.id)));
        scored.truncate(query.top_k as usize);
        Ok(scored)
    }
}

impl VectorStore for InMemoryStore {
    fn upsert(&self, records: Vec<Record>) -> BoxFuture<'static, Result<()>> {
        let mut map = self.records.write().unwrap_or_else(PoisonError::into_inner);
        for record in records {
            map.insert(record.id.clone(), record);
        }
        Box::pin(std::future::ready(Ok(())))
    }

    fn search(&self, query: Query) -> BoxFuture<'static, Result<Vec<Hit>>> {
        let hits = self.ranked(&query).map(|scored| {
            scored
                .into_iter()
                .map(|(score, r)| Hit {
                    id: r.id,
                    score,
                    document: r.document,
                })
                .collect()
        });
        Box::pin(std::future::ready(hits))
    }

    fn search_ids(&self, query: Query) -> BoxFuture<'static, Result<Vec<IdHit>>> {
        let hits = self.ranked(&query).map(|scored| {
            scored
                .into_iter()
                .map(|(score, r)| IdHit { id: r.id, score })
                .collect()
        });
        Box::pin(std::future::ready(hits))
    }

    fn delete(&self, ids: Vec<String>) -> BoxFuture<'static, Result<()>> {
        let mut map = self.records.write().unwrap_or_else(PoisonError::into_inner);
        for id in ids {
            map.remove(&id);
        }
        Box::pin(std::future::ready(Ok(())))
    }
}

pub use rig2_macros::Embed;

/// A type whose chosen fields are embedded for search.
///
/// Derive it with `#[derive(Embed)]`, marking the fields to embed with
/// `#[embed]`.
///
/// ```
/// use rig2_core::store::Embed;
///
/// #[derive(Embed)]
/// struct Article {
///     #[embed]
///     title: String,
///     #[embed]
///     body: String,
///     year: u32,
/// }
///
/// let a = Article { title: "Rust".into(), body: "Ownership.".into(), year: 2015 };
/// assert_eq!(a.embed_text(), "Rust\nOwnership.");
/// ```
pub trait Embed {
    /// The text to embed.
    fn embed_text(&self) -> String;
}

/// Embed documents and turn them into records, ready to upsert.
///
/// Each item is an id and a document; the document's [`Embed`] text is
/// embedded with `model` as [`InputType::Document`], and its JSON form is
/// stored.
pub async fn embed_records<D, M>(model: &M, items: Vec<(String, D)>) -> Result<Vec<Record>>
where
    D: Embed + Serialize + MaybeSend,
    M: Model<Embedding> + ?Sized,
{
    if items.is_empty() {
        return Ok(Vec::new());
    }
    let texts: Vec<String> = items.iter().map(|(_, d)| d.embed_text()).collect();
    let response = model
        .invoke(EmbeddingRequest::new(texts).with_input_type(InputType::Document))
        .await?;
    if response.embeddings.len() != items.len() {
        return Err(Error::new(
            ErrorKind::Decode,
            format!(
                "asked for {} embeddings, got {}",
                items.len(),
                response.embeddings.len()
            ),
        ));
    }
    items
        .into_iter()
        .zip(response.embeddings)
        .map(|((id, document), vector)| {
            Ok(Record::new(id, vector, serde_json::to_value(&document)?))
        })
        .collect()
}

#[cfg(test)]
mod tests;
