use std::fmt::Write as _;

use bytes::{BufMut, Bytes, BytesMut};

/// A `multipart/form-data` body.
///
/// ```
/// use rig2_core::http::Multipart;
///
/// let (content_type, body) = Multipart::new().text("model", "whisper-1").finish();
/// assert!(content_type.starts_with("multipart/form-data; boundary="));
/// assert!(!body.is_empty());
/// ```
#[derive(Debug)]
pub struct Multipart {
    boundary: String,
    body: BytesMut,
}

impl Default for Multipart {
    fn default() -> Self {
        Self::new()
    }
}

impl Multipart {
    /// An empty form with a fixed boundary.
    ///
    /// The boundary is fixed so that bodies are reproducible in recordings;
    /// it is long and unusual enough not to occur in real payloads.
    pub fn new() -> Self {
        Self {
            boundary: "rig2-form-boundary-7d3f1c9a2b6e4058".to_owned(),
            body: BytesMut::new(),
        }
    }

    /// Add a text field.
    pub fn text(mut self, name: &str, value: &str) -> Self {
        self.header(name, None, None);
        self.body.put_slice(value.as_bytes());
        self.body.put_slice(b"\r\n");
        self
    }

    /// Add a file field.
    pub fn file(mut self, name: &str, filename: &str, media_type: &str, bytes: &[u8]) -> Self {
        self.header(name, Some(filename), Some(media_type));
        self.body.put_slice(bytes);
        self.body.put_slice(b"\r\n");
        self
    }

    fn header(&mut self, name: &str, filename: Option<&str>, media_type: Option<&str>) {
        let mut header = format!(
            "--{}\r\nContent-Disposition: form-data; name=\"{name}\"",
            self.boundary
        );
        if let Some(filename) = filename {
            let _ = write!(header, "; filename=\"{filename}\"");
        }
        header.push_str("\r\n");
        if let Some(media_type) = media_type {
            let _ = write!(header, "Content-Type: {media_type}\r\n");
        }
        header.push_str("\r\n");
        self.body.put_slice(header.as_bytes());
    }

    /// The `Content-Type` header value and the body.
    pub fn finish(mut self) -> (String, Bytes) {
        self.body
            .put_slice(format!("--{}--\r\n", self.boundary).as_bytes());
        (
            format!("multipart/form-data; boundary={}", self.boundary),
            self.body.freeze(),
        )
    }
}
