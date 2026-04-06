mod byte_level;

use crate::json_structs::{DecoderConfig, DecoderKind};

pub use self::byte_level::ByteLevelDecoder;

/// Errors from constructing or running a decoder.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("unsupported decoder type: {0}")]
    Unsupported(String),
}

/// Extract a plain string from a pattern Value like `{"String": "..."}`.
fn extract_pattern_string(v: &serde_json::Value) -> Option<String> {
    v.get("String").and_then(|s| s.as_str()).map(String::from)
}

/// A compiled decoder ready for use.
#[derive(Debug)]
pub enum Decoder {
    ByteLevel(ByteLevelDecoder),
    Replace { pattern: String, content: String },
    Sequence(Vec<Decoder>),
}

impl Decoder {
    /// Build a decoder from its JSON configuration.
    pub fn from_config(config: DecoderConfig) -> Result<Self, Error> {
        match config {
            DecoderConfig::ByteLevel => Ok(Self::ByteLevel(ByteLevelDecoder)),
            DecoderConfig::ByteFallback => Ok(Self::Sequence(vec![])), // identity/no-op
            DecoderConfig::Fuse => Ok(Self::Sequence(vec![])), // identity/no-op
            DecoderConfig::Replace { pattern, content } => {
                let pat = extract_pattern_string(&pattern)
                    .ok_or_else(|| Error::Unsupported("Replace with non-string pattern".into()))?;
                Ok(Self::Replace { pattern: pat, content })
            }
            DecoderConfig::Sequence { decoders } => {
                let steps = decoders
                    .into_iter()
                    .map(Self::from_config)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Self::Sequence(steps))
            }
            DecoderConfig::Other(v) => {
                let typ = v.get("type").and_then(|t| t.as_str()).unwrap_or("unknown");
                Err(Error::Unsupported(typ.to_string()))
            }
            other => {
                let kind = DecoderKind::from(&other);
                Err(Error::Unsupported(kind.to_string()))
            }
        }
    }

    /// Apply this decoder step to a list of token strings, returning the
    /// transformed list.
    pub fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>, Error> {
        match self {
            Self::ByteLevel(bl) => Ok(bl.decode_chain(tokens)),
            Self::Replace { pattern, content } => {
                Ok(tokens.into_iter()
                    .map(|t| t.replace(pattern.as_str(), content.as_str()))
                    .collect())
            }
            Self::Sequence(steps) => {
                let mut current = tokens;
                for step in steps {
                    current = step.decode_chain(current)?;
                }
                Ok(current)
            }
        }
    }

    /// High-level decode: apply `decode_chain` then join.
    pub fn decode(&self, tokens: Vec<String>) -> Result<String, Error> {
        let result = self.decode_chain(tokens)?;
        Ok(result.join(""))
    }
}
