//! One error for every route in this crate. OCR still carries its own, richer enum.
//!
//! A variant is declared by the layer that produces it and nested here as is:
//! credentials by `litellm_auth` (AWS folds into it at that crate's boundary), the wire by
//! `litellm_http`, secrets by `litellm_secrets`. The transformation layer's [`LlmError`]
//! maps onto the same-named variants once, here, so no route re-declares them.

use std::sync::Arc;

use litellm_core_utils::decline::Decline;
use litellm_llms::Error as LlmError;

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("expected {expected}, got {actual}")]
    InvalidType {
        expected: &'static str,
        actual: &'static str,
    },
    #[error("missing required field: {0}")]
    MissingField(&'static str),
    #[error("invalid provider: {0}")]
    InvalidProvider(String),
    #[error("invalid request: {0}")]
    InvalidRequest(litellm_llms::ErrorDetail),
    #[error("invalid response: {0}")]
    InvalidResponse(litellm_llms::ErrorDetail),
    #[error(transparent)]
    Declined(Decline),
    #[error(transparent)]
    Auth(#[from] litellm_auth::Error),
    #[error(transparent)]
    Headers(#[from] litellm_http::request::HeaderError),
    #[error(transparent)]
    Http(litellm_http::Error),
    #[error(transparent)]
    Secret(#[from] SecretError),
    #[error("post-call hook failed: {0}")]
    PostCallHook(Arc<Error>),
}

impl From<litellm_host::machine::MachineFault> for Error {
    fn from(fault: litellm_host::machine::MachineFault) -> Self {
        use litellm_host::machine::MachineFault;
        Self::InvalidRequest(match fault {
            MachineFault::Abandoned => "host driver was abandoned".into(),
            MachineFault::Protocol(message) => format!("host {message}").into(),
        })
    }
}

impl Error {
    pub fn post_call(error: Self) -> Self {
        Self::PostCallHook(Arc::new(error))
    }

    /// The caller's request is what is wrong, as opposed to the environment, the wire, or
    /// the provider's answer.
    pub fn is_request(&self) -> bool {
        match self {
            Self::InvalidType { .. }
            | Self::MissingField(_)
            | Self::InvalidProvider(_)
            | Self::InvalidRequest(_)
            | Self::Headers(_) => true,
            Self::Auth(error) => !error.is_missing_credential(),
            Self::Http(error) => error.is_request(),
            Self::Declined(_)
            | Self::InvalidResponse(_)
            | Self::Secret(_)
            | Self::PostCallHook(_) => false,
        }
    }

    pub fn is_missing_credential(&self) -> bool {
        matches!(self, Self::Auth(error) if error.is_missing_credential())
    }

    pub fn declined(&self) -> Option<&Decline> {
        match self {
            Self::Declined(decline) => Some(decline),
            Self::InvalidType { .. }
            | Self::MissingField(_)
            | Self::InvalidProvider(_)
            | Self::InvalidRequest(_)
            | Self::InvalidResponse(_)
            | Self::Auth(_)
            | Self::Headers(_)
            | Self::Http(_)
            | Self::Secret(_)
            | Self::PostCallHook(_) => None,
        }
    }

    pub fn after_send(self) -> Self {
        match self {
            already @ Self::InvalidResponse(_) => already,
            upstream if upstream.upstream_response().is_some() => upstream,
            other => Self::InvalidResponse(other.to_string().into()),
        }
    }
}

impl Error {
    pub fn upstream_response(&self) -> Option<(u16, &str)> {
        match self {
            Self::Http(error) => error.upstream_response(),
            _ => None,
        }
    }
}

impl From<litellm_http::Error> for Error {
    fn from(error: litellm_http::Error) -> Self {
        if error.is_computed_header() {
            return Self::Declined(Decline::new("request forwards a header AWS SigV4 computes"));
        }
        Self::Http(error)
    }
}

impl From<LlmError> for Error {
    fn from(error: LlmError) -> Self {
        match error {
            LlmError::InvalidType { expected, actual } => Self::InvalidType { expected, actual },
            LlmError::MissingField(field) => Self::MissingField(field),
            LlmError::InvalidRequest(message) => Self::InvalidRequest(message),
            LlmError::InvalidResponse(message) => Self::InvalidResponse(message),
            LlmError::Declined(decline) => Self::Declined(decline),
            LlmError::Auth(error) => Self::Auth(error),
        }
    }
}

#[derive(Clone, Debug, thiserror::Error)]
#[error(transparent)]
pub struct SecretError(Arc<litellm_secrets::Error>);

impl SecretError {
    pub fn source_error(&self) -> &litellm_secrets::Error {
        &self.0
    }
}

impl From<litellm_secrets::Error> for Error {
    fn from(error: litellm_secrets::Error) -> Self {
        Self::Secret(SecretError(Arc::new(error)))
    }
}

impl PartialEq for SecretError {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for SecretError {}

#[cfg(test)]
mod tests {
    use super::Error;
    use litellm_core_utils::decline::Decline;
    use litellm_llms::{Error as LlmError, ErrorDetail};
    use rstest::rstest;

    #[test]
    fn a_transformation_decline_survives_into_the_route_error() {
        let decline = Decline::new("unrecognized request parameter");
        assert_eq!(
            Error::from(LlmError::Declined(decline)).declined(),
            Some(&decline)
        );
    }

    #[rstest]
    #[case::declined(Error::Declined(Decline::new("non-text response content block")))]
    #[case::missing_field(Error::MissingField("usage"))]
    #[case::invalid_request(Error::InvalidRequest("whatever".into()))]
    #[case::auth(Error::Auth(litellm_auth::Error::InvalidHeader))]
    fn errors_after_the_send_are_never_a_decline(#[case] error: Error) {
        let collapsed = error.after_send();
        assert!(matches!(collapsed, Error::InvalidResponse(_)));
        assert_eq!(collapsed.declined(), None);
    }

    #[test]
    fn an_upstream_status_stays_as_is_after_the_send() {
        let upstream = Error::from(litellm_http::Error::upstream(500, "boom".into()));
        assert_eq!(upstream.clone().after_send(), upstream);
    }

    #[test]
    fn a_post_call_hook_failure_is_never_a_decline() {
        let hook = Error::post_call(Error::Declined(Decline::new("hook")));
        assert_eq!(hook.declined(), None);
    }

    #[test]
    fn a_missing_api_key_is_the_environment_not_the_request() {
        assert!(
            !Error::Auth(litellm_auth::Error::MissingApiKey {
                provider: "Anthropic",
                environment_variable: "ANTHROPIC_API_KEY",
            })
            .is_request()
        );
        assert!(Error::Auth(litellm_auth::Error::InvalidHeader).is_request());
        assert!(Error::InvalidRequest("top_k".into()).is_request());
        assert!(!Error::InvalidResponse("bad json".into()).is_request());
    }
    #[test]
    fn a_header_the_signer_computes_is_left_to_the_host_on_every_route() {
        let error = Error::from(litellm_http::Error::computed_header("authorization"));
        assert!(error.declined().is_some(), "{error:?}");
    }

    #[rstest]
    #[case::request(true)]
    #[case::response(false)]
    fn contextual_errors_render_their_cause_and_route_classification(#[case] request: bool) {
        let source = serde_json::from_str::<serde_json::Value>("{").unwrap_err();
        let source_message = source.to_string();
        let detail = ErrorDetail::invalid("test payload", source);
        let error = Error::from(if request {
            LlmError::InvalidRequest(detail)
        } else {
            LlmError::InvalidResponse(detail)
        });
        assert_eq!(error.is_request(), request);
        let category = if request { "request" } else { "response" };
        assert_eq!(
            error.to_string(),
            format!("invalid {category}: invalid test payload: {source_message}")
        );
        assert!(
            std::error::Error::source(&error).is_none(),
            "a cause rendered in the message is not also exposed as source()"
        );
    }
}
