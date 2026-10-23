"""
Tests for the Realtime transcription_sessions surface used by gpt-realtime-whisper:
  - RealtimeTranscriptionSessionRequest model-resolution + passthrough
  - the beta transcription_sessions shape minted through GA client_secrets
"""

import email
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest


from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.openai.realtime.http_transformation import OpenAIRealtimeHTTPConfig
from litellm.types.realtime import RealtimeTranscriptionSessionRequest


def test_request_resolves_model_returns_none_when_both_absent():
    req = RealtimeTranscriptionSessionRequest(input_audio_format="pcm16")
    assert req.resolved_model() is None

    req = RealtimeTranscriptionSessionRequest(
        model="openai/gpt-realtime-whisper",
        input_audio_transcription={"model": "gpt-realtime-whisper"},
    )
    assert req.resolved_model() == "openai/gpt-realtime-whisper"


def test_request_resolves_model_from_input_audio_transcription():
    req = RealtimeTranscriptionSessionRequest(
        input_audio_transcription={"model": "gpt-realtime-whisper", "language": "en"},
    )
    assert req.resolved_model() == "gpt-realtime-whisper"


@pytest.mark.asyncio
async def test_client_secret_handler_still_targets_client_secrets_url():
    """Refactor regression: the client_secrets handler must keep its own URL."""
    handler = BaseLLMHTTPHandler()

    mock_response = MagicMock(spec=httpx.Response)
    mock_client = MagicMock(spec=AsyncHTTPHandler)
    mock_client.post = AsyncMock(return_value=mock_response)

    logging_obj = MagicMock()
    logging_obj.pre_call = MagicMock()

    await handler.async_realtime_client_secret_handler(
        api_base="https://api.openai.com",
        api_key="sk-test",
        request_data={"session": {"type": "realtime"}},
        logging_obj=logging_obj,
        timeout=10.0,
        provider_config=OpenAIRealtimeHTTPConfig(),
        model="gpt-4o-realtime-preview",
        client=mock_client,
    )

    _, kwargs = mock_client.post.call_args
    assert kwargs["url"] == "https://api.openai.com/v1/realtime/client_secrets"


@pytest.mark.asyncio
async def test_sdk_fn_mints_an_openai_transcription_session_through_ga_client_secrets(monkeypatch):
    import litellm

    monkeypatch.setenv("OPENAI_API_KEY", "sk-unit-test")
    requests: list[httpx.Request] = []

    def openai(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"value": "ek_upstream", "expires_at": 1792775579, "session": {"id": "sess_1"}})

    response = await litellm.acreate_realtime_transcription_session(
        model="openai/gpt-realtime-whisper",
        transcription_session={"input_audio_transcription": {"model": "gpt-realtime-whisper"}},
        client=AsyncHTTPHandler(transport=httpx.MockTransport(openai)),
    )

    (request,) = requests
    assert str(request.url) == "https://api.openai.com/v1/realtime/client_secrets"
    assert json.loads(request.content) == {
        "session": {"type": "transcription", "audio": {"input": {"transcription": {"model": "gpt-realtime-whisper"}}}}
    }
    assert response.json() == {"id": "sess_1", "client_secret": {"value": "ek_upstream", "expires_at": 1792775579}}


@pytest.mark.asyncio
async def test_an_openai_compatible_provider_mints_its_transcription_session_at_client_secrets():
    import litellm

    requests: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"value": "ek_upstream"})

    await litellm.acreate_realtime_transcription_session(
        model="hosted_vllm/whisper",
        api_base="https://vllm.example",
        api_key="vllm-key",
        transcription_session={"input_audio_transcription": {"language": "en"}},
        client=AsyncHTTPHandler(transport=httpx.MockTransport(upstream)),
    )

    (request,) = requests
    assert str(request.url) == "https://vllm.example/v1/realtime/client_secrets"
    assert request.headers["authorization"] == "Bearer vllm-key"


def _gpt_live_transcribe_session() -> dict[str, object]:
    return {"type": "transcription", "audio": {"input": {"transcription": {"model": "gpt-live-transcribe"}}}}


def _multipart_session(request: httpx.Request) -> object:
    multipart = email.message_from_bytes(
        f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode() + request.content
    )
    (session,) = (
        part for part in multipart.get_payload() if part.get_param("name", header="content-disposition") == "session"
    )
    return json.loads(session.get_payload())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "session", "expected_session"),
    [
        ("openai/gpt-live-transcribe", _gpt_live_transcribe_session(), _gpt_live_transcribe_session()),
        ("openai/gpt-realtime", {"type": "realtime"}, {"type": "realtime", "model": "gpt-realtime"}),
    ],
)
async def test_sdp_exchange_names_a_top_level_model_only_for_realtime_sessions(
    monkeypatch, model, session, expected_session
):
    import litellm

    monkeypatch.setenv("OPENAI_API_KEY", "sk-unit-test")
    requests: list[httpx.Request] = []

    def openai(request: httpx.Request) -> httpx.Response:
        request.read()
        requests.append(request)
        return httpx.Response(201, text="v=0")

    await litellm.arealtime_calls(
        openai_ephemeral_key="ek_upstream",
        sdp_body=b"v=0",
        model=model,
        session=session,
        client=AsyncHTTPHandler(transport=httpx.MockTransport(openai)),
    )

    (request,) = requests
    assert _multipart_session(request) == expected_session


def test_append_query_params_skips_existing_keys():
    from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler

    url = "wss://example.com/v1/realtime?model=gpt-4o"
    result = BaseLLMHTTPHandler._append_query_params(url, {"model": "ignored", "intent": "transcription"})
    assert "model=ignored" not in result
    assert "intent=transcription" in result


def test_append_query_params_no_params_returns_unchanged():
    from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler

    url = "wss://example.com/v1/realtime?model=gpt-4o"
    assert BaseLLMHTTPHandler._append_query_params(url, None) == url
    assert BaseLLMHTTPHandler._append_query_params(url, {}) == url


def test_append_query_params_encodes_special_chars():
    from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler

    url = "wss://example.com/v1/realtime"
    result = BaseLLMHTTPHandler._append_query_params(url, {"intent": "a&b=c"})
    assert "intent=a%26b%3Dc" in result
    assert "&b=c" not in result


def test_azure_construct_url_encodes_model_and_api_version():
    """model and api-version must be URL-encoded to prevent query-string injection."""
    from litellm.llms.azure.realtime.handler import AzureOpenAIRealtime

    h = AzureOpenAIRealtime()
    url = h.construct_url(
        "https://x.openai.azure.com",
        "deploy&evil=1",
        "2024-10-01-preview",
    )
    assert "evil=1" not in url.split("?", 1)[1]

    url_ga = h.construct_url(
        "https://x.openai.azure.com",
        "deploy&evil=1",
        None,
        realtime_protocol="GA",
    )
    assert "evil=1" not in url_ga.split("?", 1)[1]
