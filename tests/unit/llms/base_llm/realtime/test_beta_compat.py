from typing import Final

import pytest
from pydantic import JsonValue

from litellm.llms.base_llm.realtime.beta_compat import (
    beta_transcription_session,
    ga_audio_format,
    ga_transcription_client_secret_request,
)

_TRANSCRIPTION: Final[dict[str, JsonValue]] = {"model": "gpt-realtime-whisper", "language": "en", "prompt": ""}
_TURN_DETECTION: Final[dict[str, JsonValue]] = {"type": "server_vad", "threshold": 0.5, "silence_duration_ms": 500}
_NOISE_REDUCTION: Final[dict[str, JsonValue]] = {"type": "near_field"}
_INCLUDE: Final[list[JsonValue]] = ["item.input_audio_transcription.logprobs"]
_BETA_FORMATS: Final = ("pcm16", "g711_ulaw", "g711_alaw")


def test_a_beta_transcription_session_becomes_a_ga_transcription_client_secret_request() -> None:
    beta_session: Final[dict[str, JsonValue]] = {
        "input_audio_format": "pcm16",
        "input_audio_transcription": _TRANSCRIPTION,
        "turn_detection": _TURN_DETECTION,
        "input_audio_noise_reduction": _NOISE_REDUCTION,
        "include": _INCLUDE,
        "client_secret": {"expires_at": {"anchor": "created_at", "seconds": 600}},
    }

    assert ga_transcription_client_secret_request(beta_session) == {
        "expires_after": {"anchor": "created_at", "seconds": 600},
        "session": {
            "type": "transcription",
            "include": _INCLUDE,
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "transcription": _TRANSCRIPTION,
                    "turn_detection": _TURN_DETECTION,
                    "noise_reduction": _NOISE_REDUCTION,
                }
            },
        },
    }


def test_a_beta_session_without_optional_fields_sends_only_the_transcription() -> None:
    assert ga_transcription_client_secret_request({"input_audio_transcription": _TRANSCRIPTION}) == {
        "session": {"type": "transcription", "audio": {"input": {"transcription": _TRANSCRIPTION}}},
    }


def test_turn_detection_disabled_in_beta_stays_disabled_in_ga() -> None:
    request: Final = ga_transcription_client_secret_request(
        {"input_audio_transcription": _TRANSCRIPTION, "turn_detection": None}
    )

    assert request == {
        "session": {
            "type": "transcription",
            "audio": {"input": {"transcription": _TRANSCRIPTION, "turn_detection": None}},
        },
    }


@pytest.mark.parametrize(
    ("beta_format", "ga_format"),
    [
        ("pcm16", {"type": "audio/pcm", "rate": 24000}),
        ("g711_ulaw", {"type": "audio/pcmu"}),
        ("g711_alaw", {"type": "audio/pcma"}),
        ("opus", "opus"),
        ({"type": "audio/pcmu"}, {"type": "audio/pcmu"}),
    ],
    ids=["pcm16", "g711_ulaw", "g711_alaw", "unknown_name_passes_through", "ga_format_passes_through"],
)
def test_ga_audio_format(beta_format: JsonValue, ga_format: JsonValue) -> None:
    assert ga_audio_format(beta_format) == ga_format


def test_a_ga_transcription_client_secret_becomes_a_beta_transcription_session() -> None:
    ga_client_secret: Final[dict[str, JsonValue]] = {
        "value": "ek_upstream",
        "expires_at": 1792775579,
        "session": {
            "type": "transcription",
            "object": "realtime.transcription_session",
            "id": "sess_upstream",
            "expires_at": 0,
            "include": None,
            "audio": {
                "input": {
                    "format": {"type": "audio/pcmu"},
                    "transcription": _TRANSCRIPTION,
                    "noise_reduction": None,
                    "turn_detection": _TURN_DETECTION,
                }
            },
        },
    }

    assert beta_transcription_session(ga_client_secret) == {
        "id": "sess_upstream",
        "object": "realtime.transcription_session",
        "expires_at": 0,
        "include": None,
        "input_audio_format": "g711_ulaw",
        "input_audio_transcription": _TRANSCRIPTION,
        "input_audio_noise_reduction": None,
        "turn_detection": _TURN_DETECTION,
        "client_secret": {"value": "ek_upstream", "expires_at": 1792775579},
    }


@pytest.mark.parametrize("beta_format", _BETA_FORMATS)
def test_every_beta_audio_format_survives_the_round_trip_through_ga(beta_format: str) -> None:
    ga_client_secret: Final[dict[str, JsonValue]] = {
        "value": "ek_upstream",
        "expires_at": 1792775579,
        "session": {"audio": {"input": {"format": ga_audio_format(beta_format)}}},
    }

    assert beta_transcription_session(ga_client_secret)["input_audio_format"] == beta_format
