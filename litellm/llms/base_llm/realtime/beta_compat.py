from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import JsonValue

from litellm.llms.base_llm.realtime.transcription_protocol import json_mapping

_PCM_SAMPLE_RATE: Final = 24000
_GA_AUDIO_FORMAT_TYPES: Final = MappingProxyType(
    {"pcm16": "audio/pcm", "g711_ulaw": "audio/pcmu", "g711_alaw": "audio/pcma"}
)
_BETA_AUDIO_FORMATS: Final = MappingProxyType({ga: beta for beta, ga in _GA_AUDIO_FORMAT_TYPES.items()})
_GA_AUDIO_INPUT_FIELDS: Final = MappingProxyType(
    {
        "input_audio_transcription": "transcription",
        "turn_detection": "turn_detection",
        "input_audio_noise_reduction": "noise_reduction",
    }
)
_BETA_ONLY_FIELDS: Final = frozenset((*_GA_AUDIO_INPUT_FIELDS, "input_audio_format", "client_secret"))
_GA_ONLY_SESSION_FIELDS: Final = frozenset(("type", "audio"))


def ga_audio_format(beta_format: JsonValue) -> JsonValue:
    if not isinstance(beta_format, str):
        return beta_format
    match _GA_AUDIO_FORMAT_TYPES.get(beta_format):
        case None:
            return beta_format
        case "audio/pcm":
            return {"type": "audio/pcm", "rate": _PCM_SAMPLE_RATE}
        case format_type:
            return {"type": format_type}


def ga_transcription_client_secret_request(beta_session: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    request: Final[dict[str, JsonValue]] = {"session": _ga_transcription_session(beta_session)}
    expires_after: Final = json_mapping(beta_session.get("client_secret"), "client_secret").get("expires_at")
    return request if expires_after is None else {**request, "expires_after": expires_after}


def beta_transcription_session(ga_client_secret: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    session: Final = json_mapping(ga_client_secret.get("session"), "session")
    audio: Final = json_mapping(session.get("audio"), "session.audio")
    return {
        **{field: value for field, value in session.items() if field not in _GA_ONLY_SESSION_FIELDS},
        **_beta_audio_input(json_mapping(audio.get("input"), "session.audio.input")),
        "client_secret": {"value": ga_client_secret.get("value"), "expires_at": ga_client_secret.get("expires_at")},
    }


def _ga_transcription_session(beta_session: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    passthrough: Final = {field: value for field, value in beta_session.items() if field not in _BETA_ONLY_FIELDS}
    return {**passthrough, "type": "transcription", "audio": {"input": _ga_audio_input(beta_session)}}


def _ga_audio_input(beta_session: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    renamed: Final = {ga: beta_session[beta] for beta, ga in _GA_AUDIO_INPUT_FIELDS.items() if beta in beta_session}
    if "input_audio_format" not in beta_session:
        return renamed
    return {**renamed, "format": ga_audio_format(beta_session["input_audio_format"])}


def _beta_audio_input(audio_input: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    renamed: Final = {beta: audio_input[ga] for beta, ga in _GA_AUDIO_INPUT_FIELDS.items() if ga in audio_input}
    if "format" not in audio_input:
        return renamed
    return {**renamed, "input_audio_format": _beta_audio_format(audio_input["format"])}


def _beta_audio_format(ga_format: JsonValue) -> JsonValue:
    format_type: Final = ga_format.get("type") if isinstance(ga_format, dict) else None
    if not isinstance(format_type, str):
        return ga_format
    return _BETA_AUDIO_FORMATS.get(format_type, ga_format)
