"""
profile_channel — orchestrator de la capa research.

Llama a Claude Opus 4.7 con prompt caching activado en el system, le pasa la
definición del canal del config.channels y persiste el resultado en disco.

Anti-patrón evitado: nunca llamamos al LLM si hay un perfil válido en caché.
"""
import json
import logging
from datetime import datetime, timezone
from typing import Optional

from config.channels import get_channel
from .cache import load_profile, save_profile
from .client import get_client, DEFAULT_MODEL
from .models import ChannelProfile
from .prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def _build_user_message(channel_key: str, channel_def: dict) -> str:
    """User-turn message con la identidad del canal a perfilar."""
    style_names = [s.value for s in channel_def["music_styles"]]
    return (
        f"Profile this YouTube channel.\n\n"
        f"channel_key: {channel_key}\n"
        f"display_name: {channel_def['display_name']}\n"
        f"tagline: {channel_def['tagline']}\n"
        f"audience_target: {channel_def['audience_target']}\n"
        f"use_cases: {', '.join(channel_def['use_cases'])}\n"
        f"music_styles_catalog: {', '.join(style_names)}\n"
        f"music_energy: {channel_def['music_energy']}\n"
        f"visual_aesthetic_base: {channel_def['visual_aesthetic']}\n\n"
        f"Return the JSON object now, following the supplied schema and "
        f"the constraints in your system prompt. Remember: the music_styles_catalog "
        f"above is the SUPERSET; the profile you return must work for ANY of those "
        f"styles when rendered for this channel's tribe."
    )


def profile_channel(
    channel_key: str,
    *,
    force_refresh: bool = False,
    model: str = DEFAULT_MODEL,
) -> ChannelProfile:
    """
    Devuelve un ChannelProfile.

    Si hay perfil en caché y no ha caducado y no se pide force_refresh, lo
    reutiliza sin llamar al LLM. En caso contrario, llama a Claude.

    `model` debería ser claude-opus-4-7 (default) para máxima calidad. Para
    ahorrar coste se puede pasar claude-haiku-4-5.
    """
    # Caché hit
    if not force_refresh:
        cached = load_profile(channel_key)
        if cached is not None and not cached.is_expired():
            logger.info(f"Perfil {channel_key} cacheado (válido), reutilizando")
            return cached
        if cached is not None:
            logger.info(f"Perfil {channel_key} caducado, regenerando")

    channel_def = get_channel(channel_key)
    user_message = _build_user_message(channel_key, channel_def)
    client = get_client()

    logger.info(f"Llamando a Anthropic [{model}] para profile_channel({channel_key})")

    response = client.messages.parse(
        model=model,
        max_tokens=4096,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                # Prompt caching: los próximos canales perfilados en el mismo
                # batch sólo pagan el cache read (~10% del precio normal)
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_message}],
        output_format=ChannelProfile,
    )

    profile = response.parsed_output
    # Sobrescribir campos generados por el modelo con metadatos canónicos
    profile.channel = channel_key
    profile.generated_at = datetime.now(timezone.utc)
    profile.model = model
    profile.tokens_used = (
        response.usage.input_tokens
        + response.usage.output_tokens
        + (response.usage.cache_creation_input_tokens or 0)
        + (response.usage.cache_read_input_tokens or 0)
    )

    save_profile(profile)
    logger.info(
        f"Perfil {channel_key} generado | "
        f"tokens={profile.tokens_used} "
        f"(cache_read={response.usage.cache_read_input_tokens or 0})"
    )
    return profile


def get_channel_profile(channel_key: str) -> Optional[ChannelProfile]:
    """
    Devuelve el perfil cacheado si existe y NO está caducado, sin llamar al LLM.
    Útil para los prompt engines en runtime: si no hay perfil, se cae al
    template hardcoded sin crashear.
    """
    cached = load_profile(channel_key)
    if cached is None or cached.is_expired():
        return None
    return cached
