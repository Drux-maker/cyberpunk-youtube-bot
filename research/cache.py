"""
Caché en disco de perfiles de canal.

Un JSON por canal en assets/channel_profiles/<channel>.json. Se considera
caducado si supera CHANNEL_PROFILE_TTL_DAYS desde generated_at.
"""
import json
import logging
from pathlib import Path
from typing import Optional

from config.settings import settings
from .models import ChannelProfile

logger = logging.getLogger(__name__)


def _profile_path(channel_key: str) -> Path:
    return settings.CHANNEL_PROFILES_DIR / f"{channel_key}.json"


def load_profile(channel_key: str) -> Optional[ChannelProfile]:
    """Carga el perfil del disco. Devuelve None si no existe o el JSON está corrupto."""
    path = _profile_path(channel_key)
    if not path.exists():
        return None
    try:
        return ChannelProfile.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Perfil {channel_key} corrupto, ignorando: {exc}")
        return None


def save_profile(profile: ChannelProfile) -> Path:
    """Persiste el perfil. Crea el directorio si no existe."""
    settings.CHANNEL_PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    path = _profile_path(profile.channel)
    path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    logger.info(f"Perfil guardado: {path}")
    return path


def delete_profile(channel_key: str) -> bool:
    """Borra el perfil del disco. True si existía y se borró."""
    path = _profile_path(channel_key)
    if path.exists():
        path.unlink()
        return True
    return False
