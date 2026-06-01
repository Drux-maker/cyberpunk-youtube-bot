"""
Cliente Anthropic — singleton reutilizable + carga lazy del SDK.

Sigue la regla del skill claude-api: cliente único por proceso, prompt caching
en el system prompt, modelo Claude Opus 4.7 como default (no se downgrade
sin permiso explícito del usuario).
"""
import logging
from typing import Optional

from config.settings import settings

logger = logging.getLogger(__name__)

# Modelo por defecto: Opus 4.7 (más capaz para research investigativo).
# Si el usuario quiere ahorrar, puede pasar model=settings.RESEARCH_FALLBACK_MODEL
# en las llamadas (por ejemplo claude-haiku-4-5 a 1/5 del coste).
DEFAULT_MODEL = "claude-opus-4-7"

_client = None


def get_client():
    """Devuelve un cliente Anthropic singleton. Carga el SDK de forma lazy."""
    global _client
    if _client is not None:
        return _client
    if not settings.ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY no está en .env — la capa research no puede funcionar. "
            "Configúrala o pasa --no-research al script."
        )
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RuntimeError(
            "Paquete anthropic no instalado. pip install anthropic"
        ) from exc

    _client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    logger.info("Cliente Anthropic inicializado (singleton)")
    return _client


def reset_client() -> None:
    """Útil para tests; fuerza recarga del cliente en la siguiente llamada."""
    global _client
    _client = None
