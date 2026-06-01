"""
Definiciones declarativas de los canales de YouTube que gestiona el bot.

Cada canal es UN nicho con su propia identidad de marca, audiencia y rotación
de estilos musicales. La capa `research/` consume estas definiciones para
pedirle al LLM perfiles detallados de cada uno (artistas referencia, labels,
estética visual, etc.) y los inyecta en los prompt engines de música e imagen.
"""
from database.models import MusicStyle

# Cada entrada describe el "ADN" del canal — lo que NUNCA cambia entre vídeos.
# Lo que sí cambia con el research del LLM son los descriptores finos.
CHANNELS: dict[str, dict] = {
    "deepstack": {
        "display_name": "DEEPSTACK",
        "tagline": "Music for shipping code",
        "audience_target": (
            "developers, data engineers, ML researchers, late-night coders, "
            "students of CS, deep-work focused knowledge workers"
        ),
        "use_cases": ["coding", "deep focus", "writing technical docs", "studying"],
        "music_styles": [
            MusicStyle.DARK_TECHNO,
            MusicStyle.NEON_AMBIENT,
            MusicStyle.INDUSTRIAL,
        ],
        "music_energy": "controlled, hypnotic, low-distraction, build slowly",
        "visual_aesthetic": (
            "dark interiors, monitors at night, terminal green/cyan, brutalist "
            "geometry, minimal humans, neon edge-lighting on architecture"
        ),
        "voice_tone": "minimal, technical, no hype, sin emojis salvo ⚙️/▒",
        "upload_hours_utc": [7, 14],
        "playlist_pattern": "Deep Stack Vol. {n}",
    },
    "ironpulse": {
        "display_name": "IRONPULSE",
        "tagline": "Beats heavier than your PR",
        "audience_target": (
            "powerlifting, strongman, MMA, boxing, crossfit, HYROX, "
            "gym men 18-40, hardcore training community"
        ),
        "use_cases": ["heavy lifting", "MMA training", "powerlifting session", "HYROX prep"],
        "music_styles": [
            MusicStyle.INDUSTRIAL,
            MusicStyle.HARDTEK,
            MusicStyle.ACID_TECHNO,
        ],
        "music_energy": "aggressive, peak-time, BPM alto, sin pausas largas",
        "visual_aesthetic": (
            "underground gyms, low-key red and amber lighting, chalk dust, "
            "barbells and plates, brutalist concrete, sweat-on-skin macro shots"
        ),
        "voice_tone": "agresivo, directo, motivacional sin caer en cringe, CAPS estratégicos",
        "upload_hours_utc": [5, 16],
        "playlist_pattern": "Iron Set Vol. {n}",
    },
}


def get_channel(channel_key: str) -> dict:
    """Devuelve la config de un canal o lanza KeyError con mensaje útil."""
    if channel_key not in CHANNELS:
        valid = ", ".join(CHANNELS.keys())
        raise KeyError(f"Canal '{channel_key}' no existe. Válidos: {valid}")
    return CHANNELS[channel_key]


def list_channels() -> list[str]:
    return list(CHANNELS.keys())
