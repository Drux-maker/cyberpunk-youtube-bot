"""
Central configuration — all secrets from environment variables, never hardcoded.
Todo el procesamiento de IA es local (MusicGen + Stable Diffusion).
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).parent.parent


class Settings(BaseSettings):
    # ─── Project ──────────────────────────────────────────────────────────────
    PROJECT_NAME: str = "CyberpunkYouTubeBot"
    ENVIRONMENT: str = Field(default="development", alias="ENV")
    DEBUG: bool = False

    # ─── Paths ────────────────────────────────────────────────────────────────
    ASSETS_DIR: Path = BASE_DIR / "assets"
    MUSIC_DIR: Path = BASE_DIR / "assets" / "music"
    VISUALS_DIR: Path = BASE_DIR / "assets" / "visuals"
    VIDEOS_DIR: Path = BASE_DIR / "assets" / "videos"
    THUMBNAILS_DIR: Path = BASE_DIR / "assets" / "thumbnails"
    FONTS_DIR: Path = BASE_DIR / "assets" / "fonts"
    LOGS_DIR: Path = BASE_DIR / "logs"

    # ─── Database ─────────────────────────────────────────────────────────────
    DATABASE_URL: str = Field(default="sqlite:///./cyberpunk_bot.db", alias="DATABASE_URL")
    REDIS_URL: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    # ─── MusicGen local ───────────────────────────────────────────────────────
    MUSICGEN_MODEL: str = "facebook/musicgen-stereo-medium"      # estéreo nativo, ~4.2GB VRAM en fp16
    MUSICGEN_FALLBACK_MODEL: str = "facebook/musicgen-medium"    # mono, fallback si OOM
    MUSICGEN_CLIP_DURATION: int = 30                              # segundos por clip
    MUSICGEN_CONTINUATION_SECONDS: float = 10.0                   # contexto pasado al modelo entre clips
    MUSICGEN_CFG_INTRO: float = 3.0                               # CFG más libre al arrancar
    MUSICGEN_CFG_BODY: float = 4.5                                # CFG más estricto en el cuerpo
    MUSICGEN_TEMPERATURE: float = 1.0
    MUSICGEN_TOP_K: int = 250
    MUSICGEN_TOP_P: float = 0.0
    MUSICGEN_NATIVE_SR: int = 32000                               # MusicGen genera a 32 kHz

    # ─── Stable Diffusion local ───────────────────────────────────────────────
    SD_MODEL: str = "stabilityai/stable-diffusion-xl-base-1.0"
    SD_FALLBACK_MODEL: str = "runwayml/stable-diffusion-v1-5"
    SD_STEPS: int = 25
    SD_GUIDANCE_SCALE: float = 7.5
    SD_USE_XFORMERS: bool = True           # requiere xformers instalado
    SD_HALF_PRECISION: bool = True         # fp16, obligatorio para 6GB VRAM

    # ─── OpenAI (solo para SEO — títulos, descripciones, tags) ───────────────
    OPENAI_API_KEY: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    OPENAI_MODEL: str = "gpt-4o-mini"

    # ─── YouTube ──────────────────────────────────────────────────────────────
    YOUTUBE_CLIENT_ID: Optional[str] = Field(default=None, alias="YOUTUBE_CLIENT_ID")
    YOUTUBE_CLIENT_SECRET: Optional[str] = Field(default=None, alias="YOUTUBE_CLIENT_SECRET")
    YOUTUBE_REDIRECT_URI: str = Field(default="http://localhost:8080/oauth2callback", alias="YOUTUBE_REDIRECT_URI")
    YOUTUBE_TOKEN_FILE: Path = BASE_DIR / "config" / "youtube_token.json"
    YOUTUBE_CHANNEL_NAME: str = Field(default="CyberpunkAI Music", alias="YOUTUBE_CHANNEL_NAME")
    YOUTUBE_DEFAULT_CATEGORY_ID: str = "10"  # Music
    YOUTUBE_DEFAULT_PRIVACY: str = "public"

    # ─── Scheduler ────────────────────────────────────────────────────────────
    VIDEOS_PER_DAY: int = Field(default=2, alias="VIDEOS_PER_DAY")
    MIN_QUALITY_SCORE: float = 0.65

    # ─── Video specs ──────────────────────────────────────────────────────────
    VIDEO_RESOLUTIONS: dict = {
        "1080p": (1920, 1080),
        "720p":  (1280, 720),
    }
    DEFAULT_RESOLUTION: str = "1080p"
    VIDEO_FPS: int = 30
    VIDEO_BITRATE: str = "8000k"
    AUDIO_BITRATE: str = "320k"
    AUDIO_AAC_BITRATE: str = "256k"                 # YouTube prefiere AAC
    AUDIO_SAMPLE_RATE: int = 48000                  # 48 kHz: estándar profesional de vídeo
    TARGET_LOUDNESS: float = -14.0                  # LUFS — estándar YouTube
    TARGET_TRUE_PEAK: float = -1.0                  # dBTP — margen para evitar clipping en transcoders
    EXPORT_AAC: bool = True                         # exportar también M4A/AAC además de MP3

    # ─── Music durations ──────────────────────────────────────────────────────
    MUSIC_DURATIONS: list = [600, 1800, 3600]  # 10min, 30min, 1h
    DEFAULT_MUSIC_DURATION: int = 3600
    BPM_RANGE: tuple = (120, 145)

    # ─── Celery ───────────────────────────────────────────────────────────────
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    CELERY_TASK_SERIALIZER: str = "json"
    CELERY_TASK_MAX_RETRIES: int = 3
    CELERY_TASK_RETRY_BACKOFF: int = 60

    # ─── FastAPI ──────────────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
