"""
Central configuration — all secrets from environment variables, never hardcoded.
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

    # ─── AI Music APIs ────────────────────────────────────────────────────────
    SUNO_API_KEY: Optional[str] = Field(default=None, alias="SUNO_API_KEY")
    SUNO_API_BASE: str = "https://api.suno.ai/v1"
    UDIO_API_KEY: Optional[str] = Field(default=None, alias="UDIO_API_KEY")
    UDIO_API_BASE: str = "https://api.udio.com/v1"

    # ─── AI Visual APIs ───────────────────────────────────────────────────────
    STABILITY_API_KEY: Optional[str] = Field(default=None, alias="STABILITY_API_KEY")
    STABILITY_API_BASE: str = "https://api.stability.ai/v2beta"
    LEONARDO_API_KEY: Optional[str] = Field(default=None, alias="LEONARDO_API_KEY")
    REPLICATE_API_KEY: Optional[str] = Field(default=None, alias="REPLICATE_API_KEY")

    # ─── OpenAI (SEO + prompts) ───────────────────────────────────────────────
    OPENAI_API_KEY: str = Field(alias="OPENAI_API_KEY")
    OPENAI_MODEL: str = "gpt-4o-mini"

    # ─── YouTube ──────────────────────────────────────────────────────────────
    YOUTUBE_CLIENT_ID: str = Field(alias="YOUTUBE_CLIENT_ID")
    YOUTUBE_CLIENT_SECRET: str = Field(alias="YOUTUBE_CLIENT_SECRET")
    YOUTUBE_REDIRECT_URI: str = Field(default="http://localhost:8080/oauth2callback", alias="YOUTUBE_REDIRECT_URI")
    YOUTUBE_TOKEN_FILE: Path = BASE_DIR / "config" / "youtube_token.json"
    YOUTUBE_CHANNEL_NAME: str = Field(default="CyberpunkAI Music", alias="YOUTUBE_CHANNEL_NAME")
    YOUTUBE_DEFAULT_CATEGORY_ID: str = "10"  # Music
    YOUTUBE_DEFAULT_PRIVACY: str = "public"

    # ─── Scheduler ────────────────────────────────────────────────────────────
    VIDEOS_PER_DAY: int = Field(default=2, alias="VIDEOS_PER_DAY")
    PUBLISH_HOURS: list = [10, 18]  # UTC hours for publishing
    MIN_QUALITY_SCORE: float = 0.65  # Minimum score to publish

    # ─── Video specs ──────────────────────────────────────────────────────────
    VIDEO_RESOLUTIONS: dict = {
        "1080p": (1920, 1080),
        "4k": (3840, 2160),
    }
    DEFAULT_RESOLUTION: str = "1080p"
    VIDEO_FPS: int = 30
    VIDEO_BITRATE: str = "8000k"
    AUDIO_BITRATE: str = "320k"
    AUDIO_SAMPLE_RATE: int = 44100
    TARGET_LOUDNESS: float = -14.0  # LUFS (YouTube standard)

    # ─── Music generation ─────────────────────────────────────────────────────
    MUSIC_DURATIONS: list = [600, 1800, 3600]  # 10min, 30min, 1h in seconds
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
