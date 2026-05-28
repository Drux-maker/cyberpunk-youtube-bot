"""
SQLAlchemy ORM models — tracks every asset and video through the pipeline.
"""
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime,
    Text, JSON, Enum, ForeignKey, Index
)
from sqlalchemy.orm import relationship, DeclarativeBase
from sqlalchemy.sql import func
import enum


class Base(DeclarativeBase):
    pass


class VideoStatus(str, enum.Enum):
    PENDING = "pending"
    GENERATING_MUSIC = "generating_music"
    GENERATING_VISUALS = "generating_visuals"
    EDITING = "editing"
    GENERATING_THUMBNAIL = "generating_thumbnail"
    GENERATING_SEO = "generating_seo"
    UPLOADING = "uploading"
    PUBLISHED = "published"
    FAILED = "failed"
    REJECTED = "rejected"


class MusicStyle(str, enum.Enum):
    DARK_TECHNO = "dark_techno"
    CYBERPUNK = "cyberpunk"
    NEON_AMBIENT = "neon_ambient"
    INDUSTRIAL = "industrial"
    SYNTHWAVE = "synthwave"
    HARDTEK = "hardtek"
    ACID_TECHNO = "acid_techno"


class VideoJob(Base):
    __tablename__ = "video_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    uuid = Column(String(36), unique=True, nullable=False, index=True)
    status = Column(Enum(VideoStatus), default=VideoStatus.PENDING, index=True)
    style = Column(Enum(MusicStyle), nullable=False)
    duration_seconds = Column(Integer, nullable=False)
    quality_score = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)

    # Celery task IDs for tracking
    celery_task_id = Column(String(255), nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    published_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    music_asset = relationship("MusicAsset", back_populates="job", uselist=False)
    visual_assets = relationship("VisualAsset", back_populates="job")
    video_asset = relationship("VideoAsset", back_populates="job", uselist=False)
    thumbnail_assets = relationship("ThumbnailAsset", back_populates="job")
    seo_metadata = relationship("SEOMetadata", back_populates="job", uselist=False)
    youtube_video = relationship("YouTubeVideo", back_populates="job", uselist=False)

    __table_args__ = (
        Index("ix_video_jobs_status_created", "status", "created_at"),
    )


class MusicAsset(Base):
    __tablename__ = "music_assets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, ForeignKey("video_jobs.id"), nullable=False)
    provider = Column(String(50), nullable=False)  # suno, udio, musicgen
    prompt = Column(Text, nullable=False)
    file_path = Column(String(512), nullable=True)
    file_size_mb = Column(Float, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    bpm = Column(Float, nullable=True)
    loudness_lufs = Column(Float, nullable=True)
    sample_rate = Column(Integer, nullable=True)
    is_processed = Column(Boolean, default=False)
    generation_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    job = relationship("VideoJob", back_populates="music_asset")


class VisualAsset(Base):
    __tablename__ = "visual_assets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, ForeignKey("video_jobs.id"), nullable=False)
    provider = Column(String(50), nullable=False)  # stability, leonardo, replicate
    prompt = Column(Text, nullable=False)
    file_path = Column(String(512), nullable=True)
    asset_type = Column(String(50), default="image")  # image, video_clip, loop
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    is_animated = Column(Boolean, default=False)
    generation_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    job = relationship("VideoJob", back_populates="visual_assets")


class VideoAsset(Base):
    __tablename__ = "video_assets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, ForeignKey("video_jobs.id"), nullable=False)
    file_path = Column(String(512), nullable=True)
    resolution = Column(String(20), nullable=True)
    fps = Column(Integer, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    file_size_mb = Column(Float, nullable=True)
    effects_applied = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    job = relationship("VideoJob", back_populates="video_asset")


class ThumbnailAsset(Base):
    __tablename__ = "thumbnail_assets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, ForeignKey("video_jobs.id"), nullable=False)
    file_path = Column(String(512), nullable=True)
    variant_index = Column(Integer, default=0)  # 0-4, 5 variants generated
    is_selected = Column(Boolean, default=False)
    ctr_score = Column(Float, nullable=True)  # estimated CTR score
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    job = relationship("VideoJob", back_populates="thumbnail_assets")


class SEOMetadata(Base):
    __tablename__ = "seo_metadata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, ForeignKey("video_jobs.id"), nullable=False)
    title = Column(String(100), nullable=False)
    description = Column(Text, nullable=False)
    tags = Column(JSON, nullable=False)  # list of strings
    chapters = Column(JSON, nullable=True)  # list of {time, title}
    pinned_comment = Column(Text, nullable=True)
    playlist_names = Column(JSON, nullable=True)  # list of playlist names
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    job = relationship("VideoJob", back_populates="seo_metadata")


class YouTubeVideo(Base):
    __tablename__ = "youtube_videos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(Integer, ForeignKey("video_jobs.id"), nullable=False)
    youtube_id = Column(String(20), unique=True, nullable=True, index=True)
    youtube_url = Column(String(100), nullable=True)
    scheduled_at = Column(DateTime(timezone=True), nullable=True)
    published_at = Column(DateTime(timezone=True), nullable=True)
    privacy_status = Column(String(20), default="public")
    upload_status = Column(String(50), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Analytics (populated by analytics worker)
    views = Column(Integer, default=0)
    likes = Column(Integer, default=0)
    comments = Column(Integer, default=0)
    watch_time_hours = Column(Float, default=0.0)
    avg_view_duration_pct = Column(Float, default=0.0)
    ctr = Column(Float, default=0.0)
    impressions = Column(Integer, default=0)
    revenue_usd = Column(Float, default=0.0)
    analytics_updated_at = Column(DateTime(timezone=True), nullable=True)

    job = relationship("VideoJob", back_populates="youtube_video")


class PromptLibrary(Base):
    """Stores successful prompts to reuse and improve."""
    __tablename__ = "prompt_library"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prompt_type = Column(String(50), nullable=False)  # music, visual, title, description
    style = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    performance_score = Column(Float, default=0.0)
    usage_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_prompt_library_type_style", "prompt_type", "style"),
    )


class ChannelAnalytics(Base):
    """Daily channel-level analytics snapshot."""
    __tablename__ = "channel_analytics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime(timezone=True), nullable=False, unique=True)
    total_views = Column(Integer, default=0)
    total_watch_time_hours = Column(Float, default=0.0)
    subscriber_gain = Column(Integer, default=0)
    subscriber_loss = Column(Integer, default=0)
    net_subscribers = Column(Integer, default=0)
    total_revenue_usd = Column(Float, default=0.0)
    best_performing_style = Column(String(50), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
