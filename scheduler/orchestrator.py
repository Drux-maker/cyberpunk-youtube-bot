"""
Local pipeline orchestrator — replaces the previous Celery chain.

Runs the full video pipeline sequentially in a single Python process:

    music → visuals → video → thumbnail → SEO → upload

Each phase manages its own VRAM via .unload() so we fit the whole flow
on a 6 GB GPU. Errors at any phase mark the VideoJob as FAILED and
re-raise; callers (manual CLI or APScheduler) decide whether to retry.
"""
import asyncio
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config.settings import settings
from database.db import get_db
from database.models import (
    VideoJob, VideoStatus, MusicStyle, MusicAsset, VisualAsset, SEOMetadata,
)

logger = logging.getLogger(__name__)


def _set_status(job_id: int, status: VideoStatus, error: Optional[str] = None) -> None:
    with get_db() as db:
        job = db.query(VideoJob).filter(VideoJob.id == job_id).first()
        if job:
            job.status = status
            if error:
                job.error_message = error


def _create_job(style: MusicStyle, duration_seconds: int) -> int:
    with get_db() as db:
        job = VideoJob(
            uuid=str(uuid.uuid4()),
            status=VideoStatus.PENDING,
            style=style,
            duration_seconds=duration_seconds,
        )
        db.add(job)
        db.flush()
        return job.id


# ─── individual phases ───────────────────────────────────────────────────────

async def _phase_music(job_id: int, style: MusicStyle, duration: int) -> Path:
    _set_status(job_id, VideoStatus.GENERATING_MUSIC)
    from music_generator.generator import get_music_generator

    gen = get_music_generator()
    asset_id = await gen.run_with_retry(job_id, style, duration)
    gen.unload()

    with get_db() as db:
        asset = db.query(MusicAsset).filter(MusicAsset.id == asset_id).first()
        return Path(asset.file_path)


async def _phase_visuals(job_id: int, style: MusicStyle, duration: int) -> list[Path]:
    _set_status(job_id, VideoStatus.GENERATING_VISUALS)
    from visual_generator.generator import get_visual_generator

    n_images = 8 if duration <= 1800 else 12
    gen = get_visual_generator()
    asset_ids = await gen.run_batch(job_id, style, n_images)
    gen.unload()

    with get_db() as db:
        assets = db.query(VisualAsset).filter(VisualAsset.id.in_(asset_ids)).all()
        return [Path(a.file_path) for a in assets if a.file_path]


async def _phase_video(job_id: int, image_paths: list[Path], audio_path: Path, duration: int) -> Path:
    _set_status(job_id, VideoStatus.EDITING)
    from video_editor.editor import VideoEditor

    return await VideoEditor().assemble(
        job_id=job_id,
        image_paths=image_paths,
        audio_path=audio_path,
        duration_seconds=duration,
        apply_effects=True,
    )


def _phase_thumbnail(job_id: int, style: MusicStyle, base_img: Path, duration: int) -> Path:
    _set_status(job_id, VideoStatus.GENERATING_THUMBNAIL)
    from thumbnail_generator.generator import ThumbnailGenerator

    gen = ThumbnailGenerator()
    variants = gen.generate_variants(job_id, style, base_img, duration)
    return gen.select_best(job_id, variants)


async def _phase_seo(job_id: int, style: MusicStyle, duration: int, bpm: int, prompt: str) -> Optional[int]:
    _set_status(job_id, VideoStatus.GENERATING_SEO)
    from seo_engine.engine import generate_seo_metadata

    return await generate_seo_metadata(job_id, style, duration, bpm, prompt)


def _phase_upload(
    job_id: int,
    video_path: Path,
    thumbnail_path: Path,
    publish_at: Optional[datetime],
) -> Optional[str]:
    _set_status(job_id, VideoStatus.UPLOADING)
    from youtube_uploader.uploader import youtube_uploader

    with get_db() as db:
        job = db.query(VideoJob).filter(VideoJob.id == job_id).first()
        seo = job.seo_metadata

    video_id = youtube_uploader.full_publish_pipeline(
        job_id=job_id,
        video_path=video_path,
        thumbnail_path=thumbnail_path,
        seo=seo,
        publish_at=publish_at,
    )

    _set_status(job_id, VideoStatus.PUBLISHED)
    with get_db() as db:
        job = db.query(VideoJob).filter(VideoJob.id == job_id).first()
        job.published_at = datetime.now(timezone.utc)
    return video_id


# ─── public entrypoint ───────────────────────────────────────────────────────

async def run_video_pipeline(
    style: MusicStyle,
    duration_seconds: int,
    publish_at: Optional[datetime] = None,
    upload: bool = True,
) -> dict:
    """
    Full pipeline for ONE video. Sequential, no Celery.

    Returns a dict with paths, ids and (if uploaded) youtube_url.
    Raises on any phase failure after marking the job FAILED in DB.
    """
    job_id = _create_job(style, duration_seconds)
    logger.info(f"Pipeline start: job={job_id} style={style.value} duration={duration_seconds}s")

    try:
        # 1. Music
        audio_path = await _phase_music(job_id, style, duration_seconds)
        with get_db() as db:
            music = db.query(MusicAsset).filter(MusicAsset.job_id == job_id).first()
            bpm = int(music.bpm) if music and music.bpm else 130
            music_prompt = music.prompt if music else ""

        # 2. Visuals
        image_paths = await _phase_visuals(job_id, style, duration_seconds)
        if not image_paths:
            raise RuntimeError("Visual phase produced no images")

        # 3. Video
        video_path = await _phase_video(job_id, image_paths, audio_path, duration_seconds)

        # 4. Thumbnail
        thumbnail_path = _phase_thumbnail(job_id, style, image_paths[0], duration_seconds)

        # 5. SEO
        seo_id = await _phase_seo(job_id, style, duration_seconds, bpm, music_prompt)

        result = {
            "job_id": job_id,
            "audio_path": str(audio_path),
            "video_path": str(video_path),
            "thumbnail_path": str(thumbnail_path),
            "seo_id": seo_id,
        }

        # 6. Upload (optional — local dry-runs can skip)
        if upload:
            video_id = _phase_upload(job_id, video_path, thumbnail_path, publish_at)
            result["youtube_id"] = video_id
            result["youtube_url"] = f"https://youtu.be/{video_id}" if video_id else None

        logger.info(f"Pipeline OK: job={job_id} result={result}")
        return result

    except Exception as exc:
        logger.exception(f"Pipeline FAILED at job={job_id}: {exc}")
        _set_status(job_id, VideoStatus.FAILED, str(exc))
        raise


def run_video_pipeline_sync(
    style: MusicStyle,
    duration_seconds: int,
    publish_at: Optional[datetime] = None,
    upload: bool = True,
) -> dict:
    """Sync wrapper for callers that aren't already in an event loop."""
    return asyncio.run(run_video_pipeline(style, duration_seconds, publish_at, upload))
