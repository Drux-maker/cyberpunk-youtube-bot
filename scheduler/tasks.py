"""
Celery task definitions — each phase of the pipeline is an independent task
with automatic retries and status tracking.
"""
import asyncio
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from celery import Celery, chain, group
from celery.utils.log import get_task_logger

from config.settings import settings
from database.db import get_db
from database.models import VideoJob, VideoStatus, MusicStyle, MusicAsset, SEOMetadata

logger = get_task_logger(__name__)

app = Celery("cyberpunk_bot")
app.config_from_object({
    "broker_url": str(settings.CELERY_BROKER_URL),
    "result_backend": str(settings.CELERY_RESULT_BACKEND),
    "task_serializer": settings.CELERY_TASK_SERIALIZER,
    "result_serializer": "json",
    "accept_content": ["json"],
    "timezone": "UTC",
    "enable_utc": True,
    "task_acks_late": True,
    "worker_prefetch_multiplier": 1,
    "task_track_started": True,
})


def _set_status(job_id: int, status: VideoStatus, error: str = None) -> None:
    with get_db() as db:
        job = db.query(VideoJob).filter(VideoJob.id == job_id).first()
        if job:
            job.status = status
            if error:
                job.error_message = error


@app.task(
    bind=True,
    max_retries=settings.CELERY_TASK_MAX_RETRIES,
    default_retry_delay=settings.CELERY_TASK_RETRY_BACKOFF,
    name="tasks.create_job",
)
def create_job_task(self, style_name: str, duration_seconds: int) -> int:
    """Creates a VideoJob record and returns its ID."""
    style = MusicStyle(style_name)
    with get_db() as db:
        job = VideoJob(
            uuid=str(uuid.uuid4()),
            status=VideoStatus.PENDING,
            style=style,
            duration_seconds=duration_seconds,
            celery_task_id=self.request.id,
        )
        db.add(job)
        db.flush()
        job_id = job.id
    logger.info(f"Created VideoJob {job_id} ({style_name}, {duration_seconds}s)")
    return job_id


@app.task(
    bind=True,
    max_retries=settings.CELERY_TASK_MAX_RETRIES,
    default_retry_delay=settings.CELERY_TASK_RETRY_BACKOFF,
    name="tasks.generate_music",
)
def generate_music_task(self, job_id: int) -> dict:
    """Phase 1: Generate music and process audio."""
    _set_status(job_id, VideoStatus.GENERATING_MUSIC)
    try:
        with get_db() as db:
            job = db.query(VideoJob).filter(VideoJob.id == job_id).first()
            style = job.style
            duration = job.duration_seconds

        from music_generator import music_service
        asset_id = asyncio.run(music_service.generate_with_retry(job_id, style, duration))

        with get_db() as db:
            asset = db.query(MusicAsset).filter(MusicAsset.id == asset_id).first()
            raw_path = Path(asset.file_path)

        from music_generator.audio_processor import AudioProcessor
        processed_path = raw_path.parent / f"music_final_{job_id}.mp3"
        asyncio.run(AudioProcessor.process(raw_path, processed_path, duration))

        audio_info = AudioProcessor.get_audio_info(processed_path)

        with get_db() as db:
            asset = db.query(MusicAsset).filter(MusicAsset.id == asset_id).first()
            asset.file_path = str(processed_path)
            asset.is_processed = True
            asset.duration_seconds = audio_info["duration_seconds"]
            asset.loudness_lufs = settings.TARGET_LOUDNESS
            asset.file_size_mb = audio_info["file_size_mb"]

        logger.info(f"Music ready for job {job_id}: {processed_path}")
        return {"job_id": job_id, "audio_path": str(processed_path), "bpm": asset.bpm}

    except Exception as exc:
        logger.error(f"Music generation failed for job {job_id}: {exc}")
        _set_status(job_id, VideoStatus.FAILED, str(exc))
        raise self.retry(exc=exc)


@app.task(
    bind=True,
    max_retries=settings.CELERY_TASK_MAX_RETRIES,
    default_retry_delay=settings.CELERY_TASK_RETRY_BACKOFF,
    name="tasks.generate_visuals",
)
def generate_visuals_task(self, music_result: dict) -> dict:
    """Phase 2: Generate images and animate them."""
    job_id = music_result["job_id"]
    _set_status(job_id, VideoStatus.GENERATING_VISUALS)

    try:
        with get_db() as db:
            job = db.query(VideoJob).filter(VideoJob.id == job_id).first()
            style = job.style
            duration = job.duration_seconds

        from visual_generator import visual_service, Animator
        from database.models import VisualAsset

        # Generate base images (more for longer videos)
        n_images = 8 if duration <= 1800 else 12
        asset_ids = asyncio.run(visual_service.generate_batch(job_id, style, n_images))

        with get_db() as db:
            assets = db.query(VisualAsset).filter(VisualAsset.id.in_(asset_ids)).all()
            image_paths = [Path(a.file_path) for a in assets if a.file_path]

        logger.info(f"Visuals generated for job {job_id}: {len(image_paths)} images")
        return {**music_result, "image_paths": [str(p) for p in image_paths]}

    except Exception as exc:
        logger.error(f"Visual generation failed for job {job_id}: {exc}")
        _set_status(job_id, VideoStatus.FAILED, str(exc))
        raise self.retry(exc=exc)


@app.task(
    bind=True,
    max_retries=settings.CELERY_TASK_MAX_RETRIES,
    default_retry_delay=settings.CELERY_TASK_RETRY_BACKOFF,
    name="tasks.edit_video",
)
def edit_video_task(self, prev_result: dict) -> dict:
    """Phase 3: Edit final video."""
    job_id = prev_result["job_id"]
    _set_status(job_id, VideoStatus.EDITING)

    try:
        with get_db() as db:
            job = db.query(VideoJob).filter(VideoJob.id == job_id).first()
            duration = job.duration_seconds

        from video_editor import video_editor

        image_paths = [Path(p) for p in prev_result["image_paths"]]
        audio_path = Path(prev_result["audio_path"])

        video_path = asyncio.run(video_editor.assemble(
            job_id=job_id,
            image_paths=image_paths,
            audio_path=audio_path,
            duration_seconds=duration,
            apply_effects=True,
        ))

        logger.info(f"Video assembled for job {job_id}: {video_path}")
        return {**prev_result, "video_path": str(video_path)}

    except Exception as exc:
        logger.error(f"Video editing failed for job {job_id}: {exc}")
        _set_status(job_id, VideoStatus.FAILED, str(exc))
        raise self.retry(exc=exc)


@app.task(
    bind=True,
    max_retries=settings.CELERY_TASK_MAX_RETRIES,
    default_retry_delay=60,
    name="tasks.generate_thumbnail",
)
def generate_thumbnail_task(self, prev_result: dict) -> dict:
    """Phase 4: Generate thumbnails."""
    job_id = prev_result["job_id"]
    _set_status(job_id, VideoStatus.GENERATING_THUMBNAIL)

    try:
        with get_db() as db:
            job = db.query(VideoJob).filter(VideoJob.id == job_id).first()
            style = job.style
            duration = job.duration_seconds

        from thumbnail_generator import thumbnail_generator

        # Use first image as base for thumbnails
        base_img = Path(prev_result["image_paths"][0])
        variants = thumbnail_generator.generate_variants(job_id, style, base_img, duration)
        best_thumb = thumbnail_generator.select_best(job_id, variants)

        return {**prev_result, "thumbnail_path": str(best_thumb)}

    except Exception as exc:
        logger.error(f"Thumbnail generation failed for job {job_id}: {exc}")
        _set_status(job_id, VideoStatus.FAILED, str(exc))
        raise self.retry(exc=exc)


@app.task(
    bind=True,
    max_retries=settings.CELERY_TASK_MAX_RETRIES,
    default_retry_delay=30,
    name="tasks.generate_seo",
)
def generate_seo_task(self, prev_result: dict) -> dict:
    """Phase 5: Generate SEO metadata."""
    job_id = prev_result["job_id"]
    _set_status(job_id, VideoStatus.GENERATING_SEO)

    try:
        with get_db() as db:
            job = db.query(VideoJob).filter(VideoJob.id == job_id).first()
            music = job.music_asset
            style = job.style
            duration = job.duration_seconds
            prompt = music.prompt if music else ""
            bpm = int(music.bpm) if music and music.bpm else 130

        from seo_engine import generate_seo_metadata

        seo_id = asyncio.run(generate_seo_metadata(job_id, style, duration, bpm, prompt))
        return {**prev_result, "seo_id": seo_id}

    except Exception as exc:
        logger.error(f"SEO generation failed for job {job_id}: {exc}")
        _set_status(job_id, VideoStatus.FAILED, str(exc))
        raise self.retry(exc=exc)


@app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=120,
    name="tasks.upload_video",
)
def upload_video_task(self, prev_result: dict, publish_at_iso: str = None) -> dict:
    """Phase 6: Upload to YouTube."""
    job_id = prev_result["job_id"]
    _set_status(job_id, VideoStatus.UPLOADING)

    try:
        with get_db() as db:
            job = db.query(VideoJob).filter(VideoJob.id == job_id).first()
            seo = job.seo_metadata

        publish_at = datetime.fromisoformat(publish_at_iso) if publish_at_iso else None

        from youtube_uploader import youtube_uploader

        video_id = youtube_uploader.full_publish_pipeline(
            job_id=job_id,
            video_path=Path(prev_result["video_path"]),
            thumbnail_path=Path(prev_result["thumbnail_path"]),
            seo=seo,
            publish_at=publish_at,
        )

        _set_status(job_id, VideoStatus.PUBLISHED)

        with get_db() as db:
            job = db.query(VideoJob).filter(VideoJob.id == job_id).first()
            job.published_at = datetime.now(timezone.utc)

        logger.info(f"Job {job_id} published: https://youtu.be/{video_id}")
        return {**prev_result, "youtube_id": video_id, "youtube_url": f"https://youtu.be/{video_id}"}

    except Exception as exc:
        logger.error(f"Upload failed for job {job_id}: {exc}")
        _set_status(job_id, VideoStatus.FAILED, str(exc))
        raise self.retry(exc=exc)


def build_pipeline(style_name: str, duration_seconds: int, publish_at_iso: str = None) -> str:
    """
    Chains all tasks together. Returns the Celery chain task ID.
    Usage: build_pipeline("cyberpunk", 3600)
    """
    job_id_result = create_job_task.delay(style_name, duration_seconds)
    job_id = job_id_result.get(timeout=30)

    pipeline = chain(
        generate_music_task.s(job_id),
        generate_visuals_task.s(),
        edit_video_task.s(),
        generate_thumbnail_task.s(),
        generate_seo_task.s(),
        upload_video_task.s(publish_at_iso=publish_at_iso),
    )
    result = pipeline.apply_async()
    logger.info(f"Pipeline started for job {job_id}, chain task: {result.id}")
    return result.id
