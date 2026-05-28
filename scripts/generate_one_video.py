"""
Standalone script: generate + process a single video synchronously.
Use for testing the full pipeline without Celery.
Usage:
    python scripts/generate_one_video.py --style cyberpunk --duration 600
"""
import argparse
import asyncio
import logging
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from database.db import init_db, get_db
from database.models import VideoJob, VideoStatus, MusicStyle

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


async def run_pipeline(style_name: str, duration_seconds: int, upload: bool = False):
    init_db()

    # 1. Create job
    style = MusicStyle(style_name)
    with get_db() as db:
        job = VideoJob(
            uuid=str(uuid.uuid4()),
            status=VideoStatus.PENDING,
            style=style,
            duration_seconds=duration_seconds,
        )
        db.add(job)
        db.flush()
        job_id = job.id
    logger.info(f"Created job {job_id}")

    # 2. Generate music
    logger.info("=== Phase 1: Generating music ===")
    from music_generator import music_service
    from music_generator.audio_processor import AudioProcessor

    asset_id = await music_service.generate_with_retry(job_id, style, duration_seconds)
    with get_db() as db:
        from database.models import MusicAsset
        asset = db.query(MusicAsset).filter(MusicAsset.id == asset_id).first()
        raw_path = Path(asset.file_path)
        bpm = int(asset.bpm or 130)
        prompt = asset.prompt

    processed_audio = settings.MUSIC_DIR / f"music_final_{job_id}.mp3"
    await AudioProcessor.process(raw_path, processed_audio, duration_seconds)
    logger.info(f"Audio ready: {processed_audio}")

    # 3. Generate visuals
    logger.info("=== Phase 2: Generating visuals ===")
    from visual_generator import visual_service
    from database.models import VisualAsset

    n_images = 6 if duration_seconds <= 600 else 10
    asset_ids = await visual_service.generate_batch(job_id, style, n_images)
    with get_db() as db:
        assets = db.query(VisualAsset).filter(VisualAsset.id.in_(asset_ids)).all()
        image_paths = [Path(a.file_path) for a in assets]
    logger.info(f"Visuals ready: {len(image_paths)} images")

    # 4. Edit video
    logger.info("=== Phase 3: Editing video ===")
    from video_editor import video_editor

    video_path = await video_editor.assemble(
        job_id=job_id,
        image_paths=image_paths,
        audio_path=processed_audio,
        duration_seconds=duration_seconds,
        apply_effects=True,
    )
    logger.info(f"Video ready: {video_path}")

    # 5. Generate thumbnail
    logger.info("=== Phase 4: Generating thumbnails ===")
    from thumbnail_generator import thumbnail_generator

    variants = thumbnail_generator.generate_variants(job_id, style, image_paths[0], duration_seconds)
    best_thumb = thumbnail_generator.select_best(job_id, variants)
    logger.info(f"Best thumbnail: {best_thumb}")

    # 6. Generate SEO
    logger.info("=== Phase 5: Generating SEO ===")
    from seo_engine import generate_seo_metadata

    await generate_seo_metadata(job_id, style, duration_seconds, bpm, prompt)
    with get_db() as db:
        job_obj = db.query(VideoJob).filter(VideoJob.id == job_id).first()
        seo = job_obj.seo_metadata
        logger.info(f"Title: {seo.title}")

    # 7. Upload (optional)
    if upload:
        logger.info("=== Phase 6: Uploading to YouTube ===")
        from youtube_uploader import youtube_uploader
        video_id = youtube_uploader.full_publish_pipeline(
            job_id=job_id,
            video_path=video_path,
            thumbnail_path=best_thumb,
            seo=seo,
            publish_at=None,
        )
        logger.info(f"Published: https://youtu.be/{video_id}")
    else:
        logger.info(f"Skipping upload (pass --upload to enable)")
        logger.info(f"Video: {video_path}")
        logger.info(f"Thumbnail: {best_thumb}")

    logger.info("=== Pipeline complete ===")
    return {"job_id": job_id, "video_path": str(video_path), "thumbnail_path": str(best_thumb)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", default="cyberpunk",
                        choices=[s.value for s in MusicStyle],
                        help="Music/visual style")
    parser.add_argument("--duration", type=int, default=600,
                        choices=[600, 1800, 3600, 7200],
                        help="Duration in seconds")
    parser.add_argument("--upload", action="store_true",
                        help="Upload to YouTube after generating")
    args = parser.parse_args()

    result = asyncio.run(run_pipeline(args.style, args.duration, args.upload))
    print(f"\nDone: {result}")
