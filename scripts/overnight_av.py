"""
Generación overnight — SOLO audio + vídeo (sin thumbnails ni SEO).
Pensado para dejarlo corriendo de noche y revisar/subir a la mañana siguiente.

Para cada (channel_tag, style, duration_seconds) hace:
  1. Música (MusicGen Stereo Medium + cadena de mastering pro)
  2. Imágenes (JuggernautXL Lightning, 12 imágenes a duraciones largas)
  3. Vídeo (FFmpeg + Ken Burns + color grading)

Salta thumbnails y SEO a propósito. Esos los generamos al día siguiente
en segundos cuando validemos los vídeos manualmente.

Logs por canal: logs/overnight_<tag>.log
Cada paso libera VRAM antes del siguiente (crítico en 6 GB).

Uso:
    python scripts/overnight_av.py
"""
import asyncio
import logging
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from database.db import init_db, get_db
from database.models import (
    VideoJob, VideoStatus, MusicStyle, MusicAsset, VisualAsset,
)


# ─── Configuración de la tanda ───────────────────────────────────────────────
# Duración bajada a 30 min para validar el lanzamiento robusto vía Task
# Scheduler antes de comprometer 6 h en pistas de 1 h. Una vez confirmado,
# subir a 3600 sin tocar nada más.
JOBS = [
    # (channel_tag, MusicStyle, duration_seconds, n_images)
    ("ironpulse", MusicStyle.INDUSTRIAL,  1800, 8),   # canal gym (30 min)
    ("deepstack", MusicStyle.DARK_TECHNO, 1800, 8),   # canal devs (30 min)
]


# ─── Logging global ──────────────────────────────────────────────────────────
settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
# El .bat wrapper redirige stdout+stderr al log; abrir el mismo archivo
# desde Python causa PermissionError en Windows. Solo StreamHandler.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("overnight_av")


def section(title: str):
    print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}")


def _create_job(style: MusicStyle, duration: int) -> int:
    with get_db() as db:
        job = VideoJob(
            uuid=str(uuid.uuid4()),
            status=VideoStatus.PENDING,
            style=style,
            duration_seconds=duration,
        )
        db.add(job)
        db.flush()
        return job.id


def _set_status(job_id: int, status: VideoStatus, error: str = None) -> None:
    with get_db() as db:
        job = db.query(VideoJob).filter(VideoJob.id == job_id).first()
        if job:
            job.status = status
            if error:
                job.error_message = error


async def run_one(channel_tag: str, style: MusicStyle, duration: int, n_images: int) -> dict:
    job_id = _create_job(style, duration)
    section(f"[{channel_tag.upper()}] job={job_id}  style={style.value}  duration={duration}s")
    t0 = time.time()
    result = {
        "channel": channel_tag, "job_id": job_id,
        "style": style.value, "duration_seconds": duration,
        "audio_path": None, "video_path": None,
        "music_seconds": None, "visuals_seconds": None, "video_seconds": None,
        "ok": False, "error": None,
    }

    # ── 1. Música ────────────────────────────────────────────────────────────
    try:
        _set_status(job_id, VideoStatus.GENERATING_MUSIC)
        from music_generator.generator import get_music_generator
        gen_music = get_music_generator()
        logger.info(f"[{channel_tag}] VRAM antes música: {gen_music.vram_usage()}")
        t_m = time.time()
        asset_id = await gen_music.run_with_retry(job_id, style, duration, channel_key=channel_tag)
        result["music_seconds"] = time.time() - t_m
        with get_db() as db:
            asset = db.query(MusicAsset).filter(MusicAsset.id == asset_id).first()
            audio_path = Path(asset.file_path)
        result["audio_path"] = str(audio_path)
        logger.info(f"[{channel_tag}] ✅ música OK ({result['music_seconds']:.0f}s): {audio_path}")
        gen_music.unload()
        logger.info(f"[{channel_tag}] VRAM tras unload música: {gen_music.vram_usage()}")
    except Exception as exc:
        logger.exception(f"[{channel_tag}] ❌ fallo en música: {exc}")
        result["error"] = f"music: {exc}"
        _set_status(job_id, VideoStatus.FAILED, str(exc))
        return result

    # ── 2. Imágenes ──────────────────────────────────────────────────────────
    try:
        _set_status(job_id, VideoStatus.GENERATING_VISUALS)
        from visual_generator.generator import get_visual_generator
        gen_vis = get_visual_generator()
        logger.info(f"[{channel_tag}] VRAM antes visuals: {gen_vis.vram_usage()}")
        t_v = time.time()
        asset_ids = await gen_vis.run_batch(job_id, style, n_images, channel_key=channel_tag)
        result["visuals_seconds"] = time.time() - t_v
        with get_db() as db:
            assets = db.query(VisualAsset).filter(VisualAsset.id.in_(asset_ids)).all()
            image_paths = [Path(a.file_path) for a in assets if a.file_path]
        if not image_paths:
            raise RuntimeError("No se generó ninguna imagen")
        logger.info(f"[{channel_tag}] ✅ visuals OK ({result['visuals_seconds']:.0f}s): {len(image_paths)} imágenes")
        gen_vis.unload()
        logger.info(f"[{channel_tag}] VRAM tras unload visuals: {gen_vis.vram_usage()}")
    except Exception as exc:
        logger.exception(f"[{channel_tag}] ❌ fallo en imágenes: {exc}")
        result["error"] = f"visuals: {exc}"
        _set_status(job_id, VideoStatus.FAILED, str(exc))
        return result

    # ── 3. Vídeo ─────────────────────────────────────────────────────────────
    try:
        _set_status(job_id, VideoStatus.EDITING)
        from video_editor.editor import VideoEditor
        t_vid = time.time()
        video_path = await VideoEditor().assemble(
            job_id=job_id,
            image_paths=image_paths,
            audio_path=audio_path,
            duration_seconds=duration,
            apply_effects=True,
        )
        result["video_seconds"] = time.time() - t_vid
        result["video_path"] = str(video_path)
        _set_status(job_id, VideoStatus.PENDING)   # marcamos pending de uploader
        logger.info(f"[{channel_tag}] ✅ vídeo OK ({result['video_seconds']:.0f}s): {video_path}")
    except Exception as exc:
        logger.exception(f"[{channel_tag}] ❌ fallo en vídeo: {exc}")
        result["error"] = f"video: {exc}"
        _set_status(job_id, VideoStatus.FAILED, str(exc))
        return result

    result["ok"] = True
    elapsed = time.time() - t0
    logger.info(f"[{channel_tag}] 🎉 DONE en {elapsed/60:.1f} min")
    return result


async def main():
    init_db()
    t_start = time.time()
    section("OVERNIGHT BATCH — audio + vídeo (sin thumbs / SEO)")
    print(f"Tanda: {len(JOBS)} videos")
    for tag, style, dur, n in JOBS:
        print(f"  - {tag}: {style.value} {dur}s ({n} imgs)")

    results = []
    for tag, style, dur, n in JOBS:
        res = await run_one(tag, style, dur, n)
        results.append(res)

    elapsed = time.time() - t_start
    section(f"BATCH COMPLETO — total {elapsed/3600:.1f}h")
    for r in results:
        if r["ok"]:
            print(f"  ✅ {r['channel']:10s}  job={r['job_id']}  music={r['music_seconds']:.0f}s  visuals={r['visuals_seconds']:.0f}s  video={r['video_seconds']:.0f}s")
            print(f"     audio: {r['audio_path']}")
            print(f"     video: {r['video_path']}")
        else:
            print(f"  ❌ {r['channel']:10s}  job={r['job_id']}  ERROR={r['error']}")


if __name__ == "__main__":
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"\nGPU: {gpu}\n")
    asyncio.run(main())
