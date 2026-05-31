"""
Completar vídeos cuando la música YA existe pero faltan imágenes + ensamblaje.

Diseñado para terminar la tanda IRONPULSE/DEEPSTACK cuyo audio quedó
generado y masterizado, pero la fase de imágenes falló por un cache
corrupto de Juggernaut Lightning.

Para cada (tag, job_id, audio_path, style, duration, n_images):
  1. Reusa el audio ya existente (assets/music/music_<job_id>.mp3)
  2. Genera N imágenes con Juggernaut XL Lightning
  3. Ensambla vídeo con FFmpeg + Ken Burns + color grading

Logs: logs/complete_videos.log
"""
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from database.db import init_db, get_db
from database.models import VideoJob, VideoStatus, MusicStyle, VisualAsset


# (channel_tag, job_id, MusicStyle, duration_seconds, n_images)
JOBS = [
    ("ironpulse", 1, MusicStyle.INDUSTRIAL,  1800, 8),
    ("deepstack", 2, MusicStyle.DARK_TECHNO, 1800, 8),
]


settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
# NOTA: el FileHandler se omite a propósito. El .bat wrapper ya redirige
# stdout+stderr al log, y abrirlo también desde Python causa PermissionError
# en Windows (el archivo está bloqueado por el shell).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("complete_videos")


def section(title: str):
    print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}")


def _set_status(job_id: int, status: VideoStatus, error: str = None) -> None:
    with get_db() as db:
        job = db.query(VideoJob).filter(VideoJob.id == job_id).first()
        if job:
            job.status = status
            if error:
                job.error_message = error


async def complete_one(tag: str, job_id: int, style: MusicStyle, duration: int, n_images: int) -> dict:
    section(f"[{tag.upper()}] job={job_id}  style={style.value}  duration={duration}s")
    audio_path = settings.MUSIC_DIR / f"music_{job_id}.mp3"
    if not audio_path.exists():
        logger.error(f"[{tag}] audio no encontrado: {audio_path}")
        return {"tag": tag, "ok": False, "error": "audio missing"}

    # Idempotencia: si ya existe el vídeo final, saltar este job entero.
    existing_video = settings.VIDEOS_DIR / f"video_{job_id}.mp4"
    if existing_video.exists() and existing_video.stat().st_size > 1_000_000:
        logger.info(f"[{tag}] ⏭  vídeo ya existe ({existing_video}), saltando job")
        return {"tag": tag, "job_id": job_id, "audio_path": str(audio_path),
                "video_path": str(existing_video), "ok": True, "skipped": True, "error": None}

    logger.info(f"[{tag}] audio reutilizado: {audio_path}")
    t0 = time.time()
    result = {"tag": tag, "job_id": job_id, "audio_path": str(audio_path),
              "video_path": None, "ok": False, "error": None}

    # ── Imágenes ─────────────────────────────────────────────────────────────
    try:
        _set_status(job_id, VideoStatus.GENERATING_VISUALS)
        from visual_generator.generator import get_visual_generator
        gen_vis = get_visual_generator()
        logger.info(f"[{tag}] VRAM antes visuals: {gen_vis.vram_usage()}")
        t_v = time.time()
        asset_ids = await gen_vis.run_batch(job_id, style, n_images)
        with get_db() as db:
            assets = db.query(VisualAsset).filter(VisualAsset.id.in_(asset_ids)).all()
            image_paths = [Path(a.file_path) for a in assets if a.file_path]
        if not image_paths:
            raise RuntimeError("Visual phase produced no images")
        logger.info(f"[{tag}] ✅ {len(image_paths)} imágenes ({time.time()-t_v:.0f}s)")
        gen_vis.unload()
    except Exception as exc:
        logger.exception(f"[{tag}] ❌ fallo en imágenes: {exc}")
        result["error"] = f"visuals: {exc}"
        _set_status(job_id, VideoStatus.FAILED, str(exc))
        return result

    # ── Vídeo ────────────────────────────────────────────────────────────────
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
        result["video_path"] = str(video_path)
        _set_status(job_id, VideoStatus.PENDING)
        logger.info(f"[{tag}] ✅ vídeo OK ({time.time()-t_vid:.0f}s): {video_path}")
    except Exception as exc:
        logger.exception(f"[{tag}] ❌ fallo en vídeo: {exc}")
        result["error"] = f"video: {exc}"
        _set_status(job_id, VideoStatus.FAILED, str(exc))
        return result

    result["ok"] = True
    logger.info(f"[{tag}] 🎉 DONE en {(time.time()-t0)/60:.1f} min")
    return result


async def main():
    init_db()
    t_start = time.time()
    section("COMPLETE VIDEOS — reusando música existente")
    for tag, jid, style, dur, n in JOBS:
        print(f"  - {tag} (job {jid}): {style.value} {dur}s {n} imgs")

    results = []
    for tag, jid, style, dur, n in JOBS:
        results.append(await complete_one(tag, jid, style, dur, n))

    section(f"COMPLETADO — total {(time.time()-t_start)/60:.1f} min")
    for r in results:
        if r["ok"]:
            print(f"  ✅ {r['tag']:10s}  job={r['job_id']}")
            print(f"     audio: {r['audio_path']}")
            print(f"     video: {r['video_path']}")
        else:
            print(f"  ❌ {r['tag']:10s}  ERROR={r['error']}")


if __name__ == "__main__":
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"\nGPU: {gpu}\n")
    asyncio.run(main())
