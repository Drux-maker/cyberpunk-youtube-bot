"""
Test completo del pipeline — inferencia 100% local.
No requiere ninguna API de pago. Solo FFmpeg y los modelos de HuggingFace.

OPENAI_API_KEY es opcional (para SEO). Sin ella usa plantillas locales.

Uso:
    python scripts/test_local.py --style cyberpunk --duration 60
    python scripts/test_local.py --style dark_techno --duration 300
    python scripts/test_local.py --style neon_ambient --duration 60 --no-seo
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("test_local")


def step(n: int, name: str):
    print(f"\n{'='*60}\n  PASO {n}: {name}\n{'='*60}")


async def run(style_name: str, duration: int, skip_seo: bool):
    init_db()
    style = MusicStyle(style_name)
    job_id = abs(hash(f"{style_name}{duration}{uuid.uuid4()}")) % 100000

    with get_db() as db:
        db.add(VideoJob(
            id=job_id,
            uuid=str(uuid.uuid4()),
            status=VideoStatus.PENDING,
            style=style,
            duration_seconds=duration,
        ))

    from music_generator.prompt_engine import build_music_prompt_variations
    from visual_generator.prompt_engine import build_visual_batch

    music_var  = build_music_prompt_variations(style, duration, count=1)[0]
    music_prompt = music_var["prompt"]
    bpm          = music_var["bpm"]
    vis_prompts  = build_visual_batch(style, count=4 if duration <= 120 else 8)

    print(f"\n🤖 Job {job_id} | {style_name} | {duration}s | {bpm} BPM")

    # ── Paso 1: Música ────────────────────────────────────────────────────────
    step(1, "MUSICGEN — Generando audio local")
    from music_generator.generator import get_music_generator
    gen_music = get_music_generator()
    logger.info(f"VRAM: {gen_music.vram_usage()}")
    asset_id = await gen_music.run(job_id, style, duration, music_prompt, bpm)
    with get_db() as db:
        from database.models import MusicAsset
        asset = db.query(MusicAsset).filter(MusicAsset.id == asset_id).first()
        audio_path = Path(asset.file_path)
    logger.info(f"Audio: {audio_path} ({audio_path.stat().st_size / 1e6:.1f}MB)")

    # Liberar VRAM antes de cargar SDXL (crítico en GPUs de 6GB)
    gen_music.unload()
    logger.info(f"VRAM tras descargar MusicGen: {gen_music.vram_usage()}")

    # ── Paso 2: Imágenes ──────────────────────────────────────────────────────
    step(2, "STABLE DIFFUSION — Generando imágenes locales")
    from visual_generator.generator import get_visual_generator
    gen_vis = get_visual_generator()
    logger.info(f"VRAM: {gen_vis.vram_usage()}")
    output_dir = settings.VISUALS_DIR / str(job_id)
    image_paths = gen_vis.generate_batch(vis_prompts, output_dir)
    if not image_paths:
        print("ERROR: No se generaron imágenes. Abortando.")
        return
    logger.info(f"Imágenes: {len(image_paths)}")

    # Liberar VRAM tras imágenes (ffmpeg/Pillow no la necesitan)
    gen_vis.unload()
    logger.info(f"VRAM tras descargar SDXL: {gen_vis.vram_usage()}")

    # ── Paso 3: Vídeo ─────────────────────────────────────────────────────────
    step(3, "VIDEO EDITOR — Ensamblando con FFmpeg")
    from video_editor.editor import VideoEditor
    video_path = await VideoEditor().assemble(
        job_id=job_id,
        image_paths=image_paths,
        audio_path=audio_path,
        duration_seconds=duration,
        apply_effects=True,
    )
    logger.info(f"Vídeo: {video_path} ({video_path.stat().st_size / 1e6:.1f}MB)")

    # ── Paso 4: Thumbnails ────────────────────────────────────────────────────
    step(4, "THUMBNAILS — Generando miniaturas")
    from thumbnail_generator.generator import ThumbnailGenerator
    gen_thumb = ThumbnailGenerator()
    variants  = gen_thumb.generate_variants(job_id, style, image_paths[0], duration)
    best_thumb = gen_thumb.select_best(job_id, variants)
    logger.info(f"Mejor thumbnail: {best_thumb}")

    # ── Paso 5: SEO ───────────────────────────────────────────────────────────
    seo_title = "N/A"
    if not skip_seo and settings.OPENAI_API_KEY:
        step(5, "SEO ENGINE — Generando metadatos con GPT-4o-mini")
        from seo_engine.engine import generate_seo_metadata
        await generate_seo_metadata(job_id, style, duration, bpm, music_prompt)
        with get_db() as db:
            job_obj = db.query(VideoJob).filter(VideoJob.id == job_id).first()
            seo_title = job_obj.seo_metadata.title if job_obj.seo_metadata else "N/A"
    else:
        step(5, "SEO ENGINE — Saltado (no-seo o sin OPENAI_API_KEY)")

    # ── Resultado ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  ✅ PIPELINE COMPLETADO")
    print(f"{'='*60}")
    print(f"  Vídeo:      {video_path}")
    print(f"  Thumbnail:  {best_thumb}")
    print(f"  Título SEO: {seo_title}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", default="cyberpunk",
                        choices=[s.value for s in MusicStyle])
    parser.add_argument("--duration", type=int, default=60,
                        help="Segundos de vídeo (60 para test rápido)")
    parser.add_argument("--no-seo", action="store_true",
                        help="Saltar SEO (si no tienes OPENAI_API_KEY en .env)")
    args = parser.parse_args()

    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"\nGPU: {gpu}")

    asyncio.run(run(args.style, args.duration, args.no_seo))
