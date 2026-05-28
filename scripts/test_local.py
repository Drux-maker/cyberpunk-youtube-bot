"""
Test completo del pipeline usando SOLO recursos locales.
No necesita API keys de Suno, Udio, Stability ni Leonardo.
Solo necesita: OPENAI_API_KEY (para el SEO) y ffmpeg instalado.

Uso:
    python scripts/test_local.py --style cyberpunk --duration 60
    python scripts/test_local.py --style dark_techno --duration 300 --no-seo
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


def print_step(n: int, name: str):
    print(f"\n{'='*60}")
    print(f"  PASO {n}: {name}")
    print(f"{'='*60}")


async def test_musicgen(prompt: str, duration: int, job_id: int) -> Path:
    """Genera audio con MusicGen local."""
    print_step(1, "MUSICGEN — Generando audio local")
    from music_generator.local_generator import get_musicgen_client
    from music_generator.audio_processor import AudioProcessor

    client = get_musicgen_client()
    logger.info(f"VRAM antes de MusicGen: {client.get_vram_usage()}")

    raw_path = settings.MUSIC_DIR / f"raw_{job_id}.wav"

    if duration <= 30:
        # Clip único para pruebas rápidas
        client.generate_clip(prompt, duration_seconds=duration, output_path=raw_path)
    else:
        # Multi-clip con crossfade
        client.generate_long(
            prompt=prompt,
            total_duration_seconds=duration,
            output_path=raw_path,
            clip_duration=30,
        )

    # Procesar audio: normalizar loudness + fade in/out + exportar MP3
    processed = settings.MUSIC_DIR / f"music_{job_id}.mp3"
    await AudioProcessor.process(raw_path, processed, duration)
    raw_path.unlink(missing_ok=True)

    logger.info(f"VRAM después de MusicGen: {client.get_vram_usage()}")
    logger.info(f"Audio final: {processed} ({processed.stat().st_size / 1024:.0f}KB)")
    return processed


def test_stable_diffusion(prompts: list[dict], job_id: int) -> list[Path]:
    """Genera imágenes con Stable Diffusion local."""
    print_step(2, "STABLE DIFFUSION — Generando imágenes locales")
    from visual_generator.local_generator import get_sd_client

    client = get_sd_client(prefer_sdxl=True)
    logger.info(f"VRAM antes de SD: {client.get_vram_usage()}")

    output_dir = settings.VISUALS_DIR / str(job_id)
    paths = client.generate_batch(prompts, output_dir=output_dir)

    logger.info(f"VRAM después de SD: {client.get_vram_usage()}")
    logger.info(f"Imágenes generadas: {len(paths)}")
    return paths


async def test_video_editor(image_paths: list[Path], audio_path: Path,
                            duration: int, job_id: int) -> Path:
    """Ensambla el vídeo final con FFmpeg."""
    print_step(3, "VIDEO EDITOR — Ensamblando vídeo")
    from video_editor.editor import VideoEditor

    editor = VideoEditor()
    video_path = await editor.assemble(
        job_id=job_id,
        image_paths=image_paths,
        audio_path=audio_path,
        duration_seconds=duration,
        apply_effects=True,
    )
    size_mb = video_path.stat().st_size / (1024 * 1024)
    logger.info(f"Vídeo: {video_path} ({size_mb:.1f}MB)")
    return video_path


def test_thumbnails(image_paths: list[Path], style: MusicStyle,
                   duration: int, job_id: int) -> Path:
    """Genera 5 thumbnails y selecciona el mejor."""
    print_step(4, "THUMBNAILS — Generando miniaturas")
    from thumbnail_generator.generator import ThumbnailGenerator

    gen = ThumbnailGenerator()
    variants = gen.generate_variants(job_id, style, image_paths[0], duration)
    best = gen.select_best(job_id, variants)
    logger.info(f"Mejor thumbnail: {best}")
    return best


async def test_seo(style: MusicStyle, duration: int, bpm: int,
                  prompt: str, job_id: int) -> dict:
    """Genera SEO con OpenAI (requiere OPENAI_API_KEY)."""
    print_step(5, "SEO ENGINE — Generando metadatos")
    try:
        from seo_engine.engine import generate_seo_metadata
        await generate_seo_metadata(job_id, style, duration, bpm, prompt)

        with get_db() as db:
            job = db.query(VideoJob).filter(VideoJob.id == job_id).first()
            seo = job.seo_metadata
            return {"title": seo.title, "tags_count": len(seo.tags)}
    except Exception as e:
        logger.warning(f"SEO falló (¿falta OPENAI_API_KEY?): {e}")
        return {"title": f"[TEST] Cyberpunk Mix — {duration}s", "tags_count": 0}


async def run(style_name: str, duration: int, skip_seo: bool = False):
    """Pipeline completo local."""
    init_db()
    style = MusicStyle(style_name)
    job_id = abs(hash(f"{style_name}{duration}")) % 100000  # ID de prueba

    # Crear job en DB
    with get_db() as db:
        existing = db.query(VideoJob).filter(VideoJob.id == job_id).first()
        if not existing:
            db.add(VideoJob(
                id=job_id,
                uuid=str(uuid.uuid4()),
                status=VideoStatus.PENDING,
                style=style,
                duration_seconds=duration,
            ))

    # Importar prompt engines
    from music_generator.prompt_engine import build_music_prompt_variations
    from visual_generator.prompt_engine import build_visual_batch

    music_variations = build_music_prompt_variations(style, duration, count=1)
    music_prompt = music_variations[0]["prompt"]
    bpm = music_variations[0]["bpm"]

    visual_prompts = build_visual_batch(style, count=4 if duration <= 120 else 8)

    logger.info(f"Style: {style_name} | Duration: {duration}s | BPM: {bpm}")
    logger.info(f"Music prompt: {music_prompt[:100]}...")

    # ─── Ejecutar pipeline ────────────────────────────────────────────────────
    audio_path  = await test_musicgen(music_prompt, duration, job_id)
    image_paths = test_stable_diffusion(visual_prompts, job_id)

    if not image_paths:
        logger.error("No se generaron imágenes. Abortando.")
        return

    video_path  = await test_video_editor(image_paths, audio_path, duration, job_id)
    thumb_path  = test_thumbnails(image_paths, style, duration, job_id)
    seo_data    = {} if skip_seo else await test_seo(style, duration, bpm, music_prompt, job_id)

    # ─── Resultado ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  PIPELINE COMPLETADO ✅")
    print(f"{'='*60}")
    print(f"  Vídeo:      {video_path}")
    print(f"  Thumbnail:  {thumb_path}")
    print(f"  Título SEO: {seo_data.get('title', 'N/A')}")
    print(f"  Tags:       {seo_data.get('tags_count', 0)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test pipeline local — sin APIs de pago")
    parser.add_argument("--style", default="cyberpunk",
                        choices=[s.value for s in MusicStyle])
    parser.add_argument("--duration", type=int, default=60,
                        help="Duración en segundos (usa 60 para test rápido)")
    parser.add_argument("--no-seo", action="store_true",
                        help="Saltar generación SEO (si no tienes OPENAI_API_KEY)")
    args = parser.parse_args()

    print(f"\n🤖 CyberpunkBot — Test Local")
    print(f"   Style: {args.style} | Duración: {args.duration}s")
    print(f"   GPU: {'CUDA' if __import__('torch').cuda.is_available() else 'CPU'}\n")

    asyncio.run(run(args.style, args.duration, skip_seo=args.no_seo))
