"""
Teaser de audio — sólo música, sin imágenes ni vídeo. Validación rápida del
sonido tras los upgrades (MBD + mastering pro + prompts pro).

Genera UN clip de 30 segundos y aplica la cadena de mastering completa,
exporta a MP3 320k + opcionalmente AAC 256k. Sirve para escuchar y comparar
rápido antes de gastar 10+ minutos en un vídeo entero.

Uso:
    python scripts/audio_teaser.py --style cyberpunk
    python scripts/audio_teaser.py --style dark_techno
    python scripts/audio_teaser.py --style cyberpunk --no-mbd  # comparar vs sin MBD
"""
import argparse
import asyncio
import logging
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from database.models import MusicStyle


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("audio_teaser")


async def run(style_name: str, no_mbd: bool) -> None:
    style = MusicStyle(style_name)
    teaser_id = uuid.uuid4().hex[:8]

    if no_mbd:
        settings.MUSICGEN_USE_MBD = False  # override solo para esta ejecución
        logger.warning("MBD desactivado por --no-mbd (modo comparativo)")

    settings.MUSIC_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = settings.MUSIC_DIR / f"teaser_raw_{teaser_id}.wav"
    out_path = settings.MUSIC_DIR / f"teaser_{style_name}_{teaser_id}.mp3"

    from music_generator.prompt_engine import build_music_prompt
    from music_generator.generator import get_music_generator
    from music_generator.audio_processor import AudioProcessor

    prompt, bpm = build_music_prompt(style=style, duration_seconds=30, section="drop1")
    print(f"\n🎛  Teaser {style_name} | 30 s | {bpm} BPM | MBD={settings.MUSICGEN_USE_MBD}\n")
    print(f"📝 Prompt:\n{prompt}\n")

    t0 = time.time()

    # ── Generación ────────────────────────────────────────────────────────────
    gen = get_music_generator()
    logger.info(f"VRAM antes: {gen.vram_usage()}")
    gen.generate_clip(prompt, duration=30, output_path=raw_path)
    t_gen = time.time() - t0
    logger.info(f"VRAM tras generar: {gen.vram_usage()}")
    logger.info(f"⏱  Generación: {t_gen:.1f}s")

    # Liberar VRAM (no la usaremos más en el teaser)
    gen.unload()

    # ── Mastering ─────────────────────────────────────────────────────────────
    t1 = time.time()
    await AudioProcessor.process(raw_path, out_path, target_duration=30)
    t_master = time.time() - t1
    logger.info(f"⏱  Mastering: {t_master:.1f}s")

    raw_path.unlink(missing_ok=True)

    info = AudioProcessor.get_audio_info(out_path)
    size_mb = out_path.stat().st_size / 1e6

    print(f"\n{'=' * 60}")
    print("  ✅ TEASER LISTO")
    print(f"{'=' * 60}")
    print(f"  Archivo:      {out_path}")
    print(f"  Duración:     {info['duration_seconds']:.1f} s")
    print(f"  Sample rate:  {info['sample_rate']} Hz")
    print(f"  Tamaño:       {size_mb:.2f} MB")
    print(f"  Tiempo total: {(time.time() - t0):.1f} s "
          f"(gen {t_gen:.0f}s + master {t_master:.0f}s)")
    print(f"{'=' * 60}\n")
    print("▶ Reprodúcelo en tu reproductor habitual para evaluar la calidad.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--style", default="cyberpunk",
        choices=[s.value for s in MusicStyle],
        help="Estilo musical del teaser",
    )
    parser.add_argument(
        "--no-mbd", action="store_true",
        help="Desactivar Multi-Band Diffusion (para comparar A/B contra el decoder EnCodec)",
    )
    args = parser.parse_args()

    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"\nGPU: {gpu}")

    asyncio.run(run(args.style, args.no_mbd))
