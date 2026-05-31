"""
Rescate de chunks WAV generados por un run interrumpido.

Coge los `_chunk_raw_<job_id>_NNN.wav` del directorio assets/music/,
los concatena con la cadena de crossfade y aplica la cadena de mastering
profesional final. El resultado es un MP3 (+ M4A si EXPORT_AAC) que
puedes usar como si el run hubiera terminado.

Uso:
    python scripts/rescue_chunks.py --job-id 1 --tag ironpulse
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from music_generator.audio_processor import AudioProcessor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("rescue")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", type=int, required=True,
                        help="Job ID al que pertenecen los chunks (_chunk_raw_<id>_NNN.wav)")
    parser.add_argument("--tag", required=True,
                        help="Etiqueta para el archivo de salida (ironpulse, deepstack, etc.)")
    args = parser.parse_args()

    chunk_pattern = f"_chunk_raw_{args.job_id}_*.wav"
    chunks = sorted(settings.MUSIC_DIR.glob(chunk_pattern))
    if not chunks:
        print(f"❌ No se encontraron chunks con patrón: {chunk_pattern}")
        sys.exit(1)

    duration_per_chunk = settings.MUSICGEN_CLIP_DURATION   # 30s
    total_duration = len(chunks) * duration_per_chunk
    print(f"\n🛟 Rescatando {len(chunks)} chunks (~{total_duration}s de música)")
    print(f"   Job ID: {args.job_id}")
    print(f"   Tag:    {args.tag}")
    print()

    # 1) Concatenación con crossfade triangular (mantiene WAV sin pérdida)
    concat_path = settings.MUSIC_DIR / f"rescued_{args.tag}_{args.job_id}.wav"
    print(f"📎 Concatenando con crossfade…")
    await AudioProcessor.concatenate_clips_wav(chunks, concat_path)

    # 2) Master + export MP3/AAC
    out_path = settings.MUSIC_DIR / f"rescued_{args.tag}_{args.job_id}.mp3"
    print(f"🎚  Aplicando cadena de mastering profesional…")
    await AudioProcessor.process(
        input_path=concat_path,
        output_path=out_path,
        target_duration=total_duration,
    )
    concat_path.unlink(missing_ok=True)

    info = AudioProcessor.get_audio_info(out_path)
    print(f"\n{'=' * 60}")
    print("  ✅ AUDIO RESCATADO")
    print(f"{'=' * 60}")
    print(f"  Archivo:     {out_path}")
    print(f"  Duración:    {info['duration_seconds']:.1f} s ({info['duration_seconds']/60:.1f} min)")
    print(f"  Sample rate: {info['sample_rate']} Hz")
    print(f"  Tamaño:      {info['file_size_mb']:.2f} MB")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    asyncio.run(main())
