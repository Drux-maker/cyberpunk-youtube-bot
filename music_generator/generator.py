"""
Generación de música con MusicGen (Meta AudioCraft) — inferencia local.
Modelo: facebook/musicgen-medium (1.5B parámetros, ~4GB VRAM).
Optimizado para RTX 3060 6GB: fp16, CUDA cache clearing entre clips.
"""
import logging
import math
import torch
import torchaudio
from pathlib import Path
from typing import Optional

from config.settings import settings
from database.models import MusicAsset, MusicStyle
from database.db import get_db
from music_generator.prompt_engine import build_music_prompt_variations

logger = logging.getLogger(__name__)


class MusicGenerator:
    """
    Singleton que carga MusicGen una sola vez por proceso y lo reutiliza.
    Para clips largos (>30s) genera múltiples clips y los concatena con crossfade.
    """

    def __init__(self):
        self._model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.float16 if self._device == "cuda" else torch.float32
        logger.info(f"MusicGenerator — device: {self._device}")

    def _load(self):
        if self._model is not None:
            return
        from audiocraft.models import MusicGen
        logger.info(f"Cargando {settings.MUSICGEN_MODEL} (primera vez: descarga ~3GB)…")
        self._model = MusicGen.get_pretrained(settings.MUSICGEN_MODEL)
        if self._device == "cuda":
            self._model = self._model.half()
        logger.info("MusicGen listo ✅")

    def _set_params(self, duration: int):
        self._model.set_generation_params(
            duration=duration,
            top_k=250,
            top_p=0.0,
            temperature=settings.MUSICGEN_TEMPERATURE,
            cfg_coef=settings.MUSICGEN_CFG_COEF,
        )

    def generate_clip(self, prompt: str, duration: int, output_path: Path) -> Path:
        """Genera un único clip WAV de hasta 30 segundos."""
        self._load()
        self._set_params(min(duration, settings.MUSICGEN_CLIP_DURATION))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generando clip {duration}s: {prompt[:80]}…")
        with torch.cuda.amp.autocast(enabled=self._device == "cuda"):
            wav = self._model.generate([prompt], progress=True)

        wav = wav[0].cpu().float()
        torchaudio.save(str(output_path), wav, sample_rate=32000)
        torch.cuda.empty_cache()
        return output_path

    def generate_long(self, prompt: str, total_seconds: int, output_path: Path) -> Path:
        """
        Genera audio de larga duración concatenando clips de 30s con crossfade.
        Varía el prompt sutilmente en cada clip para evitar repetición.
        """
        clip_dur = settings.MUSICGEN_CLIP_DURATION
        overlap = settings.MUSICGEN_OVERLAP
        clips_needed = math.ceil(total_seconds / (clip_dur - overlap))

        logger.info(f"Generando {clips_needed} clips de {clip_dur}s → total {total_seconds}s")
        clip_paths: list[Path] = []

        for i in range(clips_needed):
            varied = self._vary_prompt(prompt, i, clips_needed)
            clip_path = output_path.parent / f"_chunk_{output_path.stem}_{i:03d}.wav"
            self.generate_clip(varied, clip_dur, clip_path)
            clip_paths.append(clip_path)
            logger.info(f"  [{i+1}/{clips_needed}] OK")

        import asyncio
        from music_generator.audio_processor import AudioProcessor

        async def _concat():
            return await AudioProcessor.concatenate_clips(clip_paths, output_path)

        result = asyncio.run(_concat())

        for p in clip_paths:
            p.unlink(missing_ok=True)

        return result

    def _vary_prompt(self, base: str, idx: int, total: int) -> str:
        pos = idx / max(total - 1, 1)
        if pos < 0.1:
            return base + ", intro, atmospheric build-up, slow start"
        elif pos < 0.3:
            return base + ", building energy, increasing intensity"
        elif pos < 0.7:
            return base + ", peak energy, full groove, main section"
        elif pos < 0.9:
            return base + ", sustained energy, hypnotic loop"
        else:
            return base + ", outro, fading intensity, winding down"

    def vram_usage(self) -> str:
        if not torch.cuda.is_available():
            return "CPU"
        used = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"{used:.1f}/{total:.1f}GB"

    async def run(
        self,
        job_id: int,
        style: MusicStyle,
        duration_seconds: int,
        prompt: str,
        bpm: int,
    ) -> int:
        """Genera música, procesa audio y guarda MusicAsset. Devuelve asset_id."""
        from music_generator.audio_processor import AudioProcessor

        raw_path = settings.MUSIC_DIR / f"raw_{job_id}.wav"
        settings.MUSIC_DIR.mkdir(parents=True, exist_ok=True)

        if duration_seconds <= settings.MUSICGEN_CLIP_DURATION:
            self.generate_clip(prompt, duration_seconds, raw_path)
        else:
            self.generate_long(prompt, duration_seconds, raw_path)

        processed_path = settings.MUSIC_DIR / f"music_{job_id}.mp3"
        await AudioProcessor.process(raw_path, processed_path, duration_seconds)
        raw_path.unlink(missing_ok=True)

        info = AudioProcessor.get_audio_info(processed_path)

        with get_db() as db:
            asset = MusicAsset(
                job_id=job_id,
                provider="musicgen-local",
                prompt=prompt,
                file_path=str(processed_path),
                bpm=float(bpm),
                duration_seconds=info["duration_seconds"],
                file_size_mb=info["file_size_mb"],
                sample_rate=settings.AUDIO_SAMPLE_RATE,
                loudness_lufs=settings.TARGET_LOUDNESS,
                is_processed=True,
                generation_metadata={"model": settings.MUSICGEN_MODEL, "device": self._device},
            )
            db.add(asset)
            db.flush()
            asset_id = asset.id

        logger.info(f"MusicAsset {asset_id} guardado ({processed_path.stat().st_size / 1e6:.1f}MB)")
        return asset_id

    async def run_with_retry(
        self,
        job_id: int,
        style: MusicStyle,
        duration_seconds: int,
    ) -> int:
        """Prueba hasta 3 variaciones de prompt antes de fallar."""
        import asyncio
        variations = build_music_prompt_variations(style, duration_seconds, count=3)
        last_error = None

        for attempt, v in enumerate(variations):
            try:
                logger.info(f"Intento {attempt + 1}/3 — BPM {v['bpm']}")
                return await self.run(
                    job_id=job_id,
                    style=style,
                    duration_seconds=duration_seconds,
                    prompt=v["prompt"],
                    bpm=v["bpm"],
                )
            except Exception as exc:
                last_error = exc
                logger.warning(f"Intento {attempt + 1} fallido: {exc}")
                torch.cuda.empty_cache()
                await asyncio.sleep(5)

        raise RuntimeError(f"Todos los intentos de generación fallaron: {last_error}")


# Singleton — una instancia por proceso worker
_instance: Optional[MusicGenerator] = None


def get_music_generator() -> MusicGenerator:
    global _instance
    if _instance is None:
        _instance = MusicGenerator()
    return _instance
