"""
MusicGen local inference — Meta AudioCraft.
Optimizado para RTX 3060 6GB VRAM con modelo 'medium' (1.5B parámetros).

Instalar: pip install audiocraft
Descarga del modelo: automática en primera ejecución (~3GB en ~/.cache)
"""
import logging
import torch
import torchaudio
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)


class MusicGenClient:
    """
    Wrapper sobre Meta AudioCraft MusicGen.
    Carga el modelo una sola vez y lo reutiliza (singleton).
    """

    MODEL_NAME = "facebook/musicgen-medium"  # 1.5B — cabe en 6GB VRAM

    def __init__(self):
        self._model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"MusicGen device: {self._device}")
        if self._device == "cuda":
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"VRAM disponible: {vram_gb:.1f} GB")

    def _load_model(self):
        if self._model is not None:
            return

        logger.info(f"Cargando {self.MODEL_NAME} (primera vez descarga ~3GB)...")
        from audiocraft.models import MusicGen
        self._model = MusicGen.get_pretrained(self.MODEL_NAME)
        self._model.set_generation_params(
            duration=30,        # segundos por clip
            top_k=250,
            top_p=0.0,
            temperature=1.0,
            cfg_coef=3.0,       # adherencia al prompt (3-5 es buen rango)
        )

        if self._device == "cuda":
            # Liberar VRAM al máximo con half precision
            self._model = self._model.half()

        logger.info("MusicGen listo")

    def generate_clip(
        self,
        prompt: str,
        duration_seconds: int = 30,
        output_path: Path | None = None,
    ) -> Path:
        """
        Genera un clip de audio y lo guarda como WAV.
        duration_seconds: máximo recomendado 30s para 6GB VRAM.
        Para clips más largos usa generate_long().
        """
        self._load_model()

        self._model.set_generation_params(
            duration=min(duration_seconds, 30),
            cfg_coef=3.5,
        )

        logger.info(f"Generando clip ({duration_seconds}s): {prompt[:80]}...")

        with torch.cuda.amp.autocast():
            wav = self._model.generate([prompt], progress=True)

        # wav shape: [batch, channels, samples]
        wav = wav[0].cpu().float()  # back to float32 para guardar

        if output_path is None:
            import uuid
            output_path = settings.MUSIC_DIR / f"clip_{uuid.uuid4().hex[:8]}.wav"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(
            str(output_path),
            wav,
            sample_rate=32000,  # MusicGen usa 32kHz
        )

        # Liberar VRAM entre clips
        torch.cuda.empty_cache()

        logger.info(f"Clip guardado: {output_path}")
        return output_path

    def generate_long(
        self,
        prompt: str,
        total_duration_seconds: int,
        output_path: Path,
        clip_duration: int = 30,
        overlap_seconds: int = 2,
    ) -> Path:
        """
        Genera audio largo concatenando clips con crossfade.
        Para un vídeo de 1h genera 120 clips de 30s.
        """
        import math
        self._load_model()

        clips_needed = math.ceil(total_duration_seconds / (clip_duration - overlap_seconds))
        logger.info(f"Generando {clips_needed} clips de {clip_duration}s para {total_duration_seconds}s total")

        clip_paths = []
        for i in range(clips_needed):
            # Variaciones sutiles en el prompt para evitar repetición
            varied_prompt = self._vary_prompt(prompt, i, clips_needed)
            clip_path = settings.MUSIC_DIR / f"chunk_{output_path.stem}_{i:03d}.wav"
            self.generate_clip(varied_prompt, clip_duration, clip_path)
            clip_paths.append(clip_path)
            logger.info(f"  Clip {i+1}/{clips_needed} OK")

        # Concatenar con crossfade usando FFmpeg
        final_path = self._concat_with_crossfade(clip_paths, output_path, overlap_seconds)

        # Limpiar clips intermedios
        for p in clip_paths:
            p.unlink(missing_ok=True)

        return final_path

    def _vary_prompt(self, base_prompt: str, clip_index: int, total_clips: int) -> str:
        """Añade variaciones al prompt según la posición en el mix."""
        position = clip_index / total_clips

        if position < 0.1:
            return base_prompt + ", intro, building atmosphere, slow start"
        elif position < 0.3:
            return base_prompt + ", building energy, increasing intensity"
        elif position < 0.7:
            return base_prompt + ", peak energy, full groove, main section"
        elif position < 0.9:
            return base_prompt + ", sustained energy, hypnotic loop"
        else:
            return base_prompt + ", outro, fading intensity, winding down"

    def _concat_with_crossfade(
        self, clip_paths: list[Path], output_path: Path, overlap: int
    ) -> Path:
        import asyncio
        from music_generator.audio_processor import AudioProcessor

        async def _run():
            return await AudioProcessor.concatenate_clips(clip_paths, output_path)

        return asyncio.run(_run())

    def get_vram_usage(self) -> str:
        if not torch.cuda.is_available():
            return "CPU mode"
        used = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"{used:.1f}GB / {total:.1f}GB"


# Singleton — se carga una vez por proceso worker
_musicgen_client: MusicGenClient | None = None


def get_musicgen_client() -> MusicGenClient:
    global _musicgen_client
    if _musicgen_client is None:
        _musicgen_client = MusicGenClient()
    return _musicgen_client
