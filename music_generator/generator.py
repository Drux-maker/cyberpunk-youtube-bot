"""
Generación de música con MusicGen (Meta AudioCraft) — inferencia local.

Mejoras de calidad clave respecto a la versión previa:
  1. Modelo estéreo (musicgen-stereo-medium) con fallback automático a mono
     si hay OOM de VRAM.
  2. Long-form COHERENTE mediante generate_continuation(): cada clip ve los
     últimos N segundos del clip anterior → no más "audio collage".
  3. CFG dinámico (más libre en intro, más estricto en cuerpo) → resultados
     menos genéricos.
  4. Sin doble codificación MP3 — la concatenación queda en WAV y el master
     final hace una única pasada de encoding.
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
from music_generator.prompt_engine import (
    build_music_prompt_variations,
    build_music_prompt,
    section_for_position,
    sample_track_identity,
)

logger = logging.getLogger(__name__)


class MusicGenerator:
    """Singleton: una sola carga del modelo por proceso."""

    def __init__(self):
        self._model = None
        self._mbd = None                             # Multi-Band Diffusion decoder
        self._use_mbd: bool = settings.MUSICGEN_USE_MBD
        self._model_name: Optional[str] = None
        self._is_stereo: bool = False
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"MusicGenerator — device: {self._device}  MBD={self._use_mbd}")

    # ─── carga del modelo ────────────────────────────────────────────────────
    def _load_model(self, model_name: str) -> None:
        from audiocraft.models import MusicGen
        logger.info(f"Cargando {model_name} (puede descargar varios GB la primera vez)…")
        self._model = MusicGen.get_pretrained(model_name)
        if self._device == "cuda":
            self._model.lm.half()
            self._model.compression_model.half()
        self._model_name = model_name
        self._is_stereo = "stereo" in model_name.lower()
        logger.info(f"Modelo listo — {'stereo' if self._is_stereo else 'mono'}")

    def _load_mbd(self) -> bool:
        """
        Carga Multi-Band Diffusion bajo demanda. Devuelve True si se cargó OK,
        False si OOM (en cuyo caso seguimos con EnCodec estándar).
        """
        if self._mbd is not None:
            return True
        if not self._use_mbd:
            return False
        try:
            from audiocraft.models import MultiBandDiffusion
            logger.info("Cargando Multi-Band Diffusion (decoder de alta calidad)…")
            self._mbd = MultiBandDiffusion.get_mbd_musicgen(device=self._device)
            logger.info(f"MBD listo  VRAM={self.vram_usage()}")
            return True
        except torch.cuda.OutOfMemoryError:
            logger.warning("OOM cargando MBD — continuamos con decoder EnCodec estándar")
            self._mbd = None
            self._use_mbd = False
            torch.cuda.empty_cache()
            return False
        except Exception as exc:
            logger.warning(f"MBD no disponible ({exc}) — usando EnCodec estándar")
            self._mbd = None
            self._use_mbd = False
            return False

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            self._load_model(settings.MUSICGEN_MODEL)
        except torch.cuda.OutOfMemoryError:
            logger.warning("OOM con modelo estéreo — fallback a mono")
            self._model = None
            torch.cuda.empty_cache()
            self._load_model(settings.MUSICGEN_FALLBACK_MODEL)
        except Exception as exc:
            logger.warning(f"Error cargando modelo principal ({exc}) — fallback a mono")
            self._model = None
            torch.cuda.empty_cache()
            self._load_model(settings.MUSICGEN_FALLBACK_MODEL)

    def _set_params(self, duration: int, cfg_coef: float) -> None:
        self._model.set_generation_params(
            duration=duration,
            top_k=settings.MUSICGEN_TOP_K,
            top_p=settings.MUSICGEN_TOP_P,
            temperature=settings.MUSICGEN_TEMPERATURE,
            cfg_coef=cfg_coef,
        )

    # ─── generación de un clip ───────────────────────────────────────────────
    def _save_wav(self, wav_tensor: torch.Tensor, output_path: Path) -> None:
        """wav_tensor: [channels, samples] en float CPU."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(
            str(output_path),
            wav_tensor,
            sample_rate=settings.MUSICGEN_NATIVE_SR,
            encoding="PCM_S",
            bits_per_sample=16,
        )

    def _decode_with_mbd(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Decodifica tokens EnCodec con Multi-Band Diffusion en lugar del decoder
        original. Devuelve wav float CPU [channels, samples] a 32 kHz.
        Si MBD no está disponible, devuelve None y el caller usa el wav EnCodec.

        Stereo handling: MBD está entrenado en MONO (4 codebooks). MusicGen
        Stereo Medium emite 8 codebooks (4 por canal). Si detectamos shape
        [B, 8, T] dividimos en [B, 4, T] izquierdo + [B, 4, T] derecho,
        decodificamos cada uno por separado y los apilamos como estéreo.
        """
        if self._mbd is None and not self._load_mbd():
            return None
        try:
            with torch.no_grad():
                n_codebooks = tokens.shape[1]
                if n_codebooks == 4:
                    # mono nativo
                    wav_mbd = self._mbd.tokens_to_wav(tokens)
                    return wav_mbd[0].cpu().float()
                if n_codebooks == 8:
                    # estéreo: 4 codebooks por canal, decodificar por separado
                    left  = self._mbd.tokens_to_wav(tokens[:, :4, :])   # [B, 1, T]
                    right = self._mbd.tokens_to_wav(tokens[:, 4:, :])
                    # Stack canales: [1, 2, samples]
                    stereo = torch.cat([left, right], dim=1)
                    return stereo[0].cpu().float()
                logger.warning(
                    f"MBD: número de codebooks inesperado ({n_codebooks}); "
                    "fallback a EnCodec"
                )
                return None
        except torch.cuda.OutOfMemoryError:
            logger.warning("OOM en MBD decode — fallback a EnCodec para este clip")
            torch.cuda.empty_cache()
            return None
        except Exception as exc:
            logger.warning(f"MBD decode falló ({exc}) — fallback a EnCodec")
            return None

    def generate_clip(
        self,
        prompt: str,
        duration: int,
        output_path: Path,
        cfg_coef: Optional[float] = None,
    ) -> Path:
        """Genera un clip único de hasta MUSICGEN_CLIP_DURATION segundos."""
        self._load()
        self._set_params(
            min(duration, settings.MUSICGEN_CLIP_DURATION),
            cfg_coef if cfg_coef is not None else settings.MUSICGEN_CFG_BODY,
        )
        logger.info(f"Generando clip {duration}s [cfg={cfg_coef}]: {prompt[:90]}…")

        with torch.cuda.amp.autocast(enabled=self._device == "cuda"):
            wav, tokens = self._model.generate([prompt], progress=True, return_tokens=True)

        # Intentar decodificar con MBD para máxima calidad de audio
        wav_mbd = self._decode_with_mbd(tokens)
        wav_final = wav_mbd if wav_mbd is not None else wav[0].cpu().float()

        self._save_wav(wav_final, output_path)
        torch.cuda.empty_cache()
        return output_path

    def _generate_continuation(
        self,
        prompt: str,
        context_wav: torch.Tensor,
        context_sr: int,
        new_duration: int,
        output_path: Path,
        cfg_coef: float,
    ) -> Path:
        """
        Continúa una pista existente. Pasamos los últimos N segundos como
        contexto para que MusicGen mantenga melodía, armonía y groove.

        Nota importante: el contexto que se pasa al modelo debe ser audio
        decodificado por el codec ORIGINAL (EnCodec), no por MBD. El modelo
        fue entrenado con representaciones EnCodec; mezclarlas confunde la
        continuación. El wav final SÍ se decodifica con MBD para máxima
        calidad de output.
        """
        ctx_seconds = context_wav.shape[-1] / context_sr
        total = int(ctx_seconds + new_duration)
        self._set_params(total, cfg_coef)

        # MusicGen espera batch dim: [B, C, T]
        if context_wav.dim() == 2:
            context_batched = context_wav.unsqueeze(0)
        else:
            context_batched = context_wav

        if self._device == "cuda":
            context_batched = context_batched.to("cuda")

        logger.info(
            f"Continuando {new_duration}s sobre {ctx_seconds:.1f}s de contexto "
            f"[cfg={cfg_coef}]: {prompt[:80]}…"
        )
        with torch.cuda.amp.autocast(enabled=self._device == "cuda"):
            wav, tokens = self._model.generate_continuation(
                prompt=context_batched,
                prompt_sample_rate=context_sr,
                descriptions=[prompt],
                progress=True,
                return_tokens=True,
            )

        # Decodificar con MBD para máxima calidad del clip final
        wav_mbd = self._decode_with_mbd(tokens)
        if wav_mbd is not None:
            wav_full = wav_mbd
        else:
            wav_full = wav[0].cpu().float()

        # recortar el contexto inicial — solo nos quedamos con lo nuevo
        ctx_samples = context_wav.shape[-1]
        new_only = wav_full[..., ctx_samples:]
        self._save_wav(new_only, output_path)
        torch.cuda.empty_cache()
        return output_path

    # ─── long-form coherente ─────────────────────────────────────────────────
    async def generate_long(
        self,
        base_prompt_fn,
        total_seconds: int,
        output_path: Path,
        style: MusicStyle,
        bpm: int,
    ) -> Path:
        """
        Genera audio de larga duración manteniendo coherencia melódica/armónica
        mediante generate_continuation().

        base_prompt_fn(section: str) -> str  permite variar la sección sin
        cambiar el resto del prompt.
        """
        self._load()
        clip_dur = settings.MUSICGEN_CLIP_DURATION
        ctx_secs = settings.MUSICGEN_CONTINUATION_SECONDS
        clips_needed = max(1, math.ceil(total_seconds / clip_dur))

        logger.info(
            f"Long-form: {clips_needed} clips × {clip_dur}s "
            f"({'stereo' if self._is_stereo else 'mono'}) → ~{total_seconds}s"
        )

        clip_paths: list[Path] = []
        prev_wav: Optional[torch.Tensor] = None
        prev_sr: int = settings.MUSICGEN_NATIVE_SR

        for i in range(clips_needed):
            section = section_for_position(i, clips_needed)
            prompt = base_prompt_fn(section)
            clip_path = output_path.parent / f"_chunk_{output_path.stem}_{i:03d}.wav"

            cfg = (
                settings.MUSICGEN_CFG_INTRO if section == "intro"
                else settings.MUSICGEN_CFG_BODY
            )

            if i == 0 or prev_wav is None:
                self.generate_clip(prompt, clip_dur, clip_path, cfg_coef=cfg)
            else:
                ctx_samples = int(ctx_secs * prev_sr)
                context = prev_wav[..., -ctx_samples:]
                self._generate_continuation(
                    prompt=prompt,
                    context_wav=context,
                    context_sr=prev_sr,
                    new_duration=clip_dur,
                    output_path=clip_path,
                    cfg_coef=cfg,
                )

            prev_wav, prev_sr = torchaudio.load(str(clip_path))
            clip_paths.append(clip_path)
            logger.info(f"  [{i+1}/{clips_needed}] OK ({section})  VRAM={self.vram_usage()}")

        from music_generator.audio_processor import AudioProcessor
        result = await AudioProcessor.concatenate_clips_wav(clip_paths, output_path)

        for p in clip_paths:
            p.unlink(missing_ok=True)
        return result

    def vram_usage(self) -> str:
        if not torch.cuda.is_available():
            return "CPU"
        used = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"{used:.1f}/{total:.1f}GB"

    def unload(self) -> None:
        """Libera VRAM. Crítico antes de cargar Juggernaut en GPUs de 6GB."""
        if self._model is None and self._mbd is None:
            return
        import gc
        if self._mbd is not None:
            try:
                # MBD no tiene un único .to() global; movemos los componentes
                for attr in ("models", "DPMs", "codec_model"):
                    obj = getattr(self._mbd, attr, None)
                    if obj is not None:
                        try:
                            if isinstance(obj, list):
                                for sub in obj:
                                    sub.to("cpu")
                            else:
                                obj.to("cpu")
                        except Exception:
                            pass
            except Exception:
                pass
            del self._mbd
            self._mbd = None
        if self._model is not None:
            try:
                self._model.lm.to("cpu")
                self._model.compression_model.to("cpu")
            except Exception:
                pass
            del self._model
            self._model = None
            self._model_name = None
            self._is_stereo = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        logger.info("MusicGenerator descargado de VRAM")

    # ─── orquestación de alto nivel ──────────────────────────────────────────
    async def run(
        self,
        job_id: int,
        style: MusicStyle,
        duration_seconds: int,
        prompt: str,
        bpm: int,
    ) -> int:
        """Genera + procesa + persiste MusicAsset. Devuelve asset_id."""
        from music_generator.audio_processor import AudioProcessor

        settings.MUSIC_DIR.mkdir(parents=True, exist_ok=True)
        raw_path = settings.MUSIC_DIR / f"raw_{job_id}.wav"

        if duration_seconds <= settings.MUSICGEN_CLIP_DURATION:
            self.generate_clip(prompt, duration_seconds, raw_path)
        else:
            # Fijar instrumentación + referencia UNA vez para toda la pista,
            # sólo variará el descriptor de sección (energía).
            track_elements, track_ref = sample_track_identity(style)

            def prompt_for_section(section: str) -> str:
                section_prompt, _ = build_music_prompt(
                    style=style, duration_seconds=duration_seconds,
                    bpm=bpm, section=section,
                    elements=track_elements, reference=track_ref,
                )
                return section_prompt

            await self.generate_long(
                base_prompt_fn=prompt_for_section,
                total_seconds=duration_seconds,
                output_path=raw_path,
                style=style,
                bpm=bpm,
            )

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
                generation_metadata={
                    "model": self._model_name or settings.MUSICGEN_MODEL,
                    "stereo": self._is_stereo,
                    "device": self._device,
                },
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
        channel_key: str | None = None,
    ) -> int:
        """Hasta 3 variaciones de prompt antes de fallar."""
        import asyncio
        variations = build_music_prompt_variations(
            style, duration_seconds, count=3, channel_key=channel_key,
        )
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


# Singleton — una instancia por proceso
_instance: Optional[MusicGenerator] = None


def get_music_generator() -> MusicGenerator:
    global _instance
    if _instance is None:
        _instance = MusicGenerator()
    return _instance
