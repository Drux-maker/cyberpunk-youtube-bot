"""
Generación de imágenes con Stable Diffusion local (HuggingFace Diffusers).
Modelo principal: SDXL (stabilityai/stable-diffusion-xl-base-1.0)
Fallback:         SD 1.5 (runwayml/stable-diffusion-v1-5)
Optimizado para RTX 3060 6GB VRAM: fp16, xformers, attention slicing, vae tiling.
"""
import logging
import uuid
import torch
from pathlib import Path
from typing import Optional

from config.settings import settings
from database.models import VisualAsset, MusicStyle
from database.db import get_db
from visual_generator.prompt_engine import build_visual_batch

logger = logging.getLogger(__name__)

# Resolución de generación — FFmpeg upscalea a 1080p si hace falta
GEN_W, GEN_H = 1280, 720


class VisualGenerator:
    """
    Singleton que carga el pipeline de SD una vez y lo reutiliza.
    Genera imágenes secuencialmente (6GB VRAM no permite batches grandes).
    Incluye manejo automático de OOM con fallback a resolución reducida.
    """

    def __init__(self):
        self._pipe = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._using_sdxl = True
        logger.info(f"VisualGenerator — device: {self._device}")

    def _load(self):
        if self._pipe is not None:
            return

        from diffusers import (
            StableDiffusionXLPipeline,
            StableDiffusionPipeline,
            DPMSolverMultistepScheduler,
        )

        # Intentar SDXL primero, caer a SD 1.5 si falla
        for model_id, is_sdxl in [
            (settings.SD_MODEL, True),
            (settings.SD_FALLBACK_MODEL, False),
        ]:
            try:
                logger.info(f"Cargando {model_id} (primera vez: descarga ~7GB)…")
                if is_sdxl:
                    pipe = StableDiffusionXLPipeline.from_pretrained(
                        model_id,
                        torch_dtype=self._dtype,
                        use_safetensors=True,
                        variant="fp16" if self._device == "cuda" else None,
                    )
                else:
                    pipe = StableDiffusionPipeline.from_pretrained(
                        model_id,
                        torch_dtype=self._dtype,
                        safety_checker=None,
                        requires_safety_checker=False,
                    )

                pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipe.scheduler.config,
                    use_karras_sigmas=True,
                    algorithm_type="dpmsolver++",
                )
                pipe = pipe.to(self._device)
                pipe.enable_attention_slicing(slice_size="auto")
                pipe.enable_vae_tiling()

                if settings.SD_USE_XFORMERS:
                    try:
                        pipe.enable_xformers_memory_efficient_attention()
                        logger.info("xformers activado ✅")
                    except Exception:
                        logger.warning("xformers no disponible — usando attention slicing")

                self._pipe = pipe
                self._using_sdxl = is_sdxl
                logger.info(f"Pipeline SD listo: {'SDXL' if is_sdxl else 'SD 1.5'} ✅")
                return

            except Exception as exc:
                logger.warning(f"{model_id} falló: {exc} — probando fallback…")
                torch.cuda.empty_cache()

        raise RuntimeError("No se pudo cargar ningún modelo de Stable Diffusion.")

    def generate_one(
        self,
        positive_prompt: str,
        negative_prompt: str,
        output_path: Path,
        width: int = GEN_W,
        height: int = GEN_H,
        steps: int = None,
        seed: Optional[int] = None,
    ) -> Path:
        self._load()
        steps = steps or settings.SD_STEPS
        # SDXL y SD requieren múltiplos de 8
        width  = (width  // 8) * 8
        height = (height // 8) * 8
        output_path.parent.mkdir(parents=True, exist_ok=True)

        generator = (
            torch.Generator(device=self._device).manual_seed(seed)
            if seed is not None else None
        )

        logger.info(f"Generando {width}×{height} | {steps} pasos | {positive_prompt[:60]}…")
        with torch.inference_mode():
            result = self._pipe(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=settings.SD_GUIDANCE_SCALE,
                generator=generator,
            )

        result.images[0].save(str(output_path), "PNG")
        torch.cuda.empty_cache()
        return output_path

    def generate_batch(
        self,
        prompts: list[dict],
        output_dir: Path,
    ) -> list[Path]:
        """
        Genera una lista de imágenes secuencialmente.
        En OOM reintenta con resolución reducida automáticamente.
        """
        self._load()
        output_dir.mkdir(parents=True, exist_ok=True)
        results: list[Path] = []

        for i, p in enumerate(prompts):
            out = output_dir / f"img_{uuid.uuid4().hex[:8]}_{i:02d}.png"
            try:
                results.append(self.generate_one(
                    p["positive"], p.get("negative", ""), out
                ))
                logger.info(f"  [{i+1}/{len(prompts)}] OK")
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"  [{i+1}/{len(prompts)}] OOM — reintentando en resolución reducida")
                torch.cuda.empty_cache()
                try:
                    results.append(self.generate_one(
                        p["positive"], p.get("negative", ""), out,
                        width=768, height=432, steps=15,
                    ))
                    logger.info(f"  [{i+1}/{len(prompts)}] OK (resolución reducida)")
                except Exception as e2:
                    logger.error(f"  [{i+1}/{len(prompts)}] Reintento fallido: {e2}")
            except Exception as e:
                logger.error(f"  [{i+1}/{len(prompts)}] Error: {e}")

        return results

    def vram_usage(self) -> str:
        if not torch.cuda.is_available():
            return "CPU"
        used  = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"{used:.1f}/{total:.1f}GB"

    async def run_batch(
        self,
        job_id: int,
        style: MusicStyle,
        n_images: int = 8,
    ) -> list[int]:
        """Genera `n_images` imágenes y las guarda como VisualAssets. Devuelve IDs."""
        prompts = build_visual_batch(style, n_images)
        output_dir = settings.VISUALS_DIR / str(job_id)
        paths = self.generate_batch(prompts, output_dir)

        asset_ids: list[int] = []
        with get_db() as db:
            for path, prompt_data in zip(paths, prompts):
                asset = VisualAsset(
                    job_id=job_id,
                    provider="stable-diffusion-local",
                    prompt=prompt_data["positive"],
                    file_path=str(path),
                    asset_type="image",
                    width=GEN_W,
                    height=GEN_H,
                    generation_metadata={
                        "model": settings.SD_MODEL if self._using_sdxl else settings.SD_FALLBACK_MODEL,
                        "subject": prompt_data.get("subject", ""),
                    },
                )
                db.add(asset)
                db.flush()
                asset_ids.append(asset.id)

        logger.info(f"Generados {len(asset_ids)}/{n_images} visuales para job {job_id}")
        return asset_ids


# Singleton
_instance: Optional[VisualGenerator] = None


def get_visual_generator() -> VisualGenerator:
    global _instance
    if _instance is None:
        _instance = VisualGenerator()
    return _instance
