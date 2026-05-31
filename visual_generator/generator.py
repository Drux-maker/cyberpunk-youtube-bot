"""
Generación de imágenes con JuggernautXL Lightning (HuggingFace Diffusers).

Modelo: RunDiffusion/Juggernaut-XL-Lightning
  - Refinado SDXL con foto-realismo superior al base.
  - Scheduler Lightning: 4-8 pasos (~5-10s/imagen en RTX 3060), CFG 1.5-3.0.
  - Cabe en 6 GB VRAM en fp16 con attention slicing + vae tiling.

Optimizado para RTX 3060 6 GB: fp16, xformers, attention slicing, vae tiling,
y manejo automático de OOM con reintento a resolución reducida.
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
    Singleton que carga el pipeline de JuggernautXL Lightning una vez y lo
    reutiliza. Genera imágenes secuencialmente (6 GB VRAM no permite batches
    grandes). En OOM reintenta automáticamente a resolución reducida.
    """

    def __init__(self):
        self._pipe = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.float16 if self._device == "cuda" else torch.float32
        logger.info(f"VisualGenerator — device: {self._device}")

    def _load(self):
        if self._pipe is not None:
            return

        from diffusers import (
            StableDiffusionXLPipeline,
            DPMSolverSinglestepScheduler,
        )

        logger.info(f"Cargando {settings.SD_MODEL}…")
        # Estrategia de carga: probamos las combinaciones más comunes en orden
        # de preferencia. Para JuggernautXL Lightning lo que funciona es
        # use_safetensors=False (el repo solo publica pytorch .bin); pero el
        # mismo loader debe valer para otros modelos que SÍ publican
        # safetensors fp16. Vamos del más eficiente al más permisivo.
        attempts = [
            # 1) Modelos modernos: safetensors + variant fp16 (mitad de tamaño)
            {"variant": "fp16", "use_safetensors": True},
            # 2) Modelos con safetensors pero sin variant fp16
            {"use_safetensors": True},
            # 3) JuggernautXL Lightning y otros que solo publican pytorch .bin
            {"use_safetensors": False},
        ]
        if self._device != "cuda":
            attempts = [{k: v for k, v in a.items() if k != "variant"} for a in attempts]

        pipe = None
        last_exc: Exception | None = None
        for opts in attempts:
            try:
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    settings.SD_MODEL,
                    torch_dtype=self._dtype,
                    **opts,
                )
                logger.info(f"SDXL cargado con opciones: {opts}")
                break
            except Exception as exc:
                last_exc = exc
                msg = str(exc)
                # Truncar para que el log no estalle con tracebacks de diffusers
                logger.warning(f"Carga con {opts} falló ({msg[:120]}…); probando siguiente…")
        if pipe is None:
            raise RuntimeError(f"No se pudo cargar {settings.SD_MODEL}. Último error: {last_exc}")

        # Scheduler óptimo según el modelo:
        # - Lightning models (pocos pasos, CFG bajo) → DPMSolverSinglestepScheduler
        # - Modelos estándar SDXL (25 pasos, CFG ~7) → mismo scheduler en modo
        #   multistep funciona, pero singlestep también es válido para Lightning.
        # use_karras_sigmas=False es lo recomendado por el autor de Juggernaut LT.
        is_lightning = "lightning" in settings.SD_MODEL.lower() or settings.SD_STEPS <= 10
        pipe.scheduler = DPMSolverSinglestepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=not is_lightning,
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
        logger.info(f"Pipeline listo: JuggernautXL Lightning ✅ "
                    f"(steps={settings.SD_STEPS}, cfg={settings.SD_GUIDANCE_SCALE})")

    def generate_one(
        self,
        positive_prompt: str,
        negative_prompt: str,
        output_path: Path,
        width: int = GEN_W,
        height: int = GEN_H,
        steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Path:
        self._load()
        steps = steps or settings.SD_STEPS
        # SDXL requiere múltiplos de 8
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
                        width=768, height=432, steps=max(settings.SD_STEPS, 6),
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

    def unload(self) -> None:
        """Libera VRAM tras generar todas las imágenes."""
        if self._pipe is None:
            return
        import gc
        try:
            self._pipe.to("cpu")
        except Exception:
            pass
        del self._pipe
        self._pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        logger.info("VisualGenerator descargado de VRAM")

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
                    provider="juggernaut-xl-lightning-local",
                    prompt=prompt_data["positive"],
                    file_path=str(path),
                    asset_type="image",
                    width=GEN_W,
                    height=GEN_H,
                    generation_metadata={
                        "model": settings.SD_MODEL,
                        "steps": settings.SD_STEPS,
                        "cfg": settings.SD_GUIDANCE_SCALE,
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
