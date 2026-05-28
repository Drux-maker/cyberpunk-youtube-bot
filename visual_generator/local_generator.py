"""
Stable Diffusion local inference — optimizado para RTX 3060 6GB VRAM.
Usa diffusers de HuggingFace con xformers + attention slicing + VAE tiling.

Modelos soportados:
  - SD 1.5:  ~2GB VRAM, 512×768, rápido (~15s)
  - SDXL:    ~5.5GB VRAM, 1024×576, más lento (~45s) — recomendado

Instalar:
  pip install diffusers transformers accelerate xformers
"""
import logging
import torch
from pathlib import Path
from typing import Optional
import uuid

from config.settings import settings

logger = logging.getLogger(__name__)

# Resolución objetivo para el vídeo
TARGET_W, TARGET_H = 1280, 720   # generamos en 720p, upscalamos luego con FFmpeg


class StableDiffusionLocal:
    """
    Pipeline local de Stable Diffusion con optimizaciones para 6GB VRAM:
    - attention_slicing: reduce uso de VRAM a costa de ~10% más de tiempo
    - xformers: si está instalado, más rápido y menos VRAM
    - vae_tiling: permite resoluciones altas sin OOM
    - half precision (fp16): obligatorio en GPU
    """

    # SDXL da mejor calidad para cyberpunk — úsalo si VRAM alcanza
    MODEL_SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
    # SD 1.5 más ligero, siempre funciona en 6GB
    MODEL_SD15 = "runwayml/stable-diffusion-v1-5"

    def __init__(self, prefer_sdxl: bool = True):
        self._pipe = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._prefer_sdxl = prefer_sdxl
        logger.info(f"SD Local device: {self._device} | dtype: {self._dtype}")

    def _load_pipeline(self):
        if self._pipe is not None:
            return

        from diffusers import (
            StableDiffusionXLPipeline,
            StableDiffusionPipeline,
            DPMSolverMultistepScheduler,
        )

        model_id = self.MODEL_SDXL if self._prefer_sdxl else self.MODEL_SD15
        logger.info(f"Cargando {model_id} (primera vez descarga ~5-7GB)...")

        try:
            if self._prefer_sdxl:
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
                    safety_checker=None,  # desactivado — contenido artístico
                    requires_safety_checker=False,
                )
        except Exception as e:
            logger.warning(f"SDXL falló ({e}), usando SD 1.5 como fallback")
            pipe = StableDiffusionPipeline.from_pretrained(
                self.MODEL_SD15,
                torch_dtype=self._dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
            self._prefer_sdxl = False

        # Scheduler DPM++ 2M — mejor calidad en pocos pasos
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="dpmsolver++",
        )

        pipe = pipe.to(self._device)

        # ─── Optimizaciones para 6GB VRAM ─────────────────────────────────────
        pipe.enable_attention_slicing(slice_size="auto")
        pipe.enable_vae_tiling()  # permite resoluciones altas

        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("xformers activado ✅")
        except Exception:
            logger.warning("xformers no disponible, usando attention slicing")

        self._pipe = pipe
        logger.info(f"Pipeline SD listo: {model_id}")

    def generate(
        self,
        positive_prompt: str,
        negative_prompt: str,
        width: int = TARGET_W,
        height: int = TARGET_H,
        steps: int = 25,
        guidance_scale: float = 7.5,
        output_path: Optional[Path] = None,
        seed: Optional[int] = None,
    ) -> Path:
        """
        Genera una imagen y la guarda como PNG.
        RTX 3060 6GB: ~20-40s por imagen a 1280×720 con SDXL.
        """
        self._load_pipeline()

        if output_path is None:
            output_path = settings.VISUALS_DIR / f"img_{uuid.uuid4().hex[:8]}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(seed)

        # SDXL funciona mejor en múltiplos de 8
        width = (width // 8) * 8
        height = (height // 8) * 8

        logger.info(f"Generando {width}×{height} | {steps} pasos | cfg={guidance_scale}")

        with torch.inference_mode():
            result = self._pipe(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

        image = result.images[0]
        image.save(str(output_path), "PNG")
        torch.cuda.empty_cache()

        logger.info(f"Imagen guardada: {output_path}")
        return output_path

    def generate_batch(
        self,
        prompts: list[dict],
        output_dir: Optional[Path] = None,
    ) -> list[Path]:
        """
        Genera varias imágenes secuencialmente.
        No en paralelo — 6GB VRAM no da para batches grandes.
        """
        self._load_pipeline()
        output_dir = output_dir or settings.VISUALS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for i, prompt_data in enumerate(prompts):
            try:
                out = output_dir / f"img_{uuid.uuid4().hex[:8]}_{i:02d}.png"
                path = self.generate(
                    positive_prompt=prompt_data["positive"],
                    negative_prompt=prompt_data.get("negative", ""),
                    output_path=out,
                )
                results.append(path)
                logger.info(f"  [{i+1}/{len(prompts)}] OK: {out.name}")
            except torch.cuda.OutOfMemoryError:
                logger.error(f"  [{i+1}/{len(prompts)}] OOM — liberando VRAM y reintentando con pasos reducidos")
                torch.cuda.empty_cache()
                # Reintento con menos pasos
                try:
                    out = output_dir / f"img_retry_{i:02d}.png"
                    path = self.generate(
                        positive_prompt=prompt_data["positive"],
                        negative_prompt=prompt_data.get("negative", ""),
                        output_path=out,
                        steps=15,
                        width=768,
                        height=432,
                    )
                    results.append(path)
                except Exception as e2:
                    logger.error(f"  Reintento fallido: {e2}")
            except Exception as e:
                logger.error(f"  [{i+1}/{len(prompts)}] Error: {e}")

        return results

    def get_vram_usage(self) -> str:
        if not torch.cuda.is_available():
            return "CPU mode"
        used = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"{used:.1f}GB / {total:.1f}GB"


# Singleton
_sd_client: StableDiffusionLocal | None = None


def get_sd_client(prefer_sdxl: bool = True) -> StableDiffusionLocal:
    global _sd_client
    if _sd_client is None:
        _sd_client = StableDiffusionLocal(prefer_sdxl=prefer_sdxl)
    return _sd_client
