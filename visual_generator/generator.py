"""
Image/video generation via Stability AI (SDXL/SD3), Leonardo, or Replicate.
Falls back through providers based on what's configured.
"""
import asyncio
import base64
import httpx
import logging
from pathlib import Path

from config.settings import settings
from database.models import VisualAsset, MusicStyle
from database.db import get_db
from visual_generator.prompt_engine import build_visual_batch

logger = logging.getLogger(__name__)


class StabilityClient:
    """Stability AI API v2beta."""

    def __init__(self):
        self.api_key = settings.STABILITY_API_KEY
        self.base_url = settings.STABILITY_API_BASE
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

    async def generate_image(
        self,
        positive_prompt: str,
        negative_prompt: str,
        width: int = 1920,
        height: int = 1080,
        steps: int = 30,
        cfg_scale: float = 7.0,
        model: str = "sd3.5-large-turbo",
    ) -> bytes:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self.base_url}/stable-image/generate/sd3",
                headers={**self.headers, "Accept": "image/*"},
                data={
                    "prompt": positive_prompt,
                    "negative_prompt": negative_prompt,
                    "model": model,
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "cfg_scale": cfg_scale,
                    "output_format": "png",
                },
            )
            resp.raise_for_status()
            return resp.content

    async def generate_video_from_image(self, image_path: Path, motion_bucket: int = 127) -> bytes:
        """Stable Video Diffusion — animates a still image."""
        with open(image_path, "rb") as f:
            image_data = f.read()

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{self.base_url}/image-to-video",
                headers=self.headers,
                data={"motion_bucket_id": motion_bucket, "image_url": ""},
                files={"image": ("image.png", image_data, "image/png")},
            )
            resp.raise_for_status()
            generation_id = resp.json().get("id")

        # Poll for result
        async with httpx.AsyncClient(timeout=30) as client:
            for _ in range(60):
                await asyncio.sleep(10)
                result = await client.get(
                    f"{self.base_url}/image-to-video/result/{generation_id}",
                    headers={**self.headers, "Accept": "video/*"},
                )
                if result.status_code == 200:
                    return result.content
                if result.status_code != 202:
                    raise RuntimeError(f"Video generation failed: {result.text}")

        raise TimeoutError("Stability video generation timed out")


class ReplicateClient:
    """Replicate API — access to Flux, SDXL, AnimateDiff, etc."""

    def __init__(self):
        self.api_key = settings.REPLICATE_API_KEY
        self.headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        }

    FLUX_MODEL = "black-forest-labs/flux-1.1-pro"
    ANIMATE_MODEL = "lucataco/animate-diff:beecf59c4764152b9d5ed388bdbef66bde5d5e3dd8e1dc5c9bf1528bd5f0c0f6"

    async def generate_image(
        self,
        prompt: str,
        negative_prompt: str,
        width: int = 1920,
        height: int = 1080,
    ) -> bytes:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"https://api.replicate.com/v1/models/{self.FLUX_MODEL}/predictions",
                headers=self.headers,
                json={
                    "input": {
                        "prompt": prompt,
                        "width": width,
                        "height": height,
                        "num_inference_steps": 28,
                        "guidance_scale": 3.5,
                    }
                },
            )
            resp.raise_for_status()
            prediction_id = resp.json()["id"]

        return await self._poll_prediction(prediction_id)

    async def _poll_prediction(self, prediction_id: str, max_wait: int = 300) -> bytes:
        async with httpx.AsyncClient(timeout=30) as client:
            for _ in range(max_wait // 5):
                await asyncio.sleep(5)
                resp = await client.get(
                    f"https://api.replicate.com/v1/predictions/{prediction_id}",
                    headers=self.headers,
                )
                data = resp.json()
                if data["status"] == "succeeded":
                    output_url = data["output"][0] if isinstance(data["output"], list) else data["output"]
                    img_resp = await client.get(output_url)
                    return img_resp.content
                if data["status"] == "failed":
                    raise RuntimeError(f"Replicate prediction failed: {data.get('error')}")
        raise TimeoutError("Replicate prediction timed out")


class VisualGeneratorService:
    """
    Generates a batch of images for a video job,
    then optionally animates them (pan/zoom or SVD).
    """

    def __init__(self):
        self.stability = StabilityClient() if settings.STABILITY_API_KEY else None
        self.replicate = ReplicateClient() if settings.REPLICATE_API_KEY else None
        self.width, self.height = settings.VIDEO_RESOLUTIONS[settings.DEFAULT_RESOLUTION]

    def _pick_provider(self) -> str:
        if self.stability:
            return "stability"
        if self.replicate:
            return "replicate"
        raise RuntimeError("No visual generation API configured. Set STABILITY_API_KEY or REPLICATE_API_KEY.")

    async def _generate_one(self, prompt_data: dict, provider: str) -> bytes:
        if provider == "stability":
            return await self.stability.generate_image(
                prompt_data["positive"],
                prompt_data["negative"],
                self.width,
                self.height,
            )
        return await self.replicate.generate_image(
            prompt_data["positive"],
            prompt_data["negative"],
            self.width,
            self.height,
        )

    async def generate_batch(
        self,
        job_id: int,
        style: MusicStyle,
        count: int = 8,
    ) -> list[int]:
        """Generate `count` images and store as VisualAssets. Returns list of asset IDs."""
        provider = self._pick_provider()
        prompts = build_visual_batch(style, count)
        asset_ids: list[int] = []

        logger.info(f"Generating {count} visuals for job {job_id} via {provider}")

        tasks = [self._generate_one(p, provider) for p in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, (result, prompt_data) in enumerate(zip(results, prompts)):
            if isinstance(result, Exception):
                logger.error(f"Visual {i} failed: {result}")
                continue

            img_path = settings.VISUALS_DIR / f"visual_{job_id}_{i:02d}.png"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            img_path.write_bytes(result)

            with get_db() as db:
                asset = VisualAsset(
                    job_id=job_id,
                    provider=provider,
                    prompt=prompt_data["positive"],
                    file_path=str(img_path),
                    asset_type="image",
                    width=self.width,
                    height=self.height,
                    generation_metadata={"subject": prompt_data.get("subject", "")},
                )
                db.add(asset)
                db.flush()
                asset_ids.append(asset.id)

        logger.info(f"Generated {len(asset_ids)}/{count} visuals for job {job_id}")
        return asset_ids

    async def animate_image(self, image_path: Path, job_id: int, index: int) -> Path:
        """Creates a 4-second animated clip from an image using Stable Video Diffusion."""
        if not self.stability:
            logger.warning("SVD not available, using FFmpeg pan/zoom instead")
            from visual_generator.animator import Animator
            return await Animator.ken_burns(image_path, duration=8)

        video_data = await self.stability.generate_video_from_image(image_path)
        out_path = settings.VISUALS_DIR / f"anim_{job_id}_{index:02d}.mp4"
        out_path.write_bytes(video_data)
        return out_path


visual_service = VisualGeneratorService()
