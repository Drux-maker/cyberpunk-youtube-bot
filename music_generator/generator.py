"""
Music generation via Suno AI, Udio, or fallback to MusicGen (local HuggingFace).
Handles polling, retries, and storing results.
"""
import asyncio
import httpx
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from config.settings import settings
from database.models import MusicAsset, MusicStyle, VideoJob
from database.db import get_db
from music_generator.prompt_engine import build_music_prompt_variations

logger = logging.getLogger(__name__)


class SunoClient:
    """Unofficial Suno API client (via suno-api proxy or official when available)."""

    def __init__(self):
        self.api_key = settings.SUNO_API_KEY
        self.base_url = settings.SUNO_API_BASE
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def generate(self, prompt: str, duration: int = 240) -> dict:
        """Submit generation job and return task ID."""
        async with httpx.AsyncClient(timeout=60) as client:
            payload = {
                "prompt": prompt,
                "make_instrumental": True,
                "wait_audio": False,
                "duration": min(duration, 240),  # Suno max ~4min per clip
            }
            resp = await client.post(
                f"{self.base_url}/generate",
                json=payload,
                headers=self.headers,
            )
            resp.raise_for_status()
            return resp.json()

    async def poll_status(self, task_id: str, max_wait: int = 300) -> dict:
        """Poll until generation completes or times out."""
        async with httpx.AsyncClient(timeout=30) as client:
            start = time.time()
            while time.time() - start < max_wait:
                resp = await client.get(
                    f"{self.base_url}/get?ids={task_id}",
                    headers=self.headers,
                )
                resp.raise_for_status()
                data = resp.json()
                clips = data if isinstance(data, list) else data.get("data", [])
                if clips and clips[0].get("status") == "complete":
                    return clips[0]
                await asyncio.sleep(10)
            raise TimeoutError(f"Suno generation timed out after {max_wait}s")

    async def download_audio(self, audio_url: str, dest_path: Path) -> Path:
        async with httpx.AsyncClient(timeout=300, follow_redirects=True) as client:
            resp = await client.get(audio_url)
            resp.raise_for_status()
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(resp.content)
        return dest_path


class UdioClient:
    """Udio API client."""

    def __init__(self):
        self.api_key = settings.UDIO_API_KEY
        self.base_url = settings.UDIO_API_BASE
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def generate(self, prompt: str) -> dict:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{self.base_url}/generate-proxy",
                json={
                    "prompt": prompt,
                    "samplerOptions": {"seed": -1},
                },
                headers=self.headers,
            )
            resp.raise_for_status()
            return resp.json()

    async def poll_status(self, track_id: str, max_wait: int = 300) -> dict:
        async with httpx.AsyncClient(timeout=30) as client:
            start = time.time()
            while time.time() - start < max_wait:
                resp = await client.get(
                    f"{self.base_url}/get-track-by-id?trackId={track_id}",
                    headers=self.headers,
                )
                resp.raise_for_status()
                data = resp.json()
                if data.get("track", {}).get("finished"):
                    return data["track"]
                await asyncio.sleep(10)
            raise TimeoutError("Udio generation timed out")

    async def download_audio(self, audio_url: str, dest_path: Path) -> Path:
        async with httpx.AsyncClient(timeout=300, follow_redirects=True) as client:
            resp = await client.get(audio_url)
            resp.raise_for_status()
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_bytes(resp.content)
        return dest_path


class MusicGeneratorService:
    """
    Orchestrates music generation:
    1. Picks best available provider (Suno > Udio > MusicGen)
    2. Generates clips of max provider duration
    3. Concatenates clips to reach target duration
    4. Stores MusicAsset in DB
    """

    MAX_CLIP_DURATION = 240  # seconds per clip (Suno limit)

    def __init__(self):
        self.suno = SunoClient() if settings.SUNO_API_KEY else None
        self.udio = UdioClient() if settings.UDIO_API_KEY else None

    def _clips_needed(self, duration_seconds: int) -> int:
        return max(1, -(-duration_seconds // self.MAX_CLIP_DURATION))  # ceiling division

    async def _generate_single_clip(self, prompt: str, provider: str) -> tuple[str, dict]:
        """Returns (audio_url, metadata)."""
        if provider == "suno" and self.suno:
            task = await self.suno.generate(prompt, self.MAX_CLIP_DURATION)
            task_id = task[0]["id"] if isinstance(task, list) else task.get("id")
            clip = await self.suno.poll_status(task_id)
            return clip["audio_url"], {"provider": "suno", "clip_id": task_id, "title": clip.get("title", "")}

        if provider == "udio" and self.udio:
            task = await self.udio.generate(prompt)
            track_id = task.get("track_ids", [None])[0]
            clip = await self.udio.poll_status(track_id)
            return clip.get("song_path", clip.get("audio_url")), {"provider": "udio", "clip_id": track_id}

        raise ValueError(f"Provider '{provider}' unavailable or not configured")

    async def generate(
        self,
        job_id: int,
        style: MusicStyle,
        duration_seconds: int,
        prompt: str,
        bpm: int,
    ) -> MusicAsset:
        provider = "suno" if self.suno else ("udio" if self.udio else None)
        if not provider:
            raise RuntimeError("No music generation API configured. Set SUNO_API_KEY or UDIO_API_KEY.")

        clips_needed = self._clips_needed(duration_seconds)
        clip_paths: list[Path] = []
        generation_metadata = {"clips": [], "provider": provider}

        logger.info(f"Generating {clips_needed} clips for job {job_id} via {provider}")

        for i in range(clips_needed):
            clip_filename = settings.MUSIC_DIR / f"clip_{job_id}_{i}.mp3"
            audio_url, clip_meta = await self._generate_single_clip(prompt, provider)
            if provider == "suno":
                await self.suno.download_audio(audio_url, clip_filename)
            else:
                await self.udio.download_audio(audio_url, clip_filename)
            clip_paths.append(clip_filename)
            generation_metadata["clips"].append(clip_meta)
            logger.info(f"  Clip {i+1}/{clips_needed} downloaded: {clip_filename}")

        # Concatenate clips if more than one
        final_path = settings.MUSIC_DIR / f"music_raw_{job_id}.mp3"
        if len(clip_paths) == 1:
            clip_paths[0].rename(final_path)
        else:
            from music_generator.audio_processor import AudioProcessor
            final_path = await AudioProcessor.concatenate_clips(clip_paths, final_path)
            for p in clip_paths:
                p.unlink(missing_ok=True)

        with get_db() as db:
            asset = MusicAsset(
                job_id=job_id,
                provider=provider,
                prompt=prompt,
                file_path=str(final_path),
                bpm=float(bpm),
                generation_metadata=generation_metadata,
            )
            db.add(asset)
            db.flush()
            asset_id = asset.id

        logger.info(f"MusicAsset {asset_id} saved for job {job_id}")
        return asset_id

    async def generate_with_retry(
        self,
        job_id: int,
        style: MusicStyle,
        duration_seconds: int,
    ) -> int:
        """Tries multiple prompt variations before failing."""
        variations = build_music_prompt_variations(style, duration_seconds, count=3)
        last_error = None

        for attempt, variation in enumerate(variations):
            try:
                logger.info(f"Music generation attempt {attempt + 1}/3 for job {job_id}")
                asset_id = await self.generate(
                    job_id=job_id,
                    style=style,
                    duration_seconds=duration_seconds,
                    prompt=variation["prompt"],
                    bpm=variation["bpm"],
                )
                return asset_id
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(30)

        raise RuntimeError(f"All music generation attempts failed. Last error: {last_error}")


music_service = MusicGeneratorService()
