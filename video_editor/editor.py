"""
Assembles the final video: animated visuals + audio + effects.
Uses FFmpeg for all heavy lifting (no MoviePy dependency issues).
"""
import asyncio
import logging
import tempfile
from pathlib import Path

from config.settings import settings
from database.models import VideoAsset
from database.db import get_db
from visual_generator.animator import Animator

logger = logging.getLogger(__name__)

W, H = settings.VIDEO_RESOLUTIONS[settings.DEFAULT_RESOLUTION]
FPS = settings.VIDEO_FPS


class VideoEditor:

    async def assemble(
        self,
        job_id: int,
        image_paths: list[Path],
        audio_path: Path,
        duration_seconds: int,
        apply_effects: bool = True,
    ) -> Path:
        """
        Full assembly pipeline:
        1. Animate each image with Ken Burns
        2. Concatenate animated clips in a seamless loop
        3. Mux with audio
        4. Apply final color grade + optional effects
        5. Export final MP4
        """
        output_path = settings.VIDEOS_DIR / f"video_{job_id}.mp4"
        settings.VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

        # Step 1: Calculate clip duration per image
        n_images = len(image_paths)
        if n_images == 0:
            raise ValueError("No images provided for video assembly")

        clip_duration = max(8.0, duration_seconds / n_images)
        # If a single clip covers the whole duration, just one image is fine
        clips_needed = max(1, -(-duration_seconds // int(clip_duration)))  # ceiling

        # If we have fewer images than clips needed, cycle through them
        extended_images = []
        for i in range(clips_needed):
            extended_images.append(image_paths[i % n_images])

        logger.info(f"Animating {len(extended_images)} clips for job {job_id}")

        # Step 2: Animate images
        animated_clips = await Animator.animate_batch(
            extended_images,
            clip_duration=clip_duration,
            apply_grade=apply_effects,
        )

        if not animated_clips:
            raise RuntimeError("All animations failed")

        # Step 3: Concatenate video clips
        concat_path = settings.VIDEOS_DIR / f"concat_{job_id}.mp4"
        await self._concatenate_clips(animated_clips, concat_path, total_duration=duration_seconds)

        # Step 4: Mux with audio
        muxed_path = settings.VIDEOS_DIR / f"muxed_{job_id}.mp4"
        await self._mux_audio(concat_path, audio_path, muxed_path, duration_seconds)

        # Step 5: Final encode with metadata
        await self._final_encode(muxed_path, output_path)

        # Cleanup intermediates
        for f in [concat_path, muxed_path]:
            f.unlink(missing_ok=True)
        for clip in animated_clips:
            clip.unlink(missing_ok=True)

        # Save VideoAsset
        with get_db() as db:
            asset = VideoAsset(
                job_id=job_id,
                file_path=str(output_path),
                resolution=settings.DEFAULT_RESOLUTION,
                fps=FPS,
                duration_seconds=float(duration_seconds),
                effects_applied={"ken_burns": True, "color_grade": apply_effects},
            )
            db.add(asset)

        logger.info(f"Video assembled: {output_path}")
        return output_path

    async def _concatenate_clips(
        self, clip_paths: list[Path], output_path: Path, total_duration: int
    ) -> None:
        """Concatenate with seamless transitions, loop until total_duration."""
        concat_list = output_path.parent / f"concat_list_{output_path.stem}.txt"
        with open(concat_list, "w") as f:
            for clip in clip_paths:
                f.write(f"file '{clip.resolve()}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-t", str(total_duration),
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        concat_list.unlink(missing_ok=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Clip concatenation failed: {stderr.decode()[-500:]}")

    async def _mux_audio(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
        duration: int,
    ) -> None:
        """Combine video and audio tracks, loop audio if needed."""
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-stream_loop", "-1", "-i", str(audio_path),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-t", str(duration),
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "320k",
            "-shortest",
            str(output_path),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Audio mux failed: {stderr.decode()[-500:]}")

    async def _final_encode(self, input_path: Path, output_path: Path) -> None:
        """Final H.264 encode optimized for YouTube streaming."""
        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-b:v", settings.VIDEO_BITRATE,
            "-maxrate", "10000k",
            "-bufsize", "20000k",
            "-profile:v", "high",
            "-level:v", "4.2",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", settings.AUDIO_BITRATE,
            "-ar", str(settings.AUDIO_SAMPLE_RATE),
            "-movflags", "+faststart",
            "-metadata", "encoder=CyberpunkBot",
            str(output_path),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Final encode failed: {stderr.decode()[-500:]}")

    async def add_intro_card(
        self,
        video_path: Path,
        title: str,
        duration: float = 3.0,
    ) -> Path:
        """Prepend a 3-second fade-in title card."""
        out_path = video_path.with_stem(video_path.stem + "_intro")

        # Generate black frame with title overlay
        title_filter = (
            f"color=black:s={W}x{H}:d={duration}[bg];"
            f"[bg]drawtext=text='{title}':fontcolor=white:fontsize=72:"
            f"x=(w-text_w)/2:y=(h-text_h)/2:alpha='if(lt(t,0.5),t/0.5,if(gt(t,{duration-0.5}),(t-{duration})/0.5,1))'[card];"
            f"[card][0:v]concat=n=2:v=1:a=0[v]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=black:s={W}x{H}:d={duration}",
            "-i", str(video_path),
            "-filter_complex", title_filter,
            "-map", "[v]", "-map", "1:a",
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "copy",
            str(out_path),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.warning(f"Intro card failed: {stderr.decode()[-300:]}")
            return video_path

        return out_path


video_editor = VideoEditor()
