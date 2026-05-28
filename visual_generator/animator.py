"""
FFmpeg-based animation effects: Ken Burns (pan/zoom), parallax, glitch.
Used when Stable Video Diffusion is not available or as additional effects.
"""
import asyncio
import random
import logging
from pathlib import Path

from config.settings import settings

logger = logging.getLogger(__name__)

W, H = settings.VIDEO_RESOLUTIONS[settings.DEFAULT_RESOLUTION]
FPS = settings.VIDEO_FPS


class Animator:

    @staticmethod
    async def ken_burns(
        image_path: Path,
        duration: float = 10.0,
        zoom_start: float = 1.0,
        zoom_end: float = 1.15,
        direction: str | None = None,
    ) -> Path:
        """Smooth Ken Burns zoom + subtle pan."""
        direction = direction or random.choice(["in", "out", "left_in", "right_in"])
        out_path = image_path.with_suffix("").parent / f"{image_path.stem}_kb.mp4"

        total_frames = int(duration * FPS)

        if direction == "in":
            zoom_filter = (
                f"scale=iw*{zoom_end}:ih*{zoom_end},"
                f"zoompan=z='min(zoom+0.0015,{zoom_end})':d={total_frames}:"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"s={W}x{H}:fps={FPS}"
            )
        elif direction == "out":
            zoom_filter = (
                f"scale=iw*{zoom_end}:ih*{zoom_end},"
                f"zoompan=z='if(eq(on,1),{zoom_end},max(zoom-0.0015,1.0))':d={total_frames}:"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                f"s={W}x{H}:fps={FPS}"
            )
        else:
            # Pan + subtle zoom
            pan_x = "x='min(on/(dur*fps)*iw*0.05,iw*0.05)'" if "right" in direction else "x='max(iw*0.05-on/(dur*fps)*iw*0.05,0)'"
            zoom_filter = (
                f"scale=iw*1.1:ih*1.1,"
                f"zoompan=z='1.05':d={total_frames}:{pan_x}:"
                f"y='ih*0.025':s={W}x{H}:fps={FPS}"
            )

        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", str(image_path),
            "-vf", zoom_filter,
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            str(out_path),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Ken Burns animation failed: {stderr.decode()[-500:]}")

        return out_path

    @staticmethod
    async def glitch_effect(input_path: Path, intensity: str = "medium") -> Path:
        """VHS glitch / chromatic aberration effect overlay."""
        out_path = input_path.with_suffix("").parent / f"{input_path.stem}_glitch.mp4"

        intensity_params = {
            "low": ("0.002", "0.3"),
            "medium": ("0.005", "0.6"),
            "high": ("0.01", "1.0"),
        }
        aberration, noise = intensity_params.get(intensity, intensity_params["medium"])

        # Chromatic aberration + subtle noise
        filter_complex = (
            f"split=3[r][g][b];"
            f"[r]lutrgb=r='min(val+20,255)':g=0:b=0[rv];"
            f"[g]lutrgb=r=0:g='val':b=0[gv];"
            f"[b]lutrgb=r=0:g=0:b='min(val+20,255)'][bv];"
            f"[rv]scale=iw+4:ih[rs];"
            f"[bv]scale=iw-4:ih[bs];"
            f"[rs][gv]blend=all_mode=addition[rg];"
            f"[rg][bs]blend=all_mode=addition[out];"
            f"[out]noise=alls={noise}:allf=t"
        )

        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
            str(out_path),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.warning(f"Glitch effect failed, returning original: {stderr.decode()[-300:]}")
            return input_path

        return out_path

    @staticmethod
    async def vhs_overlay(input_path: Path) -> Path:
        """VHS scan lines + noise + color shift for retro aesthetic."""
        out_path = input_path.with_suffix("").parent / f"{input_path.stem}_vhs.mp4"

        filter_chain = (
            "noise=alls=8:allf=t,"
            "hue=s=0.85,"
            "curves=r='0/0 0.5/0.45 1/1':g='0/0 0.5/0.5 1/0.95':b='0/0.05 0.5/0.55 1/1',"
            "vignette=PI/5"
        )

        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-vf", filter_chain,
            "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
            str(out_path),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            return input_path

        return out_path

    @staticmethod
    async def color_grade_cyberpunk(input_path: Path) -> Path:
        """Apply cyberpunk color grade: teal shadows, orange/pink highlights, contrast boost."""
        out_path = input_path.with_suffix("").parent / f"{input_path.stem}_graded.mp4"

        filter_chain = (
            "curves="
            "r='0/0 0.25/0.22 0.75/0.82 1/1':"  # push reds in highlights
            "g='0/0 0.25/0.23 0.75/0.8 1/1':"   # neutral greens
            "b='0/0.03 0.25/0.28 0.75/0.8 1/0.97',"  # teal shadows, pull blue in highlights
            "vibrance=0.3,"
            "unsharp=5:5:1.0:5:5:0.0"
        )

        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-vf", filter_chain,
            "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
            str(out_path),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            return input_path

        return out_path

    @staticmethod
    async def animate_batch(
        image_paths: list[Path],
        clip_duration: float = 12.0,
        apply_grade: bool = True,
    ) -> list[Path]:
        """Animate a list of images with random Ken Burns directions."""
        directions = ["in", "out", "left_in", "right_in", "in", "out"]  # weighted toward zoom
        tasks = []
        for i, img in enumerate(image_paths):
            direction = directions[i % len(directions)]
            tasks.append(Animator.ken_burns(img, clip_duration, direction=direction))

        clips = await asyncio.gather(*tasks, return_exceptions=True)
        result = []
        for i, clip in enumerate(clips):
            if isinstance(clip, Exception):
                logger.error(f"Animation failed for image {i}: {clip}")
                continue
            if apply_grade:
                try:
                    clip = await Animator.color_grade_cyberpunk(clip)
                except Exception as e:
                    logger.warning(f"Color grade failed: {e}")
            result.append(clip)
        return result
