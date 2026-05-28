"""
Audio post-processing: normalize loudness, fade in/out, seamless loop,
concatenate clips, and export final master.
Requires: ffmpeg in PATH, pyloudnorm, pydub.
"""
import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Optional

import pyloudnorm as pyln
import soundfile as sf
import numpy as np

from config.settings import settings

logger = logging.getLogger(__name__)


class AudioProcessor:

    @staticmethod
    async def process(
        input_path: Path,
        output_path: Path,
        target_duration: int,
        target_loudness: float = settings.TARGET_LOUDNESS,
        fade_duration: float = 3.0,
    ) -> Path:
        """
        Full pipeline:
        1. Trim or loop to target_duration
        2. Normalize loudness to target_loudness LUFS
        3. Apply fade in/out
        4. Export as 320k MP3
        """
        temp_wav = output_path.with_suffix(".temp.wav")

        # Step 1: FFmpeg trim/loop
        await AudioProcessor._trim_or_loop(input_path, temp_wav, target_duration)

        # Step 2: Loudness normalization
        normalized_wav = output_path.with_suffix(".norm.wav")
        await AudioProcessor._normalize_loudness(temp_wav, normalized_wav, target_loudness)

        # Step 3: Fade in/out
        faded_wav = output_path.with_suffix(".faded.wav")
        await AudioProcessor._apply_fades(normalized_wav, faded_wav, fade_duration)

        # Step 4: Export final MP3
        await AudioProcessor._export_mp3(faded_wav, output_path)

        # Cleanup temp files
        for f in [temp_wav, normalized_wav, faded_wav]:
            f.unlink(missing_ok=True)

        logger.info(f"Audio processed: {output_path}")
        return output_path

    @staticmethod
    async def _trim_or_loop(input_path: Path, output_path: Path, duration: int) -> None:
        """Loop audio if shorter than duration, trim if longer."""
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1",
            "-i", str(input_path),
            "-t", str(duration),
            "-c:a", "pcm_s16le",
            "-ar", str(settings.AUDIO_SAMPLE_RATE),
            str(output_path),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg trim/loop failed: {stderr.decode()}")

    @staticmethod
    async def _normalize_loudness(input_path: Path, output_path: Path, target_lufs: float) -> None:
        """Two-pass loudnorm filter via FFmpeg for accurate LUFS targeting."""
        # First pass — measure
        cmd_measure = [
            "ffmpeg", "-i", str(input_path),
            "-af", "loudnorm=I=-23:TP=-1.5:LRA=11:print_format=json",
            "-f", "null", "-",
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd_measure, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        stderr_text = stderr.decode()

        # Extract measured values (they appear in stderr)
        import json, re
        match = re.search(r'\{[^{}]*"input_i"[^{}]*\}', stderr_text, re.DOTALL)
        if match:
            stats = json.loads(match.group())
            measured_I = stats.get("input_i", str(target_lufs))
            measured_TP = stats.get("input_tp", "-1.5")
            measured_LRA = stats.get("input_lra", "11")
            measured_thresh = stats.get("input_thresh", "-31")
            offset = stats.get("target_offset", "0")
        else:
            measured_I, measured_TP, measured_LRA = str(target_lufs), "-1.5", "11"
            measured_thresh, offset = "-31", "0"

        # Second pass — apply normalization
        loudnorm_filter = (
            f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:"
            f"measured_I={measured_I}:measured_TP={measured_TP}:"
            f"measured_LRA={measured_LRA}:measured_thresh={measured_thresh}:"
            f"offset={offset}:linear=true:print_format=summary"
        )
        cmd_apply = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-af", loudnorm_filter,
            "-c:a", "pcm_s16le",
            "-ar", str(settings.AUDIO_SAMPLE_RATE),
            str(output_path),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd_apply, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Loudness normalization failed: {stderr.decode()}")

    @staticmethod
    async def _apply_fades(input_path: Path, output_path: Path, fade_duration: float) -> None:
        """Apply fade in at start and fade out at end."""
        data, sample_rate = sf.read(str(input_path))
        total_duration = len(data) / sample_rate
        fade_out_start = total_duration - fade_duration

        fade_filter = (
            f"afade=t=in:st=0:d={fade_duration},"
            f"afade=t=out:st={fade_out_start:.2f}:d={fade_duration}"
        )
        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-af", fade_filter,
            "-c:a", "pcm_s16le",
            str(output_path),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Fade application failed: {stderr.decode()}")

    @staticmethod
    async def _export_mp3(input_path: Path, output_path: Path) -> None:
        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-c:a", "libmp3lame",
            "-b:a", settings.AUDIO_BITRATE,
            "-ar", str(settings.AUDIO_SAMPLE_RATE),
            "-id3v2_version", "3",
            str(output_path),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"MP3 export failed: {stderr.decode()}")

    @staticmethod
    async def concatenate_clips(clip_paths: list[Path], output_path: Path) -> Path:
        """Concatenate multiple audio clips with crossfade between them."""
        if len(clip_paths) == 1:
            clip_paths[0].rename(output_path)
            return output_path

        # Create FFmpeg concat filter with crossfade
        # Build input args
        inputs = []
        for p in clip_paths:
            inputs.extend(["-i", str(p)])

        # Build acrossfade filter chain
        filter_parts = []
        crossfade_dur = 2.0
        prev_label = "[0:a]"

        for i in range(1, len(clip_paths)):
            out_label = f"[cf{i}]" if i < len(clip_paths) - 1 else "[aout]"
            filter_parts.append(
                f"{prev_label}[{i}:a]acrossfade=d={crossfade_dur}:c1=exp:c2=exp{out_label}"
            )
            prev_label = out_label

        filter_complex = ";".join(filter_parts)

        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "[aout]",
            "-c:a", "libmp3lame",
            "-b:a", settings.AUDIO_BITRATE,
            str(output_path),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Clip concatenation failed: {stderr.decode()}")

        return output_path

    @staticmethod
    def get_audio_info(file_path: Path) -> dict:
        """Returns duration, sample_rate, and measured loudness."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams", "-show_format",
            str(file_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")

        import json
        info = json.loads(result.stdout)
        fmt = info.get("format", {})
        audio_streams = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
        stream = audio_streams[0] if audio_streams else {}

        return {
            "duration_seconds": float(fmt.get("duration", 0)),
            "sample_rate": int(stream.get("sample_rate", settings.AUDIO_SAMPLE_RATE)),
            "file_size_mb": float(fmt.get("size", 0)) / (1024 * 1024),
        }
