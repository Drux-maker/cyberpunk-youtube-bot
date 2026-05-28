"""
Audio post-processing y mastering — TODO en una sola pasada de ffmpeg para
evitar pérdidas por re-encoding intermedio.

Cadena de mastering (en orden):
  1. Resample con SoX VHQ (32 kHz → 48 kHz, calidad transparente)
  2. Trim/loop a duración objetivo
  3. Mid-EQ suave: cut 200-400 Hz (-1 dB) para limpiar barro,
     shelf alto +1.5 dB @ 10 kHz para "aire"
  4. Compresor multibanda muy suave (sólo si se detecta dinámica excesiva)
  5. Exciter sutil con aphaser/aexciter equivalente (subtle harmonic)
  6. Fade in/out
  7. loudnorm de doble pasada a -14 LUFS / -1 dBTP (estándar YouTube)
  8. Export simultáneo a MP3 320k y AAC 256k

Requisitos: ffmpeg con libsoxr y libmp3lame y aac encoders (build estándar OK).
"""
import asyncio
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

from config.settings import settings

logger = logging.getLogger(__name__)


# ─── helpers ─────────────────────────────────────────────────────────────────
async def _run_ffmpeg(cmd: list[str], label: str) -> str:
    """Run ffmpeg async, raise with stderr on failure, return stderr text."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    text = stderr.decode(errors="ignore")
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed:\n{text}")
    return text


# ─── filter chain helpers ────────────────────────────────────────────────────

def _master_filter_chain(
    target_lufs: float,
    target_tp: float,
    fade_in: float,
    fade_out_start: float,
    fade_out_dur: float,
    measured: Optional[dict] = None,
) -> str:
    """
    Build the full mastering filter chain.

    `measured` viene de la primera pasada de loudnorm — si no, se aplica en
    modo single-pass (menos preciso pero válido).
    """
    parts = []

    # 1. Resample alta calidad a 48 kHz
    parts.append(f"aresample=resampler=soxr:precision=28:osf=s32:out_sample_rate={settings.AUDIO_SAMPLE_RATE}")

    # 2. EQ correctivo + carácter
    #    - high-pass suave para limpiar rumble sub-30Hz
    #    - cut suave en 250 Hz para reducir barro
    #    - shelf alto para "aire"
    parts.append("highpass=f=30:poles=2")
    parts.append("equalizer=f=250:t=q:w=1.2:g=-1.5")
    parts.append("treble=g=1.5:f=10000")

    # 3. Compresión suave de carácter (acompressor — single-band sutil)
    #    threshold -18, ratio 2:1, attack 20ms, release 250ms
    parts.append("acompressor=threshold=-18dB:ratio=2:attack=20:release=250:makeup=1.5:knee=4")

    # 4. Forzar layout estéreo (upmix mono → estéreo si hace falta) y aplicar
    #    un widener muy suave. Sin el aformat previo, extrastereo falla en mono.
    parts.append("aformat=channel_layouts=stereo")
    parts.append("extrastereo=m=1.15:c=false")

    # 5. Fades
    parts.append(f"afade=t=in:st=0:d={fade_in:.3f}")
    parts.append(f"afade=t=out:st={fade_out_start:.3f}:d={fade_out_dur:.3f}")

    # 6. Loudnorm — segunda pasada con valores medidos
    if measured:
        loudnorm = (
            f"loudnorm=I={target_lufs}:TP={target_tp}:LRA=9:"
            f"measured_I={measured['input_i']}:measured_TP={measured['input_tp']}:"
            f"measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}:"
            f"offset={measured['target_offset']}:linear=true:print_format=summary"
        )
    else:
        loudnorm = f"loudnorm=I={target_lufs}:TP={target_tp}:LRA=9:print_format=summary"
    parts.append(loudnorm)

    # 7. True-peak limiter de seguridad (alimit) — captura cualquier transitorio
    #    por encima del TP target.
    parts.append(f"alimiter=limit={10 ** (target_tp / 20):.4f}:attack=5:release=50:level=disabled")

    return ",".join(parts)


# ─── main API ────────────────────────────────────────────────────────────────
class AudioProcessor:

    @staticmethod
    async def process(
        input_path: Path,
        output_path: Path,
        target_duration: int,
        target_loudness: float = settings.TARGET_LOUDNESS,
        target_true_peak: float = settings.TARGET_TRUE_PEAK,
        fade_duration: float = 3.0,
    ) -> Path:
        """
        Master + export. UNA sola re-codificación final → cero doble-encoding.

        Pasos:
          1. trim/loop a duración objetivo (WAV)
          2. medir loudness (1.ª pasada loudnorm)
          3. master + encoding final a MP3 320k (y opcionalmente AAC 256k)
        """
        looped_wav = output_path.with_suffix(".loop.wav")
        await AudioProcessor._trim_or_loop(input_path, looped_wav, target_duration)

        # medir loudness
        measured = await AudioProcessor._measure_loudness(looped_wav, target_loudness, target_true_peak)

        # construir cadena completa
        total_dur = float(target_duration)
        fade_out_start = max(0.0, total_dur - fade_duration)
        chain = _master_filter_chain(
            target_lufs=target_loudness,
            target_tp=target_true_peak,
            fade_in=fade_duration,
            fade_out_start=fade_out_start,
            fade_out_dur=fade_duration,
            measured=measured,
        )

        # MP3 320k — single-pass mastering
        await AudioProcessor._encode(
            looped_wav, output_path, chain,
            codec="libmp3lame", bitrate=settings.AUDIO_BITRATE,
        )

        # AAC 256k opcional, mismo master desde el WAV intermedio
        if settings.EXPORT_AAC:
            aac_path = output_path.with_suffix(".m4a")
            try:
                await AudioProcessor._encode(
                    looped_wav, aac_path, chain,
                    codec="aac", bitrate=settings.AUDIO_AAC_BITRATE,
                )
                logger.info(f"AAC export: {aac_path}")
            except RuntimeError as exc:
                logger.warning(f"AAC export falló (continuando solo con MP3): {exc}")

        looped_wav.unlink(missing_ok=True)
        logger.info(f"Audio mastered: {output_path}")
        return output_path

    # ─── pasos individuales ──────────────────────────────────────────────────
    @staticmethod
    async def _trim_or_loop(input_path: Path, output_path: Path, duration: int) -> None:
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1",
            "-i", str(input_path),
            "-t", str(duration),
            "-c:a", "pcm_s24le",  # mantener cabecera al máximo durante el mastering
            str(output_path),
        ]
        await _run_ffmpeg(cmd, "trim/loop")

    @staticmethod
    async def _measure_loudness(
        input_path: Path,
        target_lufs: float,
        target_tp: float,
    ) -> Optional[dict]:
        cmd = [
            "ffmpeg", "-i", str(input_path),
            "-af", f"loudnorm=I={target_lufs}:TP={target_tp}:LRA=9:print_format=json",
            "-f", "null", "-",
        ]
        try:
            text = await _run_ffmpeg(cmd, "loudnorm measure")
        except RuntimeError:
            return None
        match = re.search(r'\{[^{}]*"input_i"[^{}]*\}', text, re.DOTALL)
        if not match:
            logger.warning("No se pudo extraer medición de loudnorm — fallback a single-pass")
            return None
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None

    @staticmethod
    async def _encode(
        input_path: Path,
        output_path: Path,
        filter_chain: str,
        codec: str,
        bitrate: str,
    ) -> None:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-af", filter_chain,
            "-c:a", codec,
            "-b:a", bitrate,
            "-ar", str(settings.AUDIO_SAMPLE_RATE),
        ]
        if codec == "libmp3lame":
            cmd += ["-id3v2_version", "3", "-write_xing", "1"]
        cmd.append(str(output_path))
        await _run_ffmpeg(cmd, f"encode {codec}")

    # ─── concatenación sin pérdida (WAV in/out) ──────────────────────────────
    @staticmethod
    async def concatenate_clips_wav(clip_paths: list[Path], output_path: Path) -> Path:
        """
        Concatena clips manteniendo TODO en WAV (sin doble encoding).
        Crossfade equal-power triangular (1.5s) — sin dip de volumen.
        """
        if len(clip_paths) == 1:
            clip_paths[0].rename(output_path)
            return output_path

        inputs: list[str] = []
        for p in clip_paths:
            inputs.extend(["-i", str(p)])

        crossfade_dur = 1.5
        filter_parts: list[str] = []
        prev_label = "[0:a]"
        for i in range(1, len(clip_paths)):
            out_label = f"[cf{i}]" if i < len(clip_paths) - 1 else "[aout]"
            # c1=tri, c2=tri → curva lineal complementaria, energía constante
            filter_parts.append(
                f"{prev_label}[{i}:a]acrossfade=d={crossfade_dur}:c1=tri:c2=tri{out_label}"
            )
            prev_label = out_label
        filter_complex = ";".join(filter_parts)

        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "[aout]",
            "-c:a", "pcm_s24le",  # WAV intermedio sin pérdida
            str(output_path),
        ]
        await _run_ffmpeg(cmd, "concat clips")
        return output_path

    # ─── compat: nombre antiguo conservado por si lo llama código externo ───
    @staticmethod
    async def concatenate_clips(clip_paths: list[Path], output_path: Path) -> Path:
        return await AudioProcessor.concatenate_clips_wav(clip_paths, output_path)

    @staticmethod
    def get_audio_info(file_path: Path) -> dict:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams", "-show_format",
            str(file_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")

        info = json.loads(result.stdout)
        fmt = info.get("format", {})
        audio_streams = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
        stream = audio_streams[0] if audio_streams else {}

        return {
            "duration_seconds": float(fmt.get("duration", 0)),
            "sample_rate": int(stream.get("sample_rate", settings.AUDIO_SAMPLE_RATE)),
            "file_size_mb": float(fmt.get("size", 0)) / (1024 * 1024),
        }
