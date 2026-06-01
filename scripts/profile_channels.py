"""
CLI para perfilar canales con Claude.

Uso:
    # Perfilar todos los canales (los que estén en config/channels.py)
    python scripts/profile_channels.py --all

    # Un solo canal
    python scripts/profile_channels.py --channel ironpulse

    # Forzar refresh aunque haya perfil válido en caché
    python scripts/profile_channels.py --all --force

    # Modelo más barato (sólo si quieres ahorrar — calidad inferior)
    python scripts/profile_channels.py --all --model claude-haiku-4-5

El resultado se guarda en assets/channel_profiles/<channel>.json y queda
disponible automáticamente para los prompt engines de música y visuales
en la próxima generación.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.channels import list_channels
from research.channel_profiler import profile_channel
from research.client import DEFAULT_MODEL


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("profile_channels")


def _print_profile_summary(profile) -> None:
    print(f"\n{'=' * 72}")
    print(f"  PERFIL — {profile.channel.upper()}")
    print(f"  modelo: {profile.model}   tokens: {profile.tokens_used}")
    print(f"{'=' * 72}")
    print(f"\n[música] BPM target: {profile.music.bpm_target}  rango: {profile.music.bpm_range}")
    print(f"[música] artistas: {', '.join(profile.music.reference_artists)}")
    print(f"[música] labels:   {', '.join(profile.music.reference_labels)}")
    print(f"[música] elementos must-have:")
    for el in profile.music.must_have_elements:
        print(f"   • {el}")
    print(f"[música] estructura: {profile.music.structure}")
    print(f"[música] negative: {', '.join(profile.music.negative)}")
    print()
    print(f"[visual] iluminación: {profile.visual.lighting}")
    print(f"[visual] paleta: {', '.join(profile.visual.palette)}")
    print(f"[visual] composición: {profile.visual.composition}")
    print(f"[visual] sujetos:")
    for s in profile.visual.primary_subjects:
        print(f"   • {s}")
    print(f"[visual] referencias: {', '.join(profile.visual.references)}")
    print(f"[visual] avoid: {', '.join(profile.visual.avoid)}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--all", action="store_true", help="Perfilar todos los canales registrados.")
    grp.add_argument("--channel", type=str, help="Perfilar un único canal por su clave.")
    parser.add_argument("--force", action="store_true",
                        help="Ignora la caché y regenera incluso si el perfil aún no caducó.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Modelo Claude a usar (default: {DEFAULT_MODEL}).")
    args = parser.parse_args()

    targets = list_channels() if args.all else [args.channel]
    print(f"\nPerfilando {len(targets)} canal(es) con {args.model}\n")

    total_tokens = 0
    for ch in targets:
        try:
            profile = profile_channel(ch, force_refresh=args.force, model=args.model)
            _print_profile_summary(profile)
            total_tokens += profile.tokens_used
        except Exception as exc:
            logger.error(f"Fallo perfilando {ch}: {exc}")

    print(f"{'=' * 72}")
    print(f"  TOTAL — {len(targets)} canal(es)  tokens consumidos: {total_tokens}")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
