"""
Generates 5 thumbnail variants per video optimized for CTR.
Uses Pillow for composition + a base image from the visual generator.
Rules: high contrast, short bold text, neon glow, consistent branding.
"""
import random
import logging
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageEnhance
import numpy as np

from config.settings import settings
from database.models import ThumbnailAsset, MusicStyle
from database.db import get_db

logger = logging.getLogger(__name__)

THUMB_W, THUMB_H = 1280, 720

# Text variants per style
STYLE_TEXT = {
    MusicStyle.DARK_TECHNO: [
        ("DARK TECHNO", "AI MIX"),
        ("UNDERGROUND", "TECHNO 2026"),
        ("TECHNO BUNKER", "1H NON STOP"),
        ("HARD TECHNO", "AI GENERATED"),
        ("DARK RAVE", "HYPNOTIC MIX"),
    ],
    MusicStyle.CYBERPUNK: [
        ("CYBERPUNK", "MIX 2026"),
        ("NEON CITY", "SOUNDTRACK"),
        ("CYBER BEATS", "AI MUSIC"),
        ("FUTURE SOUND", "1H MIX"),
        ("DARK CYBER", "ATMOSPHERE"),
    ],
    MusicStyle.NEON_AMBIENT: [
        ("NEON AMBIENT", "FOCUS MUSIC"),
        ("DEEP FOCUS", "NEON CITY"),
        ("STUDY MUSIC", "AI AMBIENT"),
        ("CITY DREAMS", "1H AMBIENT"),
        ("NEON DRIFT", "CONCENTRATE"),
    ],
    MusicStyle.SYNTHWAVE: [
        ("SYNTHWAVE", "RETROWAVE"),
        ("NEON 80S", "AI MIX"),
        ("RETRO FUTURE", "DRIVE MUSIC"),
        ("OUTRUN", "1H MIX"),
        ("CYBER 80S", "NIGHTDRIVE"),
    ],
    MusicStyle.INDUSTRIAL: [
        ("INDUSTRIAL", "TECHNO"),
        ("NOISE TECHNO", "WASTELAND"),
        ("HARSH TECHNO", "AI MIX"),
        ("DYSTOPIA", "1H NOISE"),
        ("APOCALYPSE", "TECHNO"),
    ],
    MusicStyle.ACID_TECHNO: [
        ("ACID TECHNO", "303 MIX"),
        ("ACID RAVE", "AI MUSIC"),
        ("HYPNOTIC 303", "LOOP"),
        ("UNDERGROUND", "ACID 1H"),
        ("RAW ACID", "RAVE MIX"),
    ],
    MusicStyle.HARDTEK: [
        ("HARDTEK", "FREE PARTY"),
        ("TEK TEK", "AI MUSIC"),
        ("PSYTEK", "1H RAVE"),
        ("HARD 170BPM", "TEKNO"),
        ("FREE PARTY", "HARDTEK"),
    ],
}

NEON_COLORS = [
    (0, 255, 255),    # Cyan
    (255, 0, 255),    # Magenta
    (0, 255, 128),    # Neon green
    (255, 64, 0),     # Orange-red
    (128, 0, 255),    # Purple
    (255, 255, 0),    # Yellow
]


class ThumbnailGenerator:

    def __init__(self):
        self._font_cache = {}

    def _get_font(self, size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
        cache_key = (size, bold)
        if cache_key not in self._font_cache:
            font_paths = [
                settings.FONTS_DIR / "BebasNeue-Regular.ttf",
                settings.FONTS_DIR / "Orbitron-Bold.ttf",
                settings.FONTS_DIR / "RobotoCondensed-Bold.ttf",
                # System fallbacks
                Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
                Path("C:/Windows/Fonts/impact.ttf"),
                Path("C:/Windows/Fonts/arial.ttf"),
            ]
            font = None
            for p in font_paths:
                if p.exists():
                    try:
                        font = ImageFont.truetype(str(p), size)
                        break
                    except Exception:
                        continue
            if font is None:
                font = ImageFont.load_default()
            self._font_cache[cache_key] = font
        return self._font_cache[cache_key]

    def _add_neon_glow(self, img: Image.Image, draw: ImageDraw.Draw, text: str,
                       x: int, y: int, font: ImageFont.FreeTypeFont,
                       color: tuple, glow_radius: int = 8) -> None:
        """Draw text with neon glow effect."""
        # Glow layers (multiple semi-transparent passes)
        glow_color = (*color, 80)
        for radius in range(glow_radius, 0, -2):
            for dx in range(-radius, radius + 1, 2):
                for dy in range(-radius, radius + 1, 2):
                    if dx * dx + dy * dy <= radius * radius:
                        draw.text((x + dx, y + dy), text, font=font, fill=glow_color)
        # Solid text on top
        draw.text((x, y), text, font=font, fill=color)

    def _compose_variant(
        self,
        base_image: Image.Image,
        title_top: str,
        title_bottom: str,
        variant_idx: int,
        duration_label: str,
    ) -> Image.Image:
        img = base_image.copy().resize((THUMB_W, THUMB_H), Image.LANCZOS)

        # Boost contrast + saturation
        img = ImageEnhance.Contrast(img).enhance(1.3)
        img = ImageEnhance.Color(img).enhance(1.4)

        # Dark gradient overlay at bottom for text readability
        gradient = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
        grad_draw = ImageDraw.Draw(gradient)
        for y in range(THUMB_H):
            alpha = int(200 * max(0, (y - THUMB_H * 0.45) / (THUMB_H * 0.55)))
            grad_draw.line([(0, y), (THUMB_W, y)], fill=(0, 0, 0, alpha))

        # Also top gradient
        for y in range(int(THUMB_H * 0.25)):
            alpha = int(150 * (1 - y / (THUMB_H * 0.25)))
            grad_draw.line([(0, y), (THUMB_W, y)], fill=(0, 0, 0, alpha))

        img = Image.alpha_composite(img.convert("RGBA"), gradient)
        draw = ImageDraw.Draw(img)

        neon_color = NEON_COLORS[variant_idx % len(NEON_COLORS)]

        # Top label (smaller, neon accent)
        font_small = self._get_font(48)
        draw.text((30, 25), duration_label, font=font_small, fill=neon_color)

        # Bottom text: large title
        font_big = self._get_font(110)
        font_med = self._get_font(72)

        # Top line of title
        w_top = draw.textlength(title_top, font=font_big)
        x_top = (THUMB_W - w_top) / 2
        self._add_neon_glow(img, draw, title_top, int(x_top), THUMB_H - 220, font_big, (255, 255, 255), glow_radius=6)

        # Bottom line of title (neon color)
        w_bot = draw.textlength(title_bottom, font=font_med)
        x_bot = (THUMB_W - w_bot) / 2
        self._add_neon_glow(img, draw, title_bottom, int(x_bot), THUMB_H - 110, font_med, neon_color, glow_radius=10)

        # Neon border accent lines
        border_color = (*neon_color, 180)
        draw.rectangle([(0, 0), (THUMB_W - 1, THUMB_H - 1)], outline=(*neon_color, 120), width=4)
        draw.line([(0, THUMB_H - 135), (THUMB_W, THUMB_H - 135)], fill=(*neon_color, 100), width=2)

        # "AI GENERATED" badge top-right
        font_badge = self._get_font(30)
        badge_text = "⬡ AI GENERATED"
        badge_w = draw.textlength(badge_text, font=font_badge)
        draw.text((THUMB_W - badge_w - 20, 25), badge_text, font=font_badge, fill=(*neon_color, 200))

        return img.convert("RGB")

    def generate_variants(
        self,
        job_id: int,
        style: MusicStyle,
        base_image_path: Path,
        duration_seconds: int,
    ) -> list[Path]:
        duration_label = {
            600: "10 MIN", 1800: "30 MIN", 3600: "1 HOUR", 7200: "2 HOURS"
        }.get(duration_seconds, f"{duration_seconds // 60} MIN")

        texts = STYLE_TEXT.get(style, STYLE_TEXT[MusicStyle.CYBERPUNK])
        base_img = Image.open(base_image_path)

        output_paths = []
        settings.THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)

        with get_db() as db:
            for i, (top_text, bottom_text) in enumerate(texts[:5]):
                thumb = self._compose_variant(base_img, top_text, bottom_text, i, duration_label)
                out_path = settings.THUMBNAILS_DIR / f"thumb_{job_id}_{i:02d}.jpg"
                thumb.save(str(out_path), "JPEG", quality=95, optimize=True)
                output_paths.append(out_path)

                asset = ThumbnailAsset(
                    job_id=job_id,
                    file_path=str(out_path),
                    variant_index=i,
                )
                db.add(asset)

        logger.info(f"Generated {len(output_paths)} thumbnails for job {job_id}")
        return output_paths

    def select_best(self, job_id: int, thumbnail_paths: list[Path]) -> Path:
        """
        Heuristic CTR scorer based on:
        - Color contrast (higher = better)
        - Brightness in center vs edges (center subject = better)
        - Color vibrancy
        Returns the best thumbnail path and updates DB.
        """
        scores = []
        for path in thumbnail_paths:
            img = np.array(Image.open(path))
            # Contrast score
            contrast = img.std()
            # Saturation score
            hsv = Image.fromarray(img).convert("HSV")
            saturation = np.array(hsv)[:, :, 1].mean()
            # Brightness balance (not too dark, not overexposed)
            brightness = np.array(img).mean()
            brightness_score = 1.0 - abs(brightness - 128) / 128

            score = (contrast / 100) * 0.4 + (saturation / 255) * 0.4 + brightness_score * 0.2
            scores.append(score)

        best_idx = scores.index(max(scores))
        best_path = thumbnail_paths[best_idx]

        with get_db() as db:
            asset = (
                db.query(ThumbnailAsset)
                .filter(ThumbnailAsset.job_id == job_id, ThumbnailAsset.variant_index == best_idx)
                .first()
            )
            if asset:
                asset.is_selected = True
                asset.ctr_score = scores[best_idx]

        logger.info(f"Best thumbnail for job {job_id}: variant {best_idx} (score={scores[best_idx]:.3f})")
        return best_path


thumbnail_generator = ThumbnailGenerator()
