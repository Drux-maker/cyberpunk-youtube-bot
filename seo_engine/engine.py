"""
SEO metadata generation.

Primary path: OpenAI GPT-4o-mini for rich, human-sounding metadata.
Fallback:     fully-local template-based metadata (used automatically when
              OPENAI_API_KEY is missing or the API call fails).

Either way, every job ends up with a SEOMetadata row containing
title + description + tags + chapters + pinned_comment.
"""
import json
import logging
from typing import Optional

from config.settings import settings
from database.models import MusicStyle, SEOMetadata
from database.db import get_db

logger = logging.getLogger(__name__)


def _get_openai_client():
    """Lazy import + construct the OpenAI client only when actually needed."""
    if not settings.OPENAI_API_KEY:
        return None
    try:
        from openai import AsyncOpenAI
        return AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    except Exception as exc:
        logger.warning(f"OpenAI client unavailable, falling back to templates: {exc}")
        return None

DURATION_MAP = {600: "10-minute", 1800: "30-minute", 3600: "1-hour", 7200: "2-hour"}

SYSTEM_PROMPT = """You are an expert YouTube SEO strategist specializing in
electronic music channels. You generate viral, search-optimized metadata for
AI-generated techno/cyberpunk music mixes.

Rules:
- Titles: max 70 chars, clickbait-SEO balance, no caps-lock spam
- Tags: mix broad terms + niche + year + AI angle
- Descriptions: 200-400 words, keywords in first 2 lines, sections with headers
- Always sound like human-curated content, never like AI spam
- Include relevant keywords: AI music, cyberpunk, techno, electronic, mix, focus, study
- Output must be valid JSON only"""

TITLE_PATTERNS = [
    "{duration} {style_name} Mix | AI Generated {year}",
    "Best {style_name} Music for Deep Focus / Study [{duration}]",
    "{style_name} AI Mix {year} — {mood} | {duration}",
    "Dark {style_name} | Neon City Vibes | {duration} Non-Stop",
    "AI Generated {style_name} | {mood} | {year}",
    "{duration} of Pure {style_name} [AI Music {year}]",
]

STYLE_NAMES = {
    MusicStyle.DARK_TECHNO: "Dark Techno",
    MusicStyle.CYBERPUNK: "Cyberpunk Electronic",
    MusicStyle.NEON_AMBIENT: "Neon Ambient",
    MusicStyle.INDUSTRIAL: "Industrial Techno",
    MusicStyle.SYNTHWAVE: "Synthwave",
    MusicStyle.ACID_TECHNO: "Acid Techno",
    MusicStyle.HARDTEK: "Hardtek",
}

STYLE_MOODS = {
    MusicStyle.DARK_TECHNO: ["Hypnotic", "Relentless", "Underground"],
    MusicStyle.CYBERPUNK: ["Electric", "Cinematic", "Nocturnal"],
    MusicStyle.NEON_AMBIENT: ["Ethereal", "Focus", "Immersive"],
    MusicStyle.INDUSTRIAL: ["Brutal", "Raw", "Apocalyptic"],
    MusicStyle.SYNTHWAVE: ["Nostalgic", "Cinematic", "Retro-Futuristic"],
    MusicStyle.ACID_TECHNO: ["Hypnotic", "Mind-Bending", "Raw"],
    MusicStyle.HARDTEK: ["Tribal", "Relentless", "Underground"],
}


async def generate_seo_metadata(
    job_id: int,
    style: MusicStyle,
    duration_seconds: int,
    bpm: int,
    music_prompt: str,
) -> int:
    duration_label = DURATION_MAP.get(duration_seconds, f"{duration_seconds // 60}-minute")
    style_name = STYLE_NAMES[style]
    from datetime import date
    year = date.today().year

    user_message = f"""
Generate YouTube metadata for this AI music video:

Style: {style_name}
Duration: {duration_label}
BPM: {bpm}
Music description: {music_prompt[:200]}
Year: {year}

Return a single JSON object with exactly these keys:
{{
  "title": "string (max 70 chars)",
  "description": "string (200-400 words, markdown-friendly)",
  "tags": ["array", "of", "20-30", "strings"],
  "chapters": [{{"time": "0:00", "title": "string"}}, ...],
  "pinned_comment": "string (engaging first comment, 1-2 sentences)"
}}

For chapters, create 5-8 chapters spread across the duration.
"""

    client = _get_openai_client()
    data: Optional[dict] = None
    if client is not None:
        try:
            response = await client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.8,
                max_tokens=1500,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            data = json.loads(raw)
            logger.info(f"SEO generated via OpenAI for job {job_id}")
        except Exception as e:
            logger.warning(f"OpenAI SEO failed ({e}); using local fallback")
            data = None

    if data is None:
        data = _fallback_metadata(style, duration_label, style_name, year, duration_seconds)
        logger.info(f"SEO generated via local templates for job {job_id}")

    # Ensure tags don't exceed YouTube's 500-char total limit
    tags = data.get("tags", [])[:30]
    total_tag_chars = sum(len(t) for t in tags)
    while total_tag_chars > 490 and tags:
        tags.pop()
        total_tag_chars = sum(len(t) for t in tags)

    with get_db() as db:
        seo = SEOMetadata(
            job_id=job_id,
            title=data["title"][:100],
            description=data["description"],
            tags=tags,
            chapters=data.get("chapters", []),
            pinned_comment=data.get("pinned_comment", ""),
            playlist_names=_get_playlists(style),
        )
        db.add(seo)
        db.flush()
        seo_id = seo.id

    logger.info(f"SEO metadata generated for job {job_id}: '{data['title']}'")
    return seo_id


def _get_playlists(style: MusicStyle) -> list[str]:
    base = ["AI Generated Music", "Electronic Mixes"]
    style_playlists = {
        MusicStyle.DARK_TECHNO: ["Dark Techno", "Underground Techno"],
        MusicStyle.CYBERPUNK: ["Cyberpunk Music", "Sci-Fi Soundscapes"],
        MusicStyle.NEON_AMBIENT: ["Focus & Study Music", "Ambient Electronic"],
        MusicStyle.INDUSTRIAL: ["Industrial Techno", "Noise & Industrial"],
        MusicStyle.SYNTHWAVE: ["Synthwave & Retrowave", "80s Inspired"],
        MusicStyle.ACID_TECHNO: ["Acid Techno", "Underground Rave"],
        MusicStyle.HARDTEK: ["Hardtek & Tekno", "Free Party Music"],
    }
    return base + style_playlists.get(style, [])


def _fallback_metadata(
    style: MusicStyle,
    duration_label: str,
    style_name: str,
    year: int,
    duration_seconds: int = 3600,
) -> dict:
    import random
    mood = random.choice(STYLE_MOODS[style])
    return {
        "title": f"{duration_label.capitalize()} {style_name} Mix — {mood} | AI Generated {year}",
        "description": (
            f"🎵 {duration_label.capitalize()} of {style_name} music, fully generated by AI.\n\n"
            f"🌆 Atmosphere: {mood.lower()} • Underground • Cinematic\n\n"
            f"Perfect for deep focus, late night sessions, or just losing yourself in the music.\n\n"
            f"All music is 100% AI-generated. No copyright. Free to stream.\n\n"
            f"This music was created using AI tools (MusicGen). "
            f"Visuals generated with JuggernautXL Lightning.\n\n"
            f"#AIMusic #{style_name.replace(' ', '')} #ElectronicMusic #TechnoMix #{year}"
        ),
        "tags": [
            style_name.lower(), "ai music", "ai generated music", f"{style_name.lower()} mix",
            "electronic music", "techno", "cyberpunk music", "no copyright music",
            "focus music", "study music", "deep focus", f"{duration_label} mix",
            f"ai techno {year}", "underground electronic", "dark electronic",
            "music for work", "concentration music", "instrumental",
            f"{style_name.lower()} {year}", "artificial intelligence music",
        ],
        "chapters": _generate_chapters(duration_seconds),
        "pinned_comment": (
            f"This mix was entirely generated by AI — the music, the visuals, everything. "
            f"Drop a 🤖 if you're here for the future of music!"
        ),
    }


def _generate_chapters(duration_seconds: int) -> list[dict]:
    chapter_names = [
        "Intro — Neon Dawn", "Phase I — Dark Pulse", "Phase II — Descent",
        "Phase III — The Grid", "Phase IV — Hypnosis", "Phase V — Overload",
        "Phase VI — Void", "Outro — Fade to Black",
    ]
    interval = duration_seconds // len(chapter_names)
    return [
        {"time": _seconds_to_timestamp(i * interval), "title": chapter_names[i]}
        for i in range(len(chapter_names))
    ]


def _seconds_to_timestamp(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"
