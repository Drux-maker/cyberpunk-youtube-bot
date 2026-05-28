"""
Generates music prompts automatically based on style, BPM, duration and mood.
Also learns from successful prompts stored in the database.
"""
import random
from database.models import MusicStyle, PromptLibrary
from database.db import get_db


STYLE_TEMPLATES = {
    MusicStyle.DARK_TECHNO: {
        "base": "dark techno, industrial beats, heavy kick drum, deep bassline, dystopian atmosphere",
        "moods": ["relentless", "hypnotic", "mechanical", "oppressive", "cold"],
        "elements": ["acid synths", "metallic percussion", "sub bass", "filtered noise", "reverb tails"],
        "references": ["Rebekah", "Surgeon", "Ancient Methods", "Phase Fatale"],
        "bpm_range": (130, 145),
    },
    MusicStyle.CYBERPUNK: {
        "base": "cyberpunk electronic music, neon atmosphere, futuristic, cinematic, urban decay",
        "moods": ["tense", "electric", "nocturnal", "dangerous", "pulsating"],
        "elements": ["glitchy synths", "heavy bassline", "dystopian pads", "vocoders", "arpeggiators"],
        "references": ["Carpenter Brut", "Perturbator", "Makeup and Vanity Set"],
        "bpm_range": (120, 135),
    },
    MusicStyle.NEON_AMBIENT: {
        "base": "ambient electronic music, neon city, atmospheric, deep immersive, floating textures",
        "moods": ["ethereal", "mysterious", "calm tension", "introspective", "vast"],
        "elements": ["long reverb pads", "subtle bass pulses", "melodic sequences", "field recordings", "granular textures"],
        "references": ["Brian Eno", "Burial", "Actress", "The Haxan Cloak"],
        "bpm_range": (80, 110),
    },
    MusicStyle.INDUSTRIAL: {
        "base": "industrial techno, harsh textures, noise elements, mechanical rhythms, apocalyptic",
        "moods": ["brutal", "aggressive", "chaotic", "grinding", "merciless"],
        "elements": ["distorted kicks", "industrial samples", "white noise", "metal hits", "power electronics"],
        "references": ["Vatican Shadow", "Blawan", "Paula Temple", "Regis"],
        "bpm_range": (140, 160),
    },
    MusicStyle.SYNTHWAVE: {
        "base": "synthwave retro-futuristic, 80s inspired, dreamy neon, nostalgic cyberpunk, cinematic",
        "moods": ["nostalgic", "triumphant", "romantic darkness", "cinematic", "retro future"],
        "elements": ["analog synths", "gated reverb drums", "arpeggio sequences", "sax leads", "chorus bass"],
        "references": ["Kavinsky", "College", "Chromatics", "Gunship"],
        "bpm_range": (95, 120),
    },
    MusicStyle.ACID_TECHNO: {
        "base": "acid techno, Roland TB-303 bassline, hypnotic groove, warehouse rave, underground",
        "moods": ["hypnotic", "relentless", "mind-bending", "trance-inducing", "raw"],
        "elements": ["303 acid bassline", "909 drums", "squelchy filter sweeps", "industrial hats", "minimal arrangement"],
        "references": ["DJ Pierre", "Josh Wink", "Hardfloor", "Dave Clarke"],
        "bpm_range": (133, 150),
    },
    MusicStyle.HARDTEK: {
        "base": "hardtek tekno, free party, distorted bassline, psychedelic, tribal, underground rave",
        "moods": ["aggressive", "tribal", "psychedelic", "uncompromising", "hypnotic"],
        "elements": ["distorted 303", "hard kicks", "psychedelic samples", "ethnic percussion", "breakdown breakdowns"],
        "references": ["Radium", "Dj Frenchie", "Teknotribe"],
        "bpm_range": (160, 190),
    },
}

DURATION_LABELS = {
    600: "10 minute",
    1800: "30 minute",
    3600: "1 hour",
    7200: "2 hour",
}


def build_music_prompt(
    style: MusicStyle,
    duration_seconds: int,
    bpm: int | None = None,
    intensity: str = "high",
    custom_elements: list[str] | None = None,
) -> str:
    template = STYLE_TEMPLATES[style]

    if bpm is None:
        bpm = random.randint(*template["bpm_range"])

    mood = random.choice(template["moods"])
    elements = random.sample(template["elements"], k=min(3, len(template["elements"])))

    if custom_elements:
        elements.extend(custom_elements)

    duration_label = DURATION_LABELS.get(duration_seconds, f"{duration_seconds // 60} minute")
    ref = random.choice(template["references"])

    prompt = (
        f"{template['base']}, {bpm} BPM, {mood} mood, "
        f"{', '.join(elements)}, "
        f"{intensity} energy, {duration_label} seamless mix, "
        f"no vocals, instrumental, high quality production, "
        f"inspired by {ref}, mastered for streaming"
    )
    return prompt, bpm


def build_music_prompt_variations(
    style: MusicStyle,
    duration_seconds: int,
    count: int = 3,
) -> list[dict]:
    """Generates multiple prompt variations to try different generations."""
    intensities = ["high", "medium", "extreme"]
    variations = []

    for i in range(count):
        intensity = intensities[i % len(intensities)]
        prompt, bpm = build_music_prompt(style, duration_seconds, intensity=intensity)
        variations.append({
            "prompt": prompt,
            "bpm": bpm,
            "intensity": intensity,
            "style": style.value,
            "duration_seconds": duration_seconds,
        })

    return variations


def get_best_prompt_from_db(style: MusicStyle, prompt_type: str = "music") -> str | None:
    """Returns the highest-performing prompt for a given style from past runs."""
    with get_db() as db:
        best = (
            db.query(PromptLibrary)
            .filter(
                PromptLibrary.prompt_type == prompt_type,
                PromptLibrary.style == style.value,
                PromptLibrary.usage_count > 3,
            )
            .order_by(PromptLibrary.performance_score.desc())
            .first()
        )
        return best.content if best else None


def save_prompt_performance(prompt: str, style: str, prompt_type: str, score: float) -> None:
    with get_db() as db:
        existing = (
            db.query(PromptLibrary)
            .filter(PromptLibrary.content == prompt)
            .first()
        )
        if existing:
            existing.usage_count += 1
            existing.performance_score = (existing.performance_score + score) / 2
        else:
            db.add(PromptLibrary(
                prompt_type=prompt_type,
                style=style,
                content=prompt,
                performance_score=score,
                usage_count=1,
            ))
