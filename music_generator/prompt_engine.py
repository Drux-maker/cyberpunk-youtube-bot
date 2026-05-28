"""
Music prompts tuned for MusicGen — descriptores musicales reales,
no metadatos de marketing (MusicGen no entrena con "high quality production").

Diseño:
- Una sola referencia artística por prompt (varias confunden al modelo).
- Tempo en lenguaje natural ("driving 130 BPM four-on-the-floor") en vez de "BPM" suelto.
- Estructura por secciones suave: solo cambian descriptores de ENERGÍA,
  no de instrumentación, para no romper la continuidad cuando se usa
  generate_continuation() en el generator.
"""
import random
from database.models import MusicStyle, PromptLibrary
from database.db import get_db


STYLE_TEMPLATES = {
    MusicStyle.DARK_TECHNO: {
        "base": "dark techno instrumental, pounding four-on-the-floor kick, deep rolling sub bass, industrial atmosphere, dystopian warehouse",
        "elements": ["acid 303 stabs", "metallic percussion", "filtered noise sweeps", "long reverb tails", "modular synth textures"],
        "references": ["Surgeon", "Phase Fatale", "Rebekah"],
        "bpm_range": (130, 140),
        "tempo_phrase": "driving {bpm} BPM four-on-the-floor groove",
    },
    MusicStyle.CYBERPUNK: {
        "base": "cyberpunk electronic instrumental, neon-lit nocturnal atmosphere, cinematic synthwave-techno hybrid, rain-soaked urban tension",
        "elements": ["analog lead synth", "heavy synth bassline", "dystopian pad layers", "arpeggiated sequencer", "vocoded textures"],
        "references": ["Perturbator", "Carpenter Brut", "Dan Terminus"],
        "bpm_range": (118, 132),
        "tempo_phrase": "pulsing {bpm} BPM mid-tempo groove",
    },
    MusicStyle.NEON_AMBIENT: {
        "base": "ambient electronic instrumental, deep immersive neon city soundscape, slow evolving textures, introspective late-night mood",
        "elements": ["long reverb pads", "sub bass pulses", "delicate melodic sequence", "granular textures", "field-recorded distant city"],
        "references": ["Burial", "The Haxan Cloak", "Tim Hecker"],
        "bpm_range": (75, 95),
        "tempo_phrase": "slow {bpm} BPM downtempo pulse",
    },
    MusicStyle.INDUSTRIAL: {
        "base": "industrial techno instrumental, harsh distorted kick, mechanical apocalyptic atmosphere, raw warehouse aggression",
        "elements": ["distorted kick drum", "industrial metal hits", "white noise bursts", "power electronics", "doom drone bass"],
        "references": ["Vatican Shadow", "Paula Temple", "Regis"],
        "bpm_range": (140, 155),
        "tempo_phrase": "relentless {bpm} BPM pounding beat",
    },
    MusicStyle.SYNTHWAVE: {
        "base": "synthwave instrumental, retro-futuristic 1980s analog production, dreamy neon nostalgia, cinematic Miami night drive",
        "elements": ["warm analog lead synth", "gated reverb snare", "arpeggio bass sequence", "DX7 electric piano", "chorus-drenched bass"],
        "references": ["Kavinsky", "Mitch Murder", "The Midnight"],
        "bpm_range": (98, 118),
        "tempo_phrase": "smooth {bpm} BPM mid-tempo drive",
    },
    MusicStyle.ACID_TECHNO: {
        "base": "acid techno instrumental, Roland TB-303 squelching bassline, hypnotic warehouse groove, raw analog underground",
        "elements": ["TB-303 acid bassline", "Roland 909 drum machine", "squelchy resonant filter sweeps", "hi-hat 16th-note pattern", "minimal arrangement"],
        "references": ["Josh Wink", "Hardfloor", "Dave Clarke"],
        "bpm_range": (133, 145),
        "tempo_phrase": "hypnotic {bpm} BPM four-on-the-floor",
    },
    MusicStyle.HARDTEK: {
        "base": "hardtek free-party instrumental, distorted hard kick, psychedelic tribal underground rave energy",
        "elements": ["distorted 303 bass", "hardcore kick drum", "ethnic tribal percussion", "psychedelic vocal chops", "rave stabs"],
        "references": ["Radium", "Dr. Macabre", "Acid Mike"],
        "bpm_range": (170, 185),
        "tempo_phrase": "fast {bpm} BPM hardcore tempo",
    },
}

# Solo cambian descriptores de energía/dinámica, NO de instrumentación,
# para no romper coherencia armónica con continuation mode.
ENERGY_SECTIONS = {
    "intro":    "atmospheric intro, restrained energy, sparse arrangement",
    "buildup":  "rising tension, layers gradually adding",
    "peak":     "full energy peak section, all elements active",
    "sustain":  "sustained groove, hypnotic locked-in flow",
    "outro":    "winding down, elements dropping out, fade-friendly tail",
}


def _tempo_phrase(template: dict, bpm: int) -> str:
    return template["tempo_phrase"].format(bpm=bpm)


def build_music_prompt(
    style: MusicStyle,
    duration_seconds: int,
    bpm: int | None = None,
    section: str | None = None,
    custom_elements: list[str] | None = None,
    elements: list[str] | None = None,
    reference: str | None = None,
) -> tuple[str, int]:
    """
    Build a single MusicGen prompt.

    `section` opcional: "intro" | "buildup" | "peak" | "sustain" | "outro".
    `elements` y `reference` opcionales: si se pasan, se reutilizan tal cual
    (clave para long-form: evita que la instrumentación cambie cada 30s).
    Si son None, se muestrean aleatoriamente del template.
    """
    template = STYLE_TEMPLATES[style]

    if bpm is None:
        bpm = random.randint(*template["bpm_range"])

    if elements is None:
        elements = random.sample(template["elements"], k=min(3, len(template["elements"])))
    else:
        elements = list(elements)
    if custom_elements:
        elements.extend(custom_elements)

    ref = reference if reference is not None else random.choice(template["references"])
    tempo = _tempo_phrase(template, bpm)

    parts = [
        template["base"],
        tempo,
        ", ".join(elements),
        f"in the style of {ref}",
        "instrumental, no vocals",
    ]
    if section and section in ENERGY_SECTIONS:
        parts.insert(2, ENERGY_SECTIONS[section])

    return ", ".join(parts), bpm


def build_music_prompt_variations(
    style: MusicStyle,
    duration_seconds: int,
    count: int = 3,
) -> list[dict]:
    """Multiple full-track prompt variations for retry logic."""
    variations = []
    for _ in range(count):
        prompt, bpm = build_music_prompt(style, duration_seconds)
        variations.append({
            "prompt": prompt,
            "bpm": bpm,
            "style": style.value,
            "duration_seconds": duration_seconds,
        })
    return variations


def sample_track_identity(style: MusicStyle) -> tuple[list[str], str]:
    """
    Selecciona elementos + referencia UNA vez para una pista entera.
    Usar al inicio de un long-form y pasar el resultado a build_music_prompt
    en cada sección, así la instrumentación no cambia clip a clip.
    """
    template = STYLE_TEMPLATES[style]
    elements = random.sample(template["elements"], k=min(3, len(template["elements"])))
    reference = random.choice(template["references"])
    return elements, reference


def section_for_position(idx: int, total: int) -> str:
    """Return section name based on relative position in long-form generation."""
    if total <= 1:
        return "peak"
    pos = idx / (total - 1)
    if pos < 0.05:
        return "intro"
    if pos < 0.2:
        return "buildup"
    if pos < 0.85:
        return "peak" if idx % 3 != 0 else "sustain"
    if pos < 0.95:
        return "sustain"
    return "outro"


def get_best_prompt_from_db(style: MusicStyle, prompt_type: str = "music") -> str | None:
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
