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


# Vocabulario "track comercial actual" — términos que MusicGen reconoce porque
# entrenó con metadata real de Beatport/Spotify/Discogs. Se aplica a todos los
# estilos para evitar el sonido "AI demo" y empujar hacia algo que podría sonar
# en un festival o en una playlist de Spotify.
PRO_PRODUCTION_KEYWORDS = (
    "Beatport top 10 production, festival-ready master, club banger, "
    "punchy compressed kick, sidechained sub bass, wide stereo image, "
    "crisp transients, polished commercial mix, peak time energy"
)

# Negative prompt para evitar lo que hace que suene a "AI generation demo".
NEGATIVE_PROMPT = (
    "muddy mix, amateur production, lo-fi, demo recording, "
    "static loop, boring, repetitive, weak bass, harsh treble, "
    "out of tune, distorted clipping, background music, ambient drone"
)


# Templates re-orientados: foco en productores y labels ACTUALES (2020-2024)
# que el modelo asocia con sonido comercial moderno, no underground 90s.
STYLE_TEMPLATES = {
    MusicStyle.DARK_TECHNO: {
        "base": "modern peak-time dark techno banger, driving four-on-the-floor kick, rolling reese bass, hypnotic warehouse energy, Drumcode label production quality",
        "elements": [
            "rolling reese bassline", "Roland 909 drum kit", "tense modular arpeggio",
            "sidechain-pumping pad", "tape-saturated hi-hats", "industrial reverb stab",
            "filter-swept lead", "snare roll build-up",
        ],
        "references": ["Charlotte de Witte", "Adam Beyer", "Amelie Lens", "I Hate Models"],
        "bpm_range": (134, 140),
        "tempo_phrase": "driving {bpm} BPM warehouse techno groove",
    },
    MusicStyle.CYBERPUNK: {
        "base": "modern cyberpunk electronic banger, cinematic dark synthwave-techno hybrid, neon-lit dystopian energy, festival-ready production",
        "elements": [
            "fat analog lead synth", "distorted bitcrushed bass", "vocoded robotic chops",
            "side-chained pumping pad", "trap-style hi-hats", "cinematic riser",
            "epic supersaw drop", "808 sub kick",
        ],
        "references": ["Perturbator", "GosT", "Dan Terminus", "Kavinsky"],
        "bpm_range": (124, 132),
        "tempo_phrase": "pulsing {bpm} BPM dark club groove",
    },
    MusicStyle.NEON_AMBIENT: {
        "base": "deep cinematic neon ambient instrumental, lush evolving cinematic textures, modern soundtrack production, polished mix",
        "elements": [
            "wide reverb pad layers", "deep sub bass pulse", "delicate piano melody",
            "warm analog texture", "field recording atmosphere", "subtle granular shimmer",
            "tape saturation", "cinematic strings",
        ],
        "references": ["Jon Hopkins", "Burial", "Tycho", "Boards of Canada"],
        "bpm_range": (80, 95),
        "tempo_phrase": "slow {bpm} BPM cinematic downtempo pulse",
    },
    MusicStyle.INDUSTRIAL: {
        "base": "hard industrial techno banger, distorted pounding kick, mechanical aggression, peak-time warehouse energy, KNTXT label production",
        "elements": [
            "distorted overdriven kick", "industrial metal percussion",
            "reese bass growl", "white noise riser",
            "sidechain-pumped acid lead", "compressed reverb stab",
            "snare drum roll", "harsh filter sweep",
        ],
        "references": ["I Hate Models", "VTSS", "SPFDJ", "Paula Temple"],
        "bpm_range": (140, 150),
        "tempo_phrase": "relentless {bpm} BPM industrial pounding",
    },
    MusicStyle.SYNTHWAVE: {
        "base": "modern synthwave banger, 80s-inspired retro-futuristic production, cinematic neon night drive, polished commercial mix",
        "elements": [
            "warm analog lead synth", "gated reverb snare drum",
            "arpeggiated bass sequence", "DX7 electric piano stab",
            "chorused bass guitar", "vintage tape saturation",
            "sidechained pad", "wide stereo synth lead",
        ],
        "references": ["The Midnight", "FM-84", "Mitch Murder", "Kavinsky"],
        "bpm_range": (100, 118),
        "tempo_phrase": "smooth {bpm} BPM driving mid-tempo",
    },
    MusicStyle.ACID_TECHNO: {
        "base": "modern acid techno banger, Roland TB-303 squelching bassline, hypnotic peak-time energy, Octopus Recordings production quality",
        "elements": [
            "TB-303 acid bassline", "Roland 909 drums",
            "resonant filter sweep", "punchy clap layer",
            "minimal hi-hat groove", "acid lead arpeggio",
            "tape-warmed kick", "warehouse reverb",
        ],
        "references": ["Sina XX", "Boris Brejcha", "Reinier Zonneveld", "Sven Väth"],
        "bpm_range": (135, 145),
        "tempo_phrase": "hypnotic {bpm} BPM acid peak-time",
    },
    MusicStyle.HARDTEK: {
        "base": "modern hardtek banger, distorted hard kick, peak-time festival rave energy, polished hard dance production",
        "elements": [
            "distorted overdriven kick drum", "screeching 303 acid bass",
            "tribal psychedelic percussion", "chopped vocal stab",
            "epic rave stab", "snare roll build-up",
            "white noise riser", "hardcore lead synth",
        ],
        "references": ["Sefa", "Dr. Peacock", "Sickmode", "Hard Driver"],
        "bpm_range": (170, 185),
        "tempo_phrase": "explosive {bpm} BPM peak-time hardcore",
    },
}

# Estructura de pista DJ comercial. Los descriptores son los que un productor
# usaría en Beatport/Discogs y MusicGen reconoce. Solo cambian estos descriptores
# entre clips — NO la instrumentación — para mantener coherencia con
# continuation mode.
ENERGY_SECTIONS = {
    "intro":      "DJ intro section, stripped kick and bass only, beatport-style minimal arrangement",
    "build1":     "first build-up, rising tension, snare drum roll, filter sweep opening up",
    "drop1":      "first drop, full energy banger, all elements active, peak time club moment",
    "breakdown":  "breakdown section, kick drops out, atmospheric pad and lead melody, building tension",
    "build2":     "second build-up, riser sweeping, snare roll accelerating, anticipation peak",
    "drop2":      "second drop, biggest moment of the track, maximum energy festival anthem",
    "outro":      "DJ outro section, elements dropping out one by one, minimal kick-and-bass tail",
}


def _tempo_phrase(template: dict, bpm: int) -> str:
    return template["tempo_phrase"].format(bpm=bpm)


def _load_channel_music_profile(channel_key: str | None):
    """Carga el perfil musical del canal si existe y no caducó. None en otro caso."""
    if channel_key is None:
        return None
    try:
        from research import get_channel_profile
        profile = get_channel_profile(channel_key)
        return profile.music if profile else None
    except Exception:
        # Si la capa research no está disponible (sin anthropic instalado o sin
        # API key), nos caemos al template hardcoded sin romper el pipeline.
        return None


def build_music_prompt(
    style: MusicStyle,
    duration_seconds: int,
    bpm: int | None = None,
    section: str | None = None,
    custom_elements: list[str] | None = None,
    elements: list[str] | None = None,
    reference: str | None = None,
    channel_key: str | None = None,
) -> tuple[str, int]:
    """
    Build a single MusicGen prompt.

    `section` opcional: "intro" | "buildup" | "peak" | "sustain" | "outro".
    `elements` y `reference` opcionales: si se pasan, se reutilizan tal cual
    (clave para long-form: evita que la instrumentación cambie cada 30s).
    `channel_key` opcional: si hay un perfil cacheado para ese canal, sus
    descriptores SOBRESCRIBEN los del template hardcoded (BPM, elementos,
    artistas/labels, keywords de producción). Si no hay perfil válido, se
    usa el template tal cual — degradación silenciosa.
    """
    template = STYLE_TEMPLATES[style]
    channel_profile = _load_channel_music_profile(channel_key)

    if bpm is None:
        if channel_profile is not None:
            bpm = random.randint(*channel_profile.bpm_range)
        else:
            bpm = random.randint(*template["bpm_range"])

    if elements is None:
        if channel_profile is not None:
            # Tomar todos los must_have_elements del perfil (4-10 items)
            elements = random.sample(
                channel_profile.must_have_elements,
                k=min(4, len(channel_profile.must_have_elements)),
            )
        else:
            elements = random.sample(template["elements"], k=min(3, len(template["elements"])))
    else:
        elements = list(elements)
    if custom_elements:
        elements.extend(custom_elements)

    if reference is not None:
        ref = reference
    elif channel_profile is not None:
        # 70% artista, 30% label — los dos son señales fuertes para MusicGen
        if random.random() < 0.7:
            ref = random.choice(channel_profile.reference_artists)
        else:
            ref = f"{random.choice(channel_profile.reference_labels)} label"
    else:
        ref = random.choice(template["references"])

    tempo = _tempo_phrase(template, bpm)

    # Keywords de producción: si hay perfil, usar los suyos; si no, los hardcoded
    if channel_profile is not None:
        production_kw = ", ".join(channel_profile.production_keywords)
    else:
        production_kw = PRO_PRODUCTION_KEYWORDS

    parts = [
        template["base"],
        tempo,
        ", ".join(elements),
        f"in the style of {ref}",
        production_kw,
        "instrumental, no vocals",
    ]
    if section and section in ENERGY_SECTIONS:
        parts.insert(2, ENERGY_SECTIONS[section])

    return ", ".join(parts), bpm


def get_negative_prompt(channel_key: str | None = None) -> str:
    """
    Negative prompt para dual-CFG. Si se pasa un channel_key con perfil
    válido en caché, se concatenan los descriptores 'negative' del perfil
    al negative común — anti-patterns específicos del nicho.
    """
    profile_music = _load_channel_music_profile(channel_key)
    if profile_music is not None and profile_music.negative:
        return NEGATIVE_PROMPT + ", " + ", ".join(profile_music.negative)
    return NEGATIVE_PROMPT


def build_music_prompt_variations(
    style: MusicStyle,
    duration_seconds: int,
    count: int = 3,
    channel_key: str | None = None,
) -> list[dict]:
    """Multiple full-track prompt variations for retry logic."""
    variations = []
    for _ in range(count):
        prompt, bpm = build_music_prompt(style, duration_seconds, channel_key=channel_key)
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
    """
    Devuelve la sección DJ-style según la posición relativa en una pista
    long-form. Patrón clásico de pista de DJ comercial:

        intro → build1 → drop1 → breakdown → build2 → drop2 → outro

    Para tracks cortos (1-2 clips) hace fallback a 'drop1' (sección de
    máxima energía) porque es lo que más vende como muestra.
    """
    if total <= 1:
        return "drop1"
    if total == 2:
        return "intro" if idx == 0 else "drop1"
    if total == 3:
        return ["intro", "drop1", "outro"][idx]

    # >=4 clips: estructura DJ completa
    pos = idx / (total - 1)
    if pos < 0.10:
        return "intro"
    if pos < 0.25:
        return "build1"
    if pos < 0.45:
        return "drop1"
    if pos < 0.60:
        return "breakdown"
    if pos < 0.75:
        return "build2"
    if pos < 0.92:
        return "drop2"
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
