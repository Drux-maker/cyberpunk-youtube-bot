"""
Generates visual prompts for JuggernautXL Lightning
consistent with the music style and channel brand.
"""
import random
from database.models import MusicStyle


STYLE_VISUAL_MAP = {
    MusicStyle.DARK_TECHNO: {
        "base": "cyberpunk underground club, industrial warehouse, dark techno rave, strobe lights, smoke machine, brutalist architecture",
        "color_palette": "deep black, electric blue, harsh white strobes, blood red accents",
        "atmosphere": "oppressive, claustrophobic, intense, underground",
        "subjects": [
            "abandoned factory with pulsating lights",
            "dark warehouse rave crowd silhouettes",
            "industrial machinery with neon accents",
            "concrete tunnels with laser grids",
            "brutalist tower blocks at night with scanning lights",
        ],
    },
    MusicStyle.CYBERPUNK: {
        "base": "cyberpunk megacity, neon-drenched streets, rain-soaked asphalt, holographic advertisements, flying vehicles",
        "color_palette": "electric purple, hot pink neon, cyan blue, orange glow, deep shadows",
        "atmosphere": "tense, electric, nocturnal, cinematic",
        "subjects": [
            "cyberpunk city aerial view with neon rain",
            "narrow alley with holographic signs and rain puddles",
            "futuristic market district at night",
            "high-tech district skyline with flying cars",
            "augmented human figure in neon-lit street",
        ],
    },
    MusicStyle.NEON_AMBIENT: {
        "base": "ethereal neon landscape, dreamlike city, floating islands, bioluminescent environment, misty atmosphere",
        "color_palette": "soft cyan, gentle purple, faint gold, deep indigo, ghostly white",
        "atmosphere": "ethereal, vast, mysterious, serene tension",
        "subjects": [
            "floating neon city islands above clouds",
            "bioluminescent ocean with cyberpunk skyline",
            "misty mountain with neon temples",
            "infinite neon corridor into darkness",
            "giant holographic jellyfish over city",
        ],
    },
    MusicStyle.INDUSTRIAL: {
        "base": "post-apocalyptic industrial wasteland, corroded metal structures, toxic atmosphere, dystopian ruins",
        "color_palette": "rust orange, toxic green, ash grey, dark red, industrial yellow",
        "atmosphere": "apocalyptic, harsh, decaying, hostile",
        "subjects": [
            "ruins of a megafactory under blood-red sky",
            "toxic waste facility at night",
            "abandoned power plant with glowing core",
            "dystopian industrial cityscape with acid rain",
            "corroded metal cathedral in wasteland",
        ],
    },
    MusicStyle.SYNTHWAVE: {
        "base": "retro-futuristic 80s aesthetic, grid landscapes, sunset on alien world, chrome surfaces, synthwave vibes",
        "color_palette": "magenta, teal, gold sunset, deep purple, chrome silver",
        "atmosphere": "nostalgic, triumphant, cinematic, dreamy",
        "subjects": [
            "infinite purple grid landscape at sunset",
            "retro-futuristic city with chrome towers",
            "sports car driving on neon grid highway",
            "alien sunset with geometric sun and grid",
            "80s retro arcade in cyberpunk city",
        ],
    },
    MusicStyle.ACID_TECHNO: {
        "base": "psychedelic warehouse rave, fluid acid visuals, underground bunker, hypnotic patterns, altered reality",
        "color_palette": "acid green, electric yellow, pulsating orange, deep black, white flash",
        "atmosphere": "hypnotic, disorienting, intense, raw",
        "subjects": [
            "underground bunker with acid-washed walls",
            "hypnotic tunnel with pulsing neon rings",
            "psychedelic warehouse with laser show",
            "liquid acid geometry in dark space",
            "rave crowd in blacklight underground",
        ],
    },
    MusicStyle.HARDTEK: {
        "base": "free party forest rave, tek-tek sound system, psychedelic nature, tribal fire, underground movement",
        "color_palette": "fire orange, UV purple, deep forest green, starry black, primal red",
        "atmosphere": "tribal, primal, wild, psychedelic",
        "subjects": [
            "forest rave with fire and lasers at night",
            "tribal psychedelic ceremony under stars",
            "massive speaker stack in industrial space",
            "underground cave rave with bioluminescence",
            "dystopian campsite with fire sculptures",
        ],
    },
}

QUALITY_SUFFIXES = [
    "ultra detailed, 8K UHD, cinematic photography",
    "photorealistic, award winning, masterpiece",
    "hyper realistic, professional photography, cinematic lighting",
    "highly detailed, trending on artstation, epic composition",
]

NEGATIVE_PROMPT = (
    "blurry, low quality, watermark, text, logo, humans close-up faces, "
    "cartoon, anime, drawing, painting, 3d render, plastic, oversaturated, "
    "distorted, ugly, amateur, stock photo"
)


def _load_channel_visual_profile(channel_key: str | None):
    """Carga el perfil visual del canal si existe y no caducó. None en otro caso."""
    if channel_key is None:
        return None
    try:
        from research import get_channel_profile
        profile = get_channel_profile(channel_key)
        return profile.visual if profile else None
    except Exception:
        return None


def build_visual_prompt(
    style: MusicStyle,
    custom_subject: str | None = None,
    channel_key: str | None = None,
) -> dict:
    """
    Construye un prompt para JuggernautXL.

    Si `channel_key` apunta a un canal con perfil cacheado, los descriptores
    visuales del perfil ENRIQUECEN el prompt (sujetos del canal con prioridad,
    iluminación, paleta, composición y referencias específicas del nicho).
    Si no, se usa el template hardcoded — degradación silenciosa.
    """
    template = STYLE_VISUAL_MAP[style]
    visual_profile = _load_channel_visual_profile(channel_key)

    # Sujeto: si hay perfil, alternar entre sujetos del canal y del estilo
    if custom_subject is not None:
        subject = custom_subject
    elif visual_profile is not None and random.random() < 0.7:
        # 70% sujetos del perfil del canal — más relevantes para la tribu
        subject = random.choice(visual_profile.primary_subjects)
    else:
        subject = random.choice(template["subjects"])

    quality = random.choice(QUALITY_SUFFIXES)

    if visual_profile is not None:
        # El perfil sobrescribe paleta/atmósfera/composición con datos hyper-específicos
        positive = (
            f"{subject}, "
            f"lighting: {visual_profile.lighting}, "
            f"color palette: {', '.join(visual_profile.palette)}, "
            f"composition: {visual_profile.composition}, "
            f"reference: {random.choice(visual_profile.references)}, "
            f"{quality}"
        )
        negative = NEGATIVE_PROMPT + ", " + ", ".join(visual_profile.avoid)
    else:
        positive = (
            f"{subject}, {template['base']}, "
            f"color palette: {template['color_palette']}, "
            f"atmosphere: {template['atmosphere']}, "
            f"{quality}"
        )
        negative = NEGATIVE_PROMPT

    return {
        "positive": positive,
        "negative": negative,
        "style": style.value,
        "subject": subject,
    }


def build_visual_batch(
    style: MusicStyle,
    count: int = 5,
    channel_key: str | None = None,
) -> list[dict]:
    """
    Generate diverse prompts for a batch of visuals.

    Si se pasa `channel_key`, los sujetos del perfil del canal entran a la
    rotación junto a los sujetos del template del estilo, garantizando que
    cada imagen tenga un sujeto distinto.
    """
    template = STYLE_VISUAL_MAP[style]
    visual_profile = _load_channel_visual_profile(channel_key)

    # Pool de sujetos: estilo + canal (si hay perfil)
    pool = list(template["subjects"])
    if visual_profile is not None:
        pool = list(visual_profile.primary_subjects) + pool

    subjects = random.sample(pool, k=min(count, len(pool)))
    while len(subjects) < count:
        subjects.append(random.choice(pool) + ", from a different angle, dynamic perspective")

    return [build_visual_prompt(style, subject, channel_key=channel_key) for subject in subjects]
