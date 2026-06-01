"""
Modelos Pydantic para el perfil de canal devuelto por el LLM.

Usados con client.messages.parse() del SDK Anthropic para garantizar que la
respuesta del LLM es JSON válido contra este schema. Si el modelo se desvía,
el SDK lanza ValidationError y la llamada falla — no recibimos basura.
"""
from datetime import datetime
from pydantic import BaseModel, Field


class MusicProfile(BaseModel):
    """Perfil musical específico del canal — descriptores que MusicGen entiende."""
    bpm_target: int = Field(..., ge=60, le=200, description="BPM óptimo de target")
    bpm_range: tuple[int, int] = Field(..., description="Rango de BPM aceptable [min, max]")
    must_have_elements: list[str] = Field(
        ..., min_length=4, max_length=10,
        description="Elementos instrumentales que SIEMPRE deben aparecer (4-10)",
    )
    reference_artists: list[str] = Field(
        ..., min_length=3, max_length=8,
        description="Productores actuales (2020-2025) que definen el sonido del canal",
    )
    reference_labels: list[str] = Field(
        ..., min_length=2, max_length=6,
        description="Labels discográficos que publican música compatible con el canal",
    )
    production_keywords: list[str] = Field(
        ..., min_length=4, max_length=10,
        description="Términos de producción/mezcla (ej. 'sidechain pump', 'club-ready master')",
    )
    structure: str = Field(
        ...,
        description="Estructura típica del track DJ en una frase (ej. 'long intro → drop → breakdown → drop → outro')",
    )
    negative: list[str] = Field(
        ..., min_length=3, max_length=8,
        description="Características que NO debe tener la música del canal",
    )


class VisualProfile(BaseModel):
    """Perfil visual específico del canal — descriptores para SDXL/Juggernaut."""
    primary_subjects: list[str] = Field(
        ..., min_length=3, max_length=8,
        description="Sujetos principales que protagonizan las imágenes (ej. 'close-up barbell with chalk')",
    )
    lighting: str = Field(..., description="Esquema de iluminación característico")
    palette: list[str] = Field(
        ..., min_length=3, max_length=6,
        description="Paleta de colores dominante (3-6 colores)",
    )
    composition: str = Field(..., description="Estilo de composición (ej. 'close-up macro 50mm low angle')")
    references: list[str] = Field(
        ..., min_length=2, max_length=6,
        description="Referencias visuales (películas, fotógrafos, documentales)",
    )
    avoid: list[str] = Field(
        ..., min_length=3, max_length=8,
        description="Aspectos visuales que rompen el branding del canal",
    )


class ChannelProfile(BaseModel):
    """Perfil completo de un canal, cacheado en disco con TTL."""
    channel: str = Field(..., description="Identificador del canal (ironpulse, deepstack...)")
    music: MusicProfile
    visual: VisualProfile
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    ttl_days: int = Field(default=7, description="Días tras los cuales el perfil se considera caducado")
    model: str = Field(default="", description="Modelo Claude usado para generar el perfil")
    tokens_used: int = Field(default=0, description="Tokens totales consumidos")

    def is_expired(self) -> bool:
        from datetime import timezone
        now = datetime.now(timezone.utc)
        gen = self.generated_at
        if gen.tzinfo is None:
            gen = gen.replace(tzinfo=timezone.utc)
        age_days = (now - gen).total_seconds() / 86400
        return age_days > self.ttl_days
