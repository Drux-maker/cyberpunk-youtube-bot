"""
Capa de research con LLM (Anthropic Claude).

Profila cada canal con un LLM para obtener descriptores hyper-específicos
(artistas referencia, labels, paletas visuales, etc.) que complementan los
templates hardcoded en los prompt engines.
"""
from .channel_profiler import profile_channel, get_channel_profile
from .models import ChannelProfile, MusicProfile, VisualProfile

__all__ = [
    "profile_channel",
    "get_channel_profile",
    "ChannelProfile",
    "MusicProfile",
    "VisualProfile",
]
