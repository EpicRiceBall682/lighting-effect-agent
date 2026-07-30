"""Public interfaces for SDL gamut mapping."""

from .mapper import GamutMapper, MappingResult
from .sdl_palette import DEFAULT_SDL_PATH, SDLPalette

__all__ = ["DEFAULT_SDL_PATH", "GamutMapper", "MappingResult", "SDLPalette"]
