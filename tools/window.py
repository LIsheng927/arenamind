"""Small pygame helpers shared by the interactive tools."""

from __future__ import annotations

import pygame

from arena.render import render


def open_window(env, scale: int, title: str):
    pygame.init()
    pygame.display.set_caption(title)
    first = render(env, scale=scale)
    screen = pygame.display.set_mode(first.size)
    # With text input on, an input method (e.g. a Chinese IME) can swallow
    # letter keys such as WASD. The tools only need raw key states.
    if hasattr(pygame.key, "stop_text_input"):
        pygame.key.stop_text_input()
    return screen


def show(screen, env, scale: int, **kwargs) -> None:
    img = render(env, scale=scale, **kwargs)
    surface = pygame.image.frombytes(img.tobytes(), img.size, "RGB")
    screen.blit(surface, (0, 0))
    pygame.display.flip()
