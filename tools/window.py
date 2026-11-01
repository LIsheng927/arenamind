"""Small pygame helpers shared by the interactive tools."""

from __future__ import annotations

import pygame

from arena.render import render


def open_window(env, scale: int, title: str):
    pygame.init()
    pygame.display.set_caption(title)
    first = render(env, scale=scale)
    return pygame.display.set_mode(first.size)


def show(screen, env, scale: int, **kwargs) -> None:
    img = render(env, scale=scale, **kwargs)
    surface = pygame.image.frombytes(img.tobytes(), img.size, "RGB")
    screen.blit(surface, (0, 0))
    pygame.display.flip()
