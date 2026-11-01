"""Draw one game of an ArenaEnv as an image (Pillow).

Used by the interactive tools (which put the image in a pygame window) and by
tools/replay.py (which writes GIFs), so every view of the game is drawn by the
same code.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import geometry as geo

COLORS = {
    "background": (250, 249, 245),
    "grid": (236, 234, 227),
    "grid_major": (222, 220, 212),
    "cover": (176, 174, 165),
    "text": (60, 60, 58),
    "ray": (190, 188, 180),
    "hp_back": (214, 212, 203),
    "players": ((55, 138, 221), (216, 90, 48)),  # blue, orange
}
HUD_HEIGHT = 30


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has a single fixed-size default font
        return ImageFont.load_default()


def render(env, game: int = 0, scale: int = 30, rays_for: int | None = None,
           status: str = "") -> Image.Image:
    """Picture of game `game`. rays_for=0 or 1 also draws that player's range rays."""
    size = env.cfg.arena.size
    w = int(round(size * scale))
    img = Image.new("RGB", (w, w + HUD_HEIGHT), COLORS["background"])
    d = ImageDraw.Draw(img)

    def px(v):
        return float(v) * scale

    def circle(c, r, **kw):
        x, y = px(c[0]), px(c[1])
        d.ellipse([x - px(r), y - px(r), x + px(r), y + px(r)], **kw)

    for i in range(1, int(size)):
        color = COLORS["grid_major"] if i % 5 == 0 else COLORS["grid"]
        d.line([(px(i), 0), (px(i), w)], fill=color)
        d.line([(0, px(i)), (w, px(i))], fill=color)
    for x0, y0, x1, y1 in env.obstacles:
        d.rectangle([px(x0), px(y0), px(x1), px(y1)], fill=COLORS["cover"])

    radius = env.cfg.agent.radius
    pos, angle = env.pos[game], env.angle[game]

    if rays_for is not None:
        n = env.cfg.obs.n_rays
        a = angle[rays_for] + np.arange(n) * (2 * np.pi / n)
        dirs = np.stack([np.cos(a), np.sin(a)], axis=-1)
        origins = np.broadcast_to(pos[rays_for], dirs.shape)
        dist = geo.raycast(origins, dirs, env.obstacles, size, env.cfg.obs.ray_range)
        for k in range(n):
            end = pos[rays_for] + dirs[k] * dist[k]
            d.line([(px(pos[rays_for][0]), px(pos[rays_for][1])), (px(end[0]), px(end[1]))],
                   fill=COLORS["ray"], width=2 if k == 0 else 1)

    # Where each player believes the other is, while they cannot see each other.
    if not env.visible[game]:
        for viewer in (0, 1):
            circle(env.last_seen[game, viewer], radius, outline=COLORS["players"][1 - viewer], width=2)

    for owner in (0, 1):
        for k in np.flatnonzero(env.bullet_alive[game, owner]):
            circle(env.bullet_pos[game, owner, k], 0.15, fill=COLORS["players"][owner])

    max_hp = env.cfg.agent.max_hp
    for p in (0, 1):
        color = COLORS["players"][p]
        circle(pos[p], radius, fill=color)
        tip = pos[p] + 0.95 * np.array([np.cos(angle[p]), np.sin(angle[p])])
        d.line([(px(pos[p][0]), px(pos[p][1])), (px(tip[0]), px(tip[1]))], fill=color,
               width=max(2, scale // 8))
        bar_x, bar_y = px(pos[p][0] - 0.6), px(pos[p][1] - 0.95)
        d.rectangle([bar_x, bar_y, bar_x + px(1.2), bar_y + px(0.14)], fill=COLORS["hp_back"])
        d.rectangle([bar_x, bar_y, bar_x + px(1.2) * env.hp[game, p] / max_hp, bar_y + px(0.14)],
                    fill=color)

    seconds = env.t[game] * env.cfg.dt
    text = (f"{seconds:4.1f} s   blue {env.hp[game, 0]:3.0f} hp   "
            f"orange {env.hp[game, 1]:3.0f} hp   {status}")
    d.text((8, w + 7), text, fill=COLORS["text"], font=_font(16))
    return img
