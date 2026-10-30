"""2D geometry for the arena, vectorised with numpy.

Every collision question in the game is answered by one of three shapes:

    * circle vs rectangle  -> a player walking into cover
    * ray vs rectangle     -> range sensors, line of sight, bullets vs cover
    * segment vs circle    -> a bullet hitting a player

All functions accept batches: a point is an array whose last axis has size 2,
so a single point (2,), N points (N, 2) and N games x 2 players (N, 2, 2)
all work with the same code. Rectangles are (x_min, y_min, x_max, y_max)
rows of a (K, 4) array.

Collision functions that look along a path return "distance travelled until
first contact" (np.inf for no contact), so callers can compare two kinds of
obstacle simply by taking the smaller distance.
"""

from __future__ import annotations

import numpy as np

ArrayLike = np.ndarray


# ---------------------------------------------------------------------------
# Circle vs rectangle: keep players out of cover and inside the arena
# ---------------------------------------------------------------------------

def push_out_of_rects(pos: ArrayLike, radius: float, rects: ArrayLike) -> np.ndarray:
    """Move circles (centre `pos`, given radius) so they no longer overlap any rectangle.

    For each rectangle, find the point of the rectangle nearest to the centre
    by clamping the centre into the rectangle. If that point is closer than
    `radius`, slide the centre straight away from it until the gap is exactly
    `radius`. Callers should move in steps shorter than `radius`, so a centre
    never ends up inside a rectangle; if it does anyway, it is pushed out
    through the nearest edge.
    """
    pos = np.array(pos, dtype=np.float64, copy=True)
    for x0, y0, x1, y1 in np.asarray(rects, dtype=np.float64).reshape(-1, 4):
        x, y = pos[..., 0], pos[..., 1]
        nx = np.clip(x, x0, x1)  # nearest point on the rectangle
        ny = np.clip(y, y0, y1)
        dx, dy = x - nx, y - ny
        dist = np.hypot(dx, dy)

        # Case 1: centre outside the rectangle but closer than `radius`.
        outside = dist > 0
        touching = outside & (dist < radius)
        safe = np.where(outside, dist, 1.0)  # avoid dividing by zero
        new_x = np.where(touching, nx + dx / safe * radius, x)
        new_y = np.where(touching, ny + dy / safe * radius, y)

        # Case 2: centre inside (or exactly on) the rectangle.
        inside = ~outside
        if inside.any():
            depth = np.stack([x - x0, x1 - x, y - y0, y1 - y], axis=-1)
            side = np.argmin(depth, axis=-1)  # 0 left, 1 right, 2 top, 3 bottom
            new_x = np.where(inside & (side == 0), x0 - radius, new_x)
            new_x = np.where(inside & (side == 1), x1 + radius, new_x)
            new_y = np.where(inside & (side == 2), y0 - radius, new_y)
            new_y = np.where(inside & (side == 3), y1 + radius, new_y)

        pos[..., 0], pos[..., 1] = new_x, new_y
    return pos


def clamp_to_arena(pos: ArrayLike, radius: float, size: float) -> np.ndarray:
    """Keep circle centres at least `radius` away from the arena walls."""
    return np.clip(np.asarray(pos, dtype=np.float64), radius, size - radius)


# ---------------------------------------------------------------------------
# Ray vs rectangle (slab method): sensors, line of sight, bullets vs cover
# ---------------------------------------------------------------------------

def ray_rect_distances(origins: ArrayLike, dirs: ArrayLike, rects: ArrayLike) -> np.ndarray:
    """Distance along each ray to the first point of each rectangle.

    `dirs` must be unit vectors, so the parameter t equals distance.
    Returns shape (..., K): one distance per ray per rectangle, np.inf on a miss,
    0 if the ray starts inside the rectangle.

    Slab method: the ray is inside the rectangle's x-range ("x slab") for
    t in [tx_near, tx_far] and inside its y-range for t in [ty_near, ty_far].
    It is inside the rectangle exactly when both hold, i.e. on the overlap
    [max(nears), min(fars)]. No overlap means a miss.
    """
    o = np.asarray(origins, dtype=np.float64)[..., None, :]  # (..., 1, 2)
    d = np.asarray(dirs, dtype=np.float64)[..., None, :]  # (..., 1, 2)
    r = np.asarray(rects, dtype=np.float64).reshape(-1, 4)
    lo, hi = r[:, :2], r[:, 2:]  # (K, 2): (x_min, y_min) and (x_max, y_max)

    with np.errstate(divide="ignore", invalid="ignore"):
        t_lo = (lo - o) / d  # t at which the ray crosses x_min (and y_min)
        t_hi = (hi - o) / d  # t at which the ray crosses x_max (and y_max)
    near = np.minimum(t_lo, t_hi)
    far = np.maximum(t_lo, t_hi)

    # A ray parallel to an axis never crosses that slab's edges (division by
    # zero above gives inf or nan). Either it is always inside the slab
    # (slab = all t) or never (slab = empty).
    parallel = d == 0
    in_slab = (o >= lo) & (o <= hi)
    near = np.where(parallel, np.where(in_slab, -np.inf, np.inf), near)
    far = np.where(parallel, np.where(in_slab, np.inf, -np.inf), far)

    t_enter = near.max(axis=-1)  # later of the two entries
    t_exit = far.min(axis=-1)  # earlier of the two exits
    hit = (t_enter <= t_exit) & (t_exit >= 0)  # overlap exists and is not behind us
    return np.where(hit, np.maximum(t_enter, 0.0), np.inf)


def ray_rects(origins: ArrayLike, dirs: ArrayLike, rects: ArrayLike) -> np.ndarray:
    """Distance along each ray to the nearest rectangle (np.inf if none)."""
    dist = ray_rect_distances(origins, dirs, rects)
    if dist.shape[-1] == 0:
        return np.full(dist.shape[:-1], np.inf)
    return dist.min(axis=-1)


def ray_arena_exit(origins: ArrayLike, dirs: ArrayLike, size: float) -> np.ndarray:
    """Distance along each ray (starting inside the arena) to the wall it hits."""
    o = np.asarray(origins, dtype=np.float64)
    d = np.asarray(dirs, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(d > 0, (size - o) / d, np.where(d < 0, -o / d, np.inf))
    return t.min(axis=-1)


def raycast(origins: ArrayLike, dirs: ArrayLike, rects: ArrayLike,
            size: float, max_dist: float) -> np.ndarray:
    """Range sensor: distance to the nearest cover or wall, capped at max_dist."""
    return np.minimum(
        np.minimum(ray_rects(origins, dirs, rects), ray_arena_exit(origins, dirs, size)),
        max_dist,
    )


def _unit(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (unit vectors, lengths); zero vectors stay zero."""
    length = np.linalg.norm(v, axis=-1)
    safe = np.where(length > 0, length, 1.0)
    return v / safe[..., None], length


def line_of_sight(a: ArrayLike, b: ArrayLike, rects: ArrayLike) -> np.ndarray:
    """True where nothing in `rects` blocks the straight line from a to b."""
    a = np.asarray(a, dtype=np.float64)
    dirs, dist = _unit(np.asarray(b, dtype=np.float64) - a)
    return ray_rects(a, dirs, rects) >= dist


def segment_rects_entry(p0: ArrayLike, p1: ArrayLike, rects: ArrayLike) -> np.ndarray:
    """Distance from p0 to the first rectangle on the segment p0->p1 (np.inf if none)."""
    p0 = np.asarray(p0, dtype=np.float64)
    dirs, length = _unit(np.asarray(p1, dtype=np.float64) - p0)
    dist = ray_rects(p0, dirs, rects)
    return np.where(dist <= length, dist, np.inf)


# ---------------------------------------------------------------------------
# Segment vs circle: a bullet's path this step against a player
# ---------------------------------------------------------------------------

def segment_circle_entry(p0: ArrayLike, p1: ArrayLike, centers: ArrayLike,
                         radius: float) -> np.ndarray:
    """Distance from p0 to where the segment p0->p1 first touches the circle.

    Checking only the end point misses fast bullets: a bullet moves 1.5 per
    step but a player is only 1.0 wide, so it can jump right over them.
    Instead, solve for the first t in [0, 1] where the moving point
    p(t) = p0 + t * (p1 - p0) is exactly `radius` from the centre:
        |f + t d|^2 = radius^2,  with f = p0 - c, d = p1 - p0
        a t^2 + b t + c' = 0,     a = d.d, b = 2 f.d, c' = f.f - radius^2
    The smaller root is the moment of first contact.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    d = np.asarray(p1, dtype=np.float64) - p0
    f = p0 - np.asarray(centers, dtype=np.float64)

    a = (d * d).sum(axis=-1)
    b = 2.0 * (f * d).sum(axis=-1)
    c = (f * f).sum(axis=-1) - radius * radius
    disc = b * b - 4.0 * a * c

    with np.errstate(divide="ignore", invalid="ignore"):
        t = (-b - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * a)

    starts_inside = c <= 0
    crosses = (a > 0) & (disc >= 0) & (t >= 0) & (t <= 1)
    t = np.where(starts_inside, 0.0, t)
    return np.where(starts_inside | crosses, t * np.sqrt(a), np.inf)
