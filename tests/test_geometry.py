"""Geometry tests. Expected values are worked out by hand in the comments."""

import math

import numpy as np
import pytest

from arena.config import GameConfig
from arena.geometry import (
    clamp_to_arena,
    line_of_sight,
    push_out_of_rects,
    ray_arena_exit,
    ray_rects,
    raycast,
    segment_circle_entry,
    segment_rects_entry,
)

CFG = GameConfig()
OBSTACLES = np.array(CFG.arena.obstacles)
CENTRE_BLOCK = np.array([[9.0, 9.0, 11.0, 11.0]])
LOWER_LEFT_BAR = np.array([[4.0, 11.0, 8.0, 12.0]])
LOWER_RIGHT_PILLAR = np.array([[13.0, 13.0, 14.0, 14.0]])
S = math.sqrt(0.5)
PLAYER = np.array([5.5, 15.0])  # a player in the lower-left part of the map


# ---- circle vs rectangle ---------------------------------------------------

def test_walking_into_cover_stops_flush_against_it():
    # At (8.6, 10), step right 0.4 to (9.0, 10): overlaps the block by 0.5.
    assert push_out_of_rects([9.0, 10.0], 0.5, CENTRE_BLOCK) == pytest.approx([8.5, 10.0])


def test_corner_contact_pushes_diagonally():
    out = push_out_of_rects([8.7, 8.7], 0.5, CENTRE_BLOCK)
    assert math.dist(out, (9.0, 9.0)) == pytest.approx(0.5)  # exactly touching the corner
    assert out[0] == pytest.approx(out[1])  # pushed straight away from the corner


def test_centre_inside_rect_is_pushed_out_nearest_edge():
    assert push_out_of_rects([9.3, 10.0], 0.5, CENTRE_BLOCK) == pytest.approx([8.5, 10.0])


def test_far_from_cover_is_unchanged_and_batches_work():
    pos = np.array([[9.0, 10.0], [5.0, 5.0]])
    out = push_out_of_rects(pos, 0.5, OBSTACLES)
    assert out == pytest.approx(np.array([[8.5, 10.0], [5.0, 5.0]]))
    assert pos[0, 0] == 9.0  # input array is not modified


def test_clamp_to_arena():
    assert clamp_to_arena([0.2, 19.9], 0.5, 20.0) == pytest.approx([0.5, 19.5])


# ---- ray vs rectangle (slab method) ------------------------------------------

def test_ray_hits_centre_block_at_t4():
    # Direction (1, -1) has t = 4 in the worked example; as a unit vector the
    # distance is 4 * sqrt(2).
    assert ray_rects(PLAYER, [S, -S], CENTRE_BLOCK) == pytest.approx(4 * math.sqrt(2))


def test_ray_misses_pillar_when_slabs_do_not_overlap():
    # x slab is t in [7.5, 8.5], y slab is t in [1, 2]: no overlap.
    assert ray_rects(PLAYER, [S, -S], LOWER_RIGHT_PILLAR) == math.inf


def test_axis_parallel_rays():
    assert ray_rects(PLAYER, [0.0, -1.0], LOWER_LEFT_BAR) == pytest.approx(3.0)  # straight up
    assert ray_rects(PLAYER, [1.0, 0.0], LOWER_RIGHT_PILLAR) == math.inf  # y=15 never in [13, 14]


def test_ray_starting_inside_or_pointing_away():
    assert ray_rects([10.0, 10.0], [1.0, 0.0], CENTRE_BLOCK) == 0.0
    assert ray_rects(PLAYER, [-S, S], CENTRE_BLOCK) == math.inf  # block is behind the ray


def test_ray_arena_exit():
    assert ray_arena_exit(PLAYER, [0.0, 1.0], 20.0) == pytest.approx(5.0)
    assert ray_arena_exit(PLAYER, [-1.0, 0.0], 20.0) == pytest.approx(5.5)


def test_raycast_in_eight_directions():
    # Up hits the lower-left bar, up-right the centre block, right is capped at
    # 10, down/down-left/down-right hit the bottom wall, left/up-left the left wall.
    dirs = np.array([[0, -1], [S, -S], [1, 0], [S, S], [0, 1], [-S, S], [-1, 0], [-S, -S]])
    origins = np.broadcast_to(PLAYER, dirs.shape)
    expected = [3.0, 4 * math.sqrt(2), 10.0, 5 * math.sqrt(2), 5.0,
                5 * math.sqrt(2), 5.5, 5.5 * math.sqrt(2)]
    assert raycast(origins, dirs, OBSTACLES, 20.0, 10.0) == pytest.approx(expected)


def test_line_of_sight():
    hidden_opponent = [15.33, 6.67]  # on the far side of the centre block
    last_seen_spot = [14.0, 10.5]
    assert not line_of_sight(PLAYER, hidden_opponent, OBSTACLES)
    assert line_of_sight(PLAYER, last_seen_spot, OBSTACLES)


def test_bullet_cannot_tunnel_through_thin_cover():
    # Both end points are outside the 1-thick bar, but the path crosses it.
    assert segment_rects_entry([6.0, 12.3], [6.0, 10.8], OBSTACLES) == pytest.approx(0.3)
    assert segment_rects_entry([6.0, 14.0], [6.0, 12.5], OBSTACLES) == math.inf  # stops short


# ---- segment vs circle -----------------------------------------------------------

def test_bullet_cannot_tunnel_through_player():
    # From 7.4 to 8.9 past a player at 8.15: both end points are 0.75 away
    # (> 0.6), yet the path goes through the centre. First contact is at 7.55.
    assert segment_circle_entry([7.4, 17.0], [8.9, 17.0], [8.15, 17.0], 0.6) == pytest.approx(0.15)


def test_bullet_passing_beside_player_misses():
    assert segment_circle_entry([7.4, 17.0], [8.9, 17.0], [8.15, 17.7], 0.6) == math.inf


def test_segment_starting_inside_circle_hits_immediately():
    assert segment_circle_entry([8.0, 17.0], [9.5, 17.0], [8.15, 17.0], 0.6) == 0.0


def test_segment_circle_batched():
    p0 = np.array([[7.4, 17.0], [7.4, 17.0]])
    p1 = np.array([[8.9, 17.0], [8.9, 17.0]])
    centres = np.array([[8.15, 17.0], [8.15, 17.7]])
    assert segment_circle_entry(p0, p1, centres, 0.6) == pytest.approx([0.15, math.inf])
