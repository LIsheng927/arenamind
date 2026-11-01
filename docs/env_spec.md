# ArenaMind environment spec (v0)

A 1v1 top-down shooter. Two players, five pieces of cover, 60 seconds.
All numbers below live in `arena/config.py`; this file explains them.

## Coordinates

Screen convention: x to the right, y downward, arena `[0, 20] x [0, 20]`.
Angle 0 faces +x; positive angles turn clockwise on screen.

## Rules (the transition function)

| Item | Value | Why |
|---|---|---|
| Map | 20x20, 5 rectangles, point-symmetric about (10, 10) | Both sides face the same map, so a 50% win rate is meaningful in self-play |
| Spawns | (3, 17) and (17, 3), +-0.5 jitter, random facing | The centre block hides the players from each other at the start |
| Step | 0.1 s; each player acts once per step (10 Hz) | Close to human reaction time; keeps the horizon short |
| Episode | at most 600 steps (60 s) | Long enough for several fights |
| Player | radius 0.5, 100 HP, speed 4 units/s | - |
| Bullet | speed 15/s, lives 1 s (range 15), 20 damage, radius 0.1 | Travel time makes dodging possible |
| Shooting | 0.4 s cooldown, at most 3 bullets in flight per player | Fixed slots keep arrays a fixed size; non-stop fire never needs more than 3 |
| Dash | 3x speed for 0.3 s, 3 s cooldown | A tool for dodging |
| End | a player reaches 0 HP, or time runs out (more HP wins; equal HP is a draw) | Avoids "both hide forever" |

Both players' actions resolve in the same step, in this order:

1. read both players' actions
2. turn
3. move (a dash is split into three normal-length moves), then push players
   out of walls, cover and each other
4. advance bullets already in flight; a bullet stops at the first thing on
   its path this step (cover, wall or the opponent)
5. fire new bullets; a new bullet starts flying on the next step
6. tick timers
7. check for the end of the game
8. compute rewards
9. update what each player has seen of the other

Each stage handles both players before the next one starts, so neither player
acts on fresher information than the other. A test plays every game against
its 180-degree mirror image with the players swapped and checks that the
results are mirrored exactly.

Each player starts out knowing where the opponent spawned, since spawn points
are part of the map.

## Actions (4 independent choices)

| Head | Choices |
|---|---|
| move | 9: stay + 8 directions relative to facing |
| turn | 5: -30, -8, 0, +8, +30 degrees per step |
| shoot | 2: no / yes (ignored while on cooldown) |
| dash | 2: no / yes (ignored while on cooldown) |

## Observation (49 numbers, egocentric, roughly in [-1, 1])

| Block | Size | Contents |
|---|---|---|
| self | 10 | HP, shoot cooldown, dash cooldown, velocity (forward, right), x, y, cos, sin of facing, time left |
| opponent | 8 | visible flag, relative position (forward, right), distance, HP, relative velocity (forward, right), time since last seen |
| rays | 16 | distance to the nearest wall or cover in 16 directions (max 10) |
| enemy bullets | 3 x 5 | for the 3 nearest: present flag, relative position, relative velocity |

When the opponent is hidden, its relative position is the last place it was
seen, and "time since last seen" grows. Time left is included because the
timeout rule makes the same picture call for different actions at second 5
and second 58.

## Reward

| Event | Reward |
|---|---|
| Win / loss / draw | +1 / -1 / 0 |
| Each point of damage dealt | +0.01 (one hit = +0.2) |
| Each point of damage taken | -0.01 (one hit = -0.2) |

The game is zero-sum: the two players' rewards always add up to 0.
Damage is measured as HP actually lost, so overkill does not break this.

The damage part never disagrees with the result. A player who wins by a kill
dealt 100 damage and took at most 80; a player who wins on time has more HP
left, so it took less damage than it dealt. A win is therefore always worth at
least +1.2, a loss at most -1.2, and a draw exactly 0.

The environment reports the outcome and damage parts separately in `info`,
so training code can decay the damage part over time without changing the
environment.
