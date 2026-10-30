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
turn, move, collide, advance existing bullets (hits), fire new bullets,
tick timers, check for the end, compute rewards, build observations.

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
The environment reports the outcome and damage parts separately in `info`,
so training code can decay the damage part over time without changing the
environment.
