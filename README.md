# ArenaMind

Training controllable, human-like NPCs for a 1v1 top-down shooter with
reinforcement learning: PPO from scratch, self-play, and deployment in Unity.

Status: the game environment is complete (stage 0). Training comes next.

## The environment

* Two players, five pieces of cover, 60-second games, bullets with travel time
  and a dash for dodging. Rules, actions, observations and rewards are
  described in [docs/env_spec.md](docs/env_spec.md).
* Written in numpy and vectorised over games: about 250,000 game steps per
  second for 256 parallel games on one CPU core, or 65,000 including
  observations (`python -m tools.benchmark`).
* Each player gets a 49-number, egocentric observation with only what it
  could know (no seeing through cover). The opponent's last known position
  and the time since it was seen stand in for memory.
* Rewards are zero-sum: +-1 for the result plus +-0.2 per hit.
* Fairness is tested, not assumed: every game is replayed against its
  180-degree mirror image with the players swapped, and the results must be
  mirrored exactly.
* A rule-based bot (chase, aim, strafe, dodge) is the first opponent for
  training. It beats a random player 1000 games out of 1000 and splits games
  evenly against itself.

## Usage

```bash
pip install -r requirements.txt
python -m pytest                                   # tests
python -m tools.viewer --blue rule --orange random # watch two bots play
python -m tools.play --opponent rule               # play against the rule bot
python -m tools.play --record recordings/me.npz    # ...and save your games
python -m tools.replay --out runs/replay.gif       # record a game as a GIF
python -m tools.benchmark                          # simulation speed
```

Viewer keys: Space pause, Right arrow single step, Up/Down speed, R show rays.
Play keys: WASD move, mouse aim, click or Space shoot, Shift dash.

## Layout

```
arena/      game: config, geometry, env (simulation), obs, bots, render
tools/      viewer, play, replay, benchmark
tests/      unit tests and whole-game checks
docs/       environment spec
```
