# ArenaMind

Training controllable, human-like NPCs for a 1v1 top-down shooter with
reinforcement learning: PPO from scratch, self-play, and deployment in Unity.

![A trained policy (blue) against the rule-based bot (orange)](docs/stage1_ppo_vs_rule.gif)

Status: the game environment (stage 0) and PPO training against a rule-based
bot (stage 1) are done. Self-play comes next.

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
Play keys: WASD or arrow keys move, mouse aim, click or Space shoot, Shift dash.

## Training (stage 1)

PPO written from scratch in PyTorch (`rl/`): separate actor and critic
networks, one action head per choice (move, turn, shoot, dash), GAE
advantages, clipped policy updates, an entropy bonus and learning-rate
annealing. 256 games run in parallel against the rule-based bot.

20 million steps took about 10 minutes (about 33,000 steps per second with
an RTX 5080).

![Training curves](docs/stage1_training.png)

The policy learned in three phases. First it stopped dying: hits taken fell
from 5 per game to under 1, and games ran into the 60-second limit. Then it
learned to hit back, and the win rate rose from 10% to 99% between 1 and 3.5
million steps. After that it kept winning, but faster: the average game went
from 59 to 17 seconds. Rewards are discounted by 1% per step (gamma = 0.99),
so a quick win is worth more than a slow one.

Results over 1000 games each (`python -m rl.evaluate`):

| Opponent | Policy plays | Win | Loss | Draw |
|---|---|---|---|---|
| rule bot | blue | 99.1% | 0.9% | 0% |
| rule bot | orange | 99.2% | 0.8% | 0% |
| idle bot (stands still, never shoots) | blue | 96.7% | 0% | 3.3% |
| idle bot, most likely action only (`--greedy`) | blue | 100% | 0% | 0% |

The policy was trained as blue only; the orange result shows that it can
play either side (see the observation notes in the spec). The idle bot checks
that it goes looking for an opponent that does not come to it. A draw there
means it never landed a hit in 60 seconds; this only happened with sampled
actions.

Install PyTorch for your machine first (see pytorch.org), then:

```bash
python -m rl.train --name first --steps 20000000   # progress in runs/first/log.csv
python -m rl.evaluate runs/first/model.pt --side orange --opponent rule
python -m tools.viewer --blue ppo:runs/first/model.pt --orange rule
python -m tools.play --opponent ppo-greedy:runs/first/model.pt
python -m tools.plot_training runs/first/log.csv   # curves like the ones above
```

`ppo:` samples actions from the policy; `ppo-greedy:` always takes the most
likely one.

## Layout

```
arena/      game: config, geometry, env (simulation), obs, bots, render
rl/         PPO: model, GAE, training loop, evaluation, policy as a bot
tools/      viewer, play, replay, benchmark, training plots
tests/      unit tests and whole-game checks
docs/       environment spec, training curves
```
