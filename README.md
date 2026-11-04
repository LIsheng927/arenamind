# ArenaMind

Training controllable, human-like NPCs for a 1v1 top-down shooter with
reinforcement learning: PPO from scratch, self-play, and deployment in Unity.

![The self-play policy (blue) against the flanker bot, which it never met in training (orange)](docs/stage2_selfplay_vs_flanker.gif)

Status: the game environment (stage 0), PPO training against a rule-based
bot (stage 1) and self-play (stage 2) are done. Difficulty and personality
settings come next.

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
python -m rl.league rule camper rusher kiter random # rate bots against each other (Elo)
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

![The stage-1 policy (blue) against the rule-based bot (orange)](docs/stage1_ppo_vs_rule.gif)

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

## Self-play (stage 2)

### The problem: a policy that only knows one opponent

Beating the rule bot 99% of the time does not mean the stage-1 policy learned
to fight. To find out, four more bots aim and shoot exactly like the rule bot
but move differently. No training run ever uses them; they are a test set.

| Bot | Moves like this |
|---|---|
| camper | stays near its spawn point and waits |
| rusher | runs straight at the opponent and dashes to close in |
| kiter | keeps 10 to 14 units away and backs off when the opponent comes closer |
| flanker | swings out to the side and keeps circling the same way |

The stage-1 policy scored 14% against the camper and 35% against the rusher:
it had learned the rule bot's habits, not the game. The flanker was kept back
as a final exam. No policy played it until all training was done.

### Training against itself

`rl/selfplay.py` uses the same PPO as stage 1, but every game draws its
opponent from a pool when it starts:

* the rule bot (20% of games), so the basics are not forgotten;
* a frozen copy of the policy, refreshed every 10 iterations (50%);
* three older copies, picked at random from every snapshot saved so far (30%
  between them), so the policy keeps beating styles it has left behind.

Self-play only fixes the weaknesses that some copy of the policy happens to
exploit. So, as in AlphaStar's league, a separate *exploiter* is trained too:
a new policy whose only opponent is a frozen copy of the main policy. Its best
score measures how exploitable the main policy is, and the main policy then
gets the exploiter as an opponent, so that the weakness is trained away.

| Run | Opponents | Steps | Time |
|---|---|---|---|
| self-play 1 | the pool above, starting from scratch | 40M | 50 min |
| exploiter 1 | frozen self-play 1 only | 20M | 17 min |
| self-play 2 | continues self-play 1 with its 123 snapshots; exploiter 1 in 20% of games, rule bot 15%, latest copy 40%, older copies 25% | 30M | 44 min |
| exploiter 2 | frozen self-play 2 only | 20M | 16 min |

Every 50 iterations the policy also plays 200 games against opponents that
never change (the rule bot and an earlier model). The pool keeps changing, so
its win rates cannot show progress; these scores can.

### Results

Score against each opponent (a win counts 1, a draw 1/2) over 400 games per
pair, 200 on each side, from one league (`python -m rl.league`):

| Opponent | Stage 1 | Self-play 1 | Self-play 2 |
|---|---|---|---|
| rule bot (in training) | 99% | 98% | 100% |
| kiter | 96% | 96% | 100% |
| camper | 14% | 78% | 97% |
| rusher | 35% | 35% | 84% |
| **flanker (final exam)** | **48%** | **95%** | **100%** |
| random | 97% | 100% | 100% |

Over 1000 games as blue (`python -m rl.evaluate`), self-play 2 won 99.3% of
its games against the flanker and all 1000 against the idle bot. The stage-1
policy won only 43.8% against the flanker, which the rule bot itself beats
about two games in three.

Elo ratings from the same league, with the rule bot pinned at 1000:

| Player | Elo |
|---|---|
| self-play 2 | 1505 |
| self-play 1 | 1219 |
| stage 1 | 1004 |
| rule bot | 1000 |
| rusher | 993 |
| kiter | 983 |
| flanker | 934 |
| camper | 851 |
| random | 458 |

How far each self-play policy could be exploited (score from 200 games per
evaluation):

| Frozen policy | Exploiter's best score against it | At the end of training | Exploiter's score against the rule bot |
|---|---|---|---|
| self-play 1 | 56.5% | 53.3% | 13.0% |
| self-play 2 | 23.0% | 19.0% | 72.5% |

![Exploiter scores during training](docs/stage2_exploiters.png)

Exploiter 1 found a cheap trick: keep its distance, because self-play 1 did
not come after it. 68% of its games ended level at the 60-second limit (a
draw), and when it landed the first hit it kept away and won on HP. It scored
only 13% against the rule bot: no good as a fighter, only as a counter to one
opponent. Self-play 2 learned to punish the trick within 1.7 million steps:
its win rate against exploiter 1 went from 16% to 100%. Exploiter 2 found no
such shortcut. To score at all it had to learn to fight, which made it a
real fighter (72.5% against the rule bot), and it still never got above 23%.

What these numbers do not show:

* The test bots share the rule bot's aim, so they test different movement,
  not different players. Human play is a harder test.
* The camper, rusher and kiter results were looked at while deciding what to
  train next, so only the flanker was truly unseen until the end.
* An exploiter's score is a lower bound: a longer run or another method might
  find more.
* Matchups are not transitive. The rusher scores only 16% against the rule
  bot but 65% against the stage-1 policy and self-play 1. Elo squeezes this
  into one number per player, and the numbers depend on who is in the
  league.
* The rusher is still the hardest opponent for self-play 2 (84%). Their
  games are short (8.4 seconds on average) and self-play 2 is hit about 3
  times per game, against 1.3 times by the rule bot.

```bash
python -m rl.selfplay --name selfplay --steps 40000000 --reference runs/first/model.pt
python -m rl.selfplay --name exploiter1 --steps 20000000 --fixed runs/selfplay/model.pt --p-fixed 1 --p-rule 0 --p-latest 0 --reference runs/selfplay/model.pt
python -m rl.selfplay --name selfplay2 --steps 30000000 --init runs/selfplay/model.pt --seed-pool runs/selfplay/pool --fixed runs/exploiter1/model.pt --p-fixed 0.2 --p-rule 0.15 --p-latest 0.4 --reference runs/selfplay/model.pt
python -m rl.selfplay --name exploiter2 --steps 20000000 --fixed runs/selfplay2/model.pt --p-fixed 1 --p-rule 0 --p-latest 0 --reference runs/selfplay2/model.pt
python -m rl.evaluate runs/selfplay2/model.pt --opponent rule camper rusher kiter idle flanker
python -m rl.league rule kiter rusher camper flanker random ppo:runs/first/model.pt ppo:runs/selfplay/model.pt ppo:runs/selfplay2/model.pt
python -m tools.plot_logs "vs self-play 1=runs/exploiter1/log.csv:eval_reference" "vs self-play 2=runs/exploiter2/log.csv:eval_reference" --percent --mark 0.5:even
```

In a self-play log, `win_vs_rule`, `win_vs_latest`, `win_vs_past` and
`win_vs_fixed` are win rates against each kind of pool opponent, and
`eval_rule` and `eval_reference` are the scores against the fixed opponents.

## Layout

```
arena/      game: config, geometry, env (simulation), obs, bots, render
rl/         PPO: model, GAE, training loop, self-play, evaluation, Elo league, policy as a bot
tools/      viewer, play, replay, benchmark, training plots
tests/      unit tests and whole-game checks
docs/       environment spec, training curves
```
