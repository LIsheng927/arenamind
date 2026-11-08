"""Personality sliders that a policy is trained with, so one network can play in several styles.

A style-conditioned policy gets two numbers after the game's own observation:

    aggression   -1 (careful) .. +1 (aggressive)
    discipline    0 (shoots whenever it can) .. 1 (never shoots blind)

In training (python -m rl.selfplay --style) every game draws both sliders
when it starts, and the learner's reward gets extra terms that depend on them:

    aggression a > 0   + a x AGGRESSION_HIT for every hit it lands, and
                       - a x AGGRESSION_TIME for every step the game goes on
    aggression a < 0   - |a| x CAUTION_HIT for every hit it takes
    discipline d       - d x BLIND_SHOT for every shot fired while the
                       opponent is out of sight

The game's own reward (+-1 for the result, +-0.2 per hit) stays, so every
style still plays to win; at a = 0 and d = 0 nothing is added. All the extra
terms can be scaled up together (rl.selfplay --style-scale). Because the
sliders are inputs, the network learns what each setting is rewarded for, and
a game sets them when it plays (rl/personality.py):

    ppo:runs/style/model.pt@aggression=1,discipline=1

Opponent copies in the pool get sliders of their own, drawn the same way, so
the learner also meets every style.
"""

from __future__ import annotations

import numpy as np

SLIDERS = ("aggression", "discipline")
LOW = np.array([-1.0, 0.0])
HIGH = np.array([1.0, 1.0])

AGGRESSION_HIT = 0.2  # as much again as the game's own reward for a hit
AGGRESSION_TIME = 0.001  # per step: 0.2 over a 20-second game
CAUTION_HIT = 0.3
BLIND_SHOT = 0.02  # about 0.4 a game for a policy that fires blind once a second


def sample_styles(rng: np.random.Generator, n: int) -> np.ndarray:
    """Sliders for n new games, (n, 2), each uniform over its range."""
    return rng.uniform(LOW, HIGH, size=(n, len(SLIDERS)))


def style_reward(styles: np.ndarray, hits_dealt: np.ndarray, hits_taken: np.ndarray,
                 blind_shots: np.ndarray) -> np.ndarray:
    """The extra reward for one step of every game, given each game's sliders (n, 2)."""
    aggression, discipline = styles[:, 0], styles[:, 1]
    eager, careful = np.maximum(aggression, 0.0), np.maximum(-aggression, 0.0)
    return (eager * (AGGRESSION_HIT * hits_dealt - AGGRESSION_TIME)
            - careful * CAUTION_HIT * hits_taken
            - discipline * BLIND_SHOT * blind_shots)


def with_style(obs: np.ndarray, styles: np.ndarray) -> np.ndarray:
    """Observations (n, obs_dim) with each game's sliders (n, k) appended; k may be 0."""
    return np.concatenate([obs, styles.astype(obs.dtype)], axis=-1)
