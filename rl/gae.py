"""Generalized Advantage Estimation (GAE).

For each step t of a rollout:

    delta(t) = r(t) + gamma * V(t+1) * (1 - done(t)) - V(t)
        how much better this step went than the critic expected
    A(t) = delta(t) + gamma * lam * (1 - done(t)) * A(t+1)
        this step's surprise plus the discounted surprises that follow

Both are computed backwards from the last step. done(t) means the game ended
at step t, so nothing after it (the next game) is counted. lam = 1 gives the
plain return minus the critic's estimate; lam = 0 looks only one step ahead.

The critic is trained towards returns = A + V.
"""

from __future__ import annotations

import numpy as np


def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,
                last_values: np.ndarray, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Advantages and critic targets for a rollout.

    rewards, values, dones: shape (T, N), one column per parallel game.
    last_values: shape (N,), the critic's estimate for the state after the
    last step (used only for games that had not ended by then).
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    not_done = 1.0 - np.asarray(dones, dtype=np.float64)
    steps = rewards.shape[0]

    advantages = np.zeros_like(rewards)
    running = np.zeros(rewards.shape[1:], dtype=np.float64)
    for t in reversed(range(steps)):
        next_values = last_values if t == steps - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_values * not_done[t] - values[t]
        running = delta + gamma * lam * not_done[t] * running
        advantages[t] = running
    return advantages, advantages + values
