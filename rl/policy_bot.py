"""Use a trained policy like any other bot (viewer, matches, play)."""

from __future__ import annotations

import numpy as np
import torch

from arena.env import ArenaEnv
from arena.obs import observe

from .ppo import load_policy


class PolicyBot:
    def __init__(self, env: ArenaEnv, player: int, checkpoint: str, greedy: bool = False,
                 seed: int | None = None) -> None:
        self.player = player
        self.greedy = greedy
        self.model = load_policy(checkpoint)
        if seed is not None:
            torch.manual_seed(seed)

    def act(self, env: ArenaEnv) -> np.ndarray:
        obs = torch.as_tensor(observe(env)[:, self.player])
        actions, _, _ = self.model.act(obs, greedy=self.greedy)
        return actions.numpy()

    def on_done(self, env: ArenaEnv, done: np.ndarray) -> None:
        pass
