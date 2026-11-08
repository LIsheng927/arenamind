"""Use a trained policy like any other bot (viewer, matches, play, evaluation)."""

from __future__ import annotations

import numpy as np
import torch

from arena.env import ArenaEnv
from arena.obs import observe

from .model import ActorCritic
from .ppo import load_policy


class PolicyBot:
    """A policy from a checkpoint file, or a model already in memory (`model=`)."""

    def __init__(self, env: ArenaEnv, player: int, checkpoint: str | None = None,
                 greedy: bool = False, seed: int | None = None,
                 model: ActorCritic | None = None) -> None:
        if (checkpoint is None) == (model is None):
            raise ValueError("give either a checkpoint or a model")
        self.player = player
        self.greedy = greedy
        self.model = model if model is not None else load_policy(checkpoint)
        self.device = next(self.model.parameters()).device
        if seed is not None:
            torch.manual_seed(seed)

    def act(self, env: ArenaEnv) -> np.ndarray:
        obs = observe(env)[:, self.player]
        extra = self.model.actor[0].in_features - obs.shape[1]
        if extra > 0:  # trained with personality sliders (rl/style.py): play them at 0
            obs = np.concatenate([obs, np.zeros((len(obs), extra), dtype=obs.dtype)], axis=1)
        actions, _, _ = self.model.act(torch.as_tensor(obs, device=self.device), greedy=self.greedy)
        return actions.cpu().numpy()

    def on_done(self, env: ArenaEnv, done: np.ndarray) -> None:
        pass
