"""The actor (policy) and the critic (value function).

Actor: 49 observation numbers -> hidden layers -> 18 scores, split into four
heads (move 9, turn 5, shoot 2, dash 2). Each head is turned into
probabilities with a softmax and sampled on its own; the four choices make up
one action. The log-probability of a whole action is the sum of the four
heads' log-probabilities.

Critic: 49 observation numbers -> hidden layers -> 1 number, the expected
return from this situation.

The two are separate networks, so learning to predict values never disturbs
the policy and vice versa.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Categorical


def linear(n_in: int, n_out: int, gain: float = math.sqrt(2)) -> nn.Linear:
    layer = nn.Linear(n_in, n_out)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


def mlp(n_in: int, hidden: int, n_out: int, out_gain: float) -> nn.Sequential:
    return nn.Sequential(
        linear(n_in, hidden), nn.Tanh(),
        linear(hidden, hidden), nn.Tanh(),
        linear(hidden, n_out, out_gain),
    )


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_sizes: tuple[int, ...], hidden: int = 128) -> None:
        super().__init__()
        self.action_sizes = list(action_sizes)
        # A tiny output gain makes every head start almost uniform: an untrained
        # policy behaves like the random bot and explores everything.
        self.actor = mlp(obs_dim, hidden, sum(action_sizes), out_gain=0.01)
        self.critic = mlp(obs_dim, hidden, 1, out_gain=1.0)

    def heads(self, obs: torch.Tensor) -> list[Categorical]:
        scores = self.actor(obs)
        return [Categorical(logits=s) for s in torch.split(scores, self.action_sizes, dim=-1)]

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor, greedy: bool = False):
        """Pick actions. Returns (actions (N, 4), log-probabilities (N,), values (N,))."""
        heads = self.heads(obs)
        if greedy:
            actions = torch.stack([h.probs.argmax(dim=-1) for h in heads], dim=-1)
        else:
            actions = torch.stack([h.sample() for h in heads], dim=-1)
        logp = sum(h.log_prob(actions[..., i]) for i, h in enumerate(heads))
        return actions, logp, self.value(obs)

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        """For training: log-probabilities, entropies and values of given actions."""
        heads = self.heads(obs)
        logp = sum(h.log_prob(actions[..., i]) for i, h in enumerate(heads))
        entropy = sum(h.entropy() for h in heads)
        return logp, entropy, self.value(obs)
