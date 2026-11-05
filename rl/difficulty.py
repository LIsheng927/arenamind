"""Make a trained policy weaker without retraining it.

Every knob works while the policy plays; the network itself is unchanged.

    temp    the scores are divided by this before an action is drawn. Above 1
            the choices spread out (weaker, but twitchy); 0 always takes the
            most likely action.
    delay   the policy sees the game as it was this many steps ago, like a
            reaction time (one step is 0.1 seconds).
    aim     the opponent's direction is seen off by an angle that drifts
            slowly, with this standard deviation in degrees, like an unsteady
            hand.
    every   the policy decides only every this many steps and repeats its last
            action in between, like a slower hand.

On the command line a policy takes its settings after an @ (tools/bot_spec.py):

    ppo:runs/selfplay3/model.pt@delay=3,aim=8
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from arena.config import GameConfig

AIM_DRIFT_SECONDS = 0.5  # how long the aim error takes to change noticeably
KNOBS = {"temp": "temperature", "delay": "delay", "aim": "aim", "every": "every"}

_obs = GameConfig().obs
OPPONENT = _obs.self_dim  # where the opponent block starts in an observation
OPPONENT_POSITION = (OPPONENT + 1, OPPONENT + 2)  # (forward, right), see arena/obs.py
OPPONENT_VELOCITY = (OPPONENT + 5, OPPONENT + 6)


@dataclass(frozen=True)
class Difficulty:
    temperature: float = 1.0
    delay: int = 0
    aim: float = 0.0
    every: int = 1

    def __post_init__(self) -> None:
        if self.temperature < 0 or self.delay < 0 or self.aim < 0 or self.every < 1:
            raise ValueError(f"invalid difficulty settings: {self}")

    @classmethod
    def parse(cls, text: str) -> "Difficulty":
        """'delay=3,aim=8' -> Difficulty(delay=3, aim=8.0). Knobs left out keep full strength."""
        values: dict[str, float | int] = {}
        for part in filter(None, text.split(",")):
            key, sep, value = part.partition("=")
            if not sep or key not in KNOBS:
                raise ValueError(f"unknown setting {part!r}; use {', '.join(KNOBS)} (e.g. delay=3)")
            name = KNOBS[key]
            values[name] = int(value) if name in ("delay", "every") else float(value)
        return cls(**values)

    def label(self) -> str:
        """The settings that differ from full strength, in the same form parse() reads."""
        default, short = asdict(Difficulty()), {v: k for k, v in KNOBS.items()}
        changed = [f"{short[k]}={v:g}" for k, v in asdict(self).items() if v != default[k]]
        return ",".join(changed) or "full"


class Handicap:
    """The knobs that do not need the network: what the policy sees, and when it decides.

    Per game it keeps the drifting aim error, the last few observations and the
    last action, and starts afresh whenever a game ends.
    """

    def __init__(self, knobs: Difficulty, num_envs: int, dt: float, seed: int | None = None) -> None:
        self.knobs = knobs
        self.rng = np.random.default_rng(seed)
        self.sd = math.radians(knobs.aim)
        self.rho = math.exp(-dt / AIM_DRIFT_SECONDS)  # step-to-step correlation of the error
        self.error = self.rng.normal(0.0, self.sd, num_envs) if self.sd else np.zeros(num_envs)
        self.history: list[np.ndarray] = []  # recent observations, oldest first
        self.last_action = np.zeros((num_envs, 4), dtype=np.int64)
        self.since = np.zeros(num_envs, dtype=np.int64)  # steps since the last decision
        self.fresh = np.ones(num_envs, dtype=bool)  # games that have just started

    def view(self, obs: np.ndarray) -> np.ndarray:
        """What the policy gets to see instead of `obs` (num_envs, obs_dim)."""
        obs = obs.copy()
        if self.sd:
            noise = self.rng.normal(0.0, self.sd, len(self.error))
            self.error = self.rho * self.error + math.sqrt(1 - self.rho ** 2) * noise
            for a, b in (OPPONENT_POSITION, OPPONENT_VELOCITY):
                obs[:, a], obs[:, b] = rotate(obs[:, a], obs[:, b], self.error)
        if self.knobs.delay:
            self.history.append(obs)
            del self.history[:-(self.knobs.delay + 1)]
            for past in self.history:  # a new game has no past yet: fill it with the present
                past[self.fresh] = obs[self.fresh]
            obs = self.history[0]
        return obs

    def deciding(self) -> np.ndarray:
        """Games in which the policy picks a new action this step."""
        return self.fresh | (self.since >= self.knobs.every)

    def choose(self, new_actions: np.ndarray) -> np.ndarray:
        """New actions where the policy decides now, the last ones elsewhere."""
        decide = self.deciding()
        self.last_action = np.where(decide[:, None], new_actions, self.last_action)
        self.since = np.where(decide, 1, self.since + 1)
        self.fresh[:] = False
        return self.last_action.copy()

    def on_done(self, done: np.ndarray) -> None:
        self.fresh |= done
        if self.sd:
            self.error[done] = self.rng.normal(0.0, self.sd, int(done.sum()))


def rotate(forward: np.ndarray, right: np.ndarray, angle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Turn (forward, right) vectors by `angle` radians."""
    cos, sin = np.cos(angle), np.sin(angle)
    return forward * cos - right * sin, forward * sin + right * cos


class DifficultyBot:
    """A trained policy played with difficulty settings (needs PyTorch)."""

    def __init__(self, env, player: int, checkpoint: str | None = None,
                 knobs: Difficulty = Difficulty(), seed: int | None = None, model=None) -> None:
        import torch

        from rl.ppo import load_policy

        if (checkpoint is None) == (model is None):
            raise ValueError("give either a checkpoint or a model")
        self.player = player
        self.knobs = knobs
        self.model = model if model is not None else load_policy(checkpoint)
        self.device = next(self.model.parameters()).device
        self.generator = torch.Generator(device=self.device)
        if seed is not None:
            self.generator.manual_seed(seed)
        self.handicap = Handicap(knobs, env.num_envs, env.cfg.dt, seed)

    def act(self, env) -> np.ndarray:
        from arena.obs import observe

        seen = self.handicap.view(observe(env)[:, self.player])
        return self.handicap.choose(self.sample(seen))

    def sample(self, obs: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            scores = self.model.actor(torch.as_tensor(obs, device=self.device))
            picks = []
            for s in torch.split(scores, self.model.action_sizes, dim=-1):
                if self.knobs.temperature == 0:
                    picks.append(s.argmax(dim=-1))
                else:
                    probs = torch.softmax(s / self.knobs.temperature, dim=-1)
                    picks.append(torch.multinomial(probs, 1, generator=self.generator).squeeze(-1))
        return torch.stack(picks, dim=-1).cpu().numpy()

    def on_done(self, env, done: np.ndarray) -> None:
        self.handicap.on_done(done)
