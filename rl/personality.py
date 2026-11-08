"""Change how a trained policy plays, not how well, without retraining it.

Each setting adds a fixed amount to the scores of some actions before one is
drawn (DifficultyBot in rl/difficulty.py draws them), so the policy picks those
actions more or less often than it would:

    approach  added to each movement direction in proportion to how much it
              heads towards the opponent's last known position; negative
              values favour moving away. Aggressive players close in, careful
              ones keep their distance.
    hold      taken off the score of shooting while the opponent is out of
              sight. A disciplined player waits for a clear shot.
    dash      added to the score of dashing; negative values dash less.

A policy trained with personality sliders (rl/style.py) also takes

    aggression, discipline   the sliders it was trained with, given to the
                             network as inputs instead of changing its scores

Adding 1 to a score makes that action e = 2.7 times as likely relative to the
others (for approach: moving straight at the opponent against standing still).

On the command line personality settings go after the @ with the difficulty
ones (tools/bot_spec.py):

    ppo:runs/selfplay3/model.pt@approach=2,hold=4
    ppo:runs/selfplay3/model.pt@every=2.1,hold=4    # level L3, holding its fire
    ppo:runs/style/model.pt@aggression=-1           # trained to be careful
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from arena.config import MOVE_DIRECTIONS, GameConfig

from .difficulty import KNOBS, OPPONENT, OPPONENT_POSITION, Difficulty
from .style import SLIDERS

TRAITS = ("approach", "hold", "dash")  # added to the scores
OPPONENT_VISIBLE = OPPONENT  # the first number of the opponent block, see arena/obs.py
MOVES = np.array(MOVE_DIRECTIONS, dtype=np.float64)  # (9, 2) as (forward, right); 0 is standing still
_sizes = GameConfig().action_sizes


@dataclass(frozen=True)
class Personality:
    approach: float = 0.0
    hold: float = 0.0
    dash: float = 0.0
    aggression: float = 0.0
    discipline: float = 0.0

    def label(self) -> str:
        """The settings that differ from the policy's own play, as parse_settings() reads them."""
        changed = [f"{k}={v:g}" for k, v in asdict(self).items() if v != 0]
        return ",".join(changed) or "own"

    def __bool__(self) -> bool:
        return any(v != 0 for v in asdict(self).values())

    @property
    def biased(self) -> bool:
        """Whether any score is shifted (the sliders are inputs, not shifts)."""
        return any(getattr(self, t) != 0 for t in TRAITS)

    def sliders(self) -> np.ndarray:
        """The inputs for a policy trained with sliders, in the order of rl/style.py."""
        return np.array([getattr(self, s) for s in SLIDERS], dtype=np.float32)

    def biases(self, obs: np.ndarray) -> list[np.ndarray]:
        """What to add to each action head's scores, one (num_envs, choices) array per head.

        `obs` is what the policy sees (num_envs, obs_dim), after any difficulty
        settings have changed it.
        """
        n = len(obs)
        move, turn, shoot, dash = (np.zeros((n, size)) for size in _sizes)
        if self.approach:
            forward, right = obs[:, OPPONENT_POSITION[0]], obs[:, OPPONENT_POSITION[1]]
            length = np.hypot(forward, right)
            toward = np.stack([forward, right], axis=1) / np.maximum(length, 1e-9)[:, None]
            toward[length == 0] = 0.0  # standing on the spot it was last seen: no direction
            move = self.approach * toward @ MOVES.T
        if self.hold:
            shoot[:, 1] = -self.hold * (obs[:, OPPONENT_VISIBLE] < 0.5)
        if self.dash:
            dash[:, 1] = self.dash
        return [move, turn, shoot, dash]


def parse_settings(text: str) -> tuple[Difficulty, Personality]:
    """'every=2.1,hold=4' -> (Difficulty(every=2.1), Personality(hold=4.0))."""
    difficulty, personality = [], {}
    for part in filter(None, text.split(",")):
        key, sep, value = part.partition("=")
        if sep and key in (*TRAITS, *SLIDERS):
            personality[key] = float(value)
        elif sep and key in KNOBS:
            difficulty.append(part)
        else:
            known = ", ".join([*KNOBS, *TRAITS, *SLIDERS])
            raise ValueError(f"unknown setting {part!r}; use {known} (e.g. every=2,hold=4)")
    return Difficulty.parse(",".join(difficulty)), Personality(**personality)


def settings_label(difficulty: Difficulty, personality: Personality) -> str:
    """The inverse of parse_settings(): 'full' when nothing is changed."""
    parts = [p for p in (difficulty.label(), personality.label()) if p not in ("full", "own")]
    return ",".join(parts) or "full"
