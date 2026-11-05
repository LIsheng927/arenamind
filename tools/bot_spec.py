"""Turn a bot name from the command line into a bot.

    random | rule | idle                built-in bots (idle stands still, never shoots)
    camper | rusher | kiter | flanker   rule-bot variants that move differently
    ppo:<checkpoint>                    a trained policy, sampling its actions
    ppo-greedy:<checkpoint>             the same policy, always its most likely action
    ppo:<checkpoint>@delay=3,aim=8      a trained policy with difficulty settings
                                        (temp, delay, aim, every; see rl/difficulty.py)
"""

from __future__ import annotations

from arena.bots import BOTS

HELP = ("random, rule, idle, camper, rusher, kiter, flanker, ppo:<path to model.pt>, "
        "ppo-greedy:<path>, or either with difficulty settings, e.g. ppo:<path>@delay=3,aim=8")


def make_bot(spec: str, env, player: int, seed: int | None = None):
    for prefix, greedy in (("ppo:", False), ("ppo-greedy:", True)):
        if spec.startswith(prefix):
            path, _, settings = spec[len(prefix):].partition("@")
            if settings:
                from rl.difficulty import Difficulty, DifficultyBot  # needs PyTorch
                try:
                    knobs = Difficulty.parse(settings)
                except ValueError as error:
                    raise SystemExit(str(error)) from None
                if greedy:
                    if knobs.temperature != 1.0:
                        raise SystemExit("ppo-greedy always takes the most likely action; drop temp=")
                    knobs = Difficulty(0.0, knobs.delay, knobs.aim, knobs.every)
                return DifficultyBot(env, player, path, knobs, seed)
            from rl.policy_bot import PolicyBot  # needs PyTorch, so only imported when used
            return PolicyBot(env, player, path, greedy=greedy, seed=seed)
    if spec not in BOTS:
        raise SystemExit(f"unknown bot {spec!r}; use {HELP}")
    return BOTS[spec](env, player, seed)
