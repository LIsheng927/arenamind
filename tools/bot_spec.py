"""Turn a bot name from the command line into a bot.

    random | rule | idle      built-in bots (idle stands still and never shoots)
    ppo:<checkpoint>          a trained policy, sampling its actions
    ppo-greedy:<checkpoint>   the same policy, always taking its most likely action
"""

from __future__ import annotations

from arena.bots import BOTS

HELP = "random, rule, idle, ppo:<path to model.pt> or ppo-greedy:<path>"


def make_bot(spec: str, env, player: int, seed: int | None = None):
    for prefix, greedy in (("ppo:", False), ("ppo-greedy:", True)):
        if spec.startswith(prefix):
            from rl.policy_bot import PolicyBot  # needs PyTorch, so only imported when used
            return PolicyBot(env, player, spec[len(prefix):], greedy=greedy, seed=seed)
    if spec not in BOTS:
        raise SystemExit(f"unknown bot {spec!r}; use {HELP}")
    return BOTS[spec](env, player, seed)
