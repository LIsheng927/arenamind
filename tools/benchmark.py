"""Measure how fast the environment runs.

    python -m tools.benchmark
"""

from __future__ import annotations

import time

import numpy as np

from arena.env import ArenaEnv
from arena.obs import observe


def measure(num_envs: int, steps: int, with_obs: bool) -> float:
    env = ArenaEnv(num_envs, seed=0)
    rng = np.random.default_rng(0)
    sizes = env.cfg.action_sizes
    batches = [np.stack([rng.integers(0, k, size=(num_envs, 2)) for k in sizes], axis=-1)
               for _ in range(32)]
    start = time.perf_counter()
    for i in range(steps):
        env.step(batches[i % len(batches)])
        if with_obs:
            observe(env)
    return num_envs * steps / (time.perf_counter() - start)


def main() -> None:
    print(f"{'games':>6} {'steps/s (env)':>15} {'steps/s (env + obs)':>21}")
    for n in (1, 64, 256, 1024):
        steps = 2000 if n == 1 else 300
        print(f"{n:6d} {measure(n, steps, False):15,.0f} {measure(n, steps, True):21,.0f}")
    print("one step = one game advanced by 0.1 s (both players act)")


if __name__ == "__main__":
    main()
