"""Checks for a trained policy played with difficulty settings. Skipped without PyTorch."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from arena.config import GameConfig  # noqa: E402
from arena.env import ArenaEnv  # noqa: E402
from arena.obs import observe  # noqa: E402
from rl.difficulty import Difficulty, DifficultyBot  # noqa: E402
from rl.model import ActorCritic  # noqa: E402
from rl.ppo import PPOConfig, save_checkpoint  # noqa: E402
from tools.bot_spec import make_bot  # noqa: E402

CFG = GameConfig()


def tiny_model():
    torch.manual_seed(0)
    return ActorCritic(CFG.obs.obs_dim, CFG.action_sizes, hidden=32)


def test_temperature_zero_takes_the_most_likely_action():
    env = ArenaEnv(16, seed=0)
    model = tiny_model()
    bot = DifficultyBot(env, 0, model=model, knobs=Difficulty(temperature=0.0))
    obs = torch.as_tensor(observe(env)[:, 0])
    greedy, _, _ = model.act(obs, greedy=True)
    assert np.array_equal(bot.act(env), greedy.numpy())


def test_same_seed_same_actions():
    env = ArenaEnv(16, seed=1)
    a = DifficultyBot(env, 0, model=tiny_model(), knobs=Difficulty(temperature=2.0), seed=5)
    b = DifficultyBot(env, 0, model=tiny_model(), knobs=Difficulty(temperature=2.0), seed=5)
    assert np.array_equal(a.act(env), b.act(env))


def test_a_spec_with_settings_plays_valid_actions(tmp_path):
    path = tmp_path / "model.pt"
    save_checkpoint(path, tiny_model(), PPOConfig(hidden=32), step=0)
    env = ArenaEnv(8, seed=2)
    bot = make_bot(f"ppo:{path}@delay=2,aim=4,every=2", env, 1, seed=0)
    assert isinstance(bot, DifficultyBot) and bot.knobs == Difficulty(delay=2, aim=4.0, every=2)
    for _ in range(20):
        actions = np.zeros((8, 2, 4), dtype=np.int64)
        actions[:, 1] = bot.act(env)
        assert ((actions[:, 1] >= 0) & (actions[:, 1] < np.array(CFG.action_sizes))).all()
        _, done, _ = env.step(actions)
        bot.on_done(env, done)
    with pytest.raises(SystemExit):
        make_bot(f"ppo-greedy:{path}@temp=2", env, 1)
