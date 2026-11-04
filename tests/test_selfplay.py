"""Checks for self-play training. Skipped when PyTorch is not installed."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from arena.config import GameConfig  # noqa: E402
from arena.env import ArenaEnv  # noqa: E402
from arena.obs import observe  # noqa: E402
from rl.model import ActorCritic  # noqa: E402
from rl.ppo import PPOConfig, save_checkpoint  # noqa: E402
from rl.selfplay import OpponentPool, SelfPlayConfig, train_selfplay  # noqa: E402

CFG = GameConfig()


def tiny_model():
    torch.manual_seed(0)
    return ActorCritic(CFG.obs.obs_dim, CFG.action_sizes, hidden=32)


def test_pool_deals_opponents_in_the_configured_shares():
    env = ArenaEnv(4000, seed=0)
    cfg = SelfPlayConfig(hidden=32, p_rule=0.2, p_latest=0.5, past_slots=3)
    pool = OpponentPool(env, cfg, tiny_model(), np.random.default_rng(0))
    share = np.bincount(pool.slot, minlength=5) / env.num_envs
    assert abs(share[0] - 0.2) < 0.03 and abs(share[1] - 0.5) < 0.03
    assert abs(share[2:].sum() - 0.3) < 0.03


def test_pool_plays_valid_actions_and_redeals_finished_games():
    env = ArenaEnv(64, seed=1)
    cfg = SelfPlayConfig(hidden=32)
    pool = OpponentPool(env, cfg, tiny_model(), np.random.default_rng(1))
    obs = torch.as_tensor(observe(env)[:, 1])
    actions = pool.act(env, obs)
    assert actions.shape == (64, 4)
    assert ((actions >= 0) & (actions < np.array(CFG.action_sizes))).all()
    before = pool.slot.copy()
    pool.on_done(env, np.zeros(64, dtype=bool))
    assert (pool.slot == before).all()  # unfinished games keep their opponent


def test_snapshots_are_frozen_copies():
    env = ArenaEnv(8, seed=2)
    model = tiny_model()
    pool = OpponentPool(env, SelfPlayConfig(hidden=32), model, np.random.default_rng(2))
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)  # training changes the live model...
    for k, v in pool.snapshots[0].items():
        assert not torch.equal(v, model.state_dict()[k])  # ...but not the copy in the pool
    pool.add(model)
    assert len(pool.snapshots) == 2
    assert torch.equal(pool.models[0].actor[0].weight, model.actor[0].weight)


def test_tiny_selfplay_run_completes(tmp_path):
    reference = tmp_path / "ref.pt"
    save_checkpoint(reference, tiny_model(), PPOConfig(hidden=32), step=0)
    cfg = SelfPlayConfig(total_steps=3 * 16 * 32, num_envs=16, rollout_steps=32, epochs=1,
                         minibatch_size=128, hidden=32, device="cpu", save_every=1,
                         snapshot_every=1, eval_every=2, eval_games=8, past_slots=2,
                         reference=str(reference))
    train_selfplay(cfg, tmp_path / "run")
    assert (tmp_path / "run" / "model.pt").exists()
    assert len(list((tmp_path / "run" / "pool").glob("iter_*.pt"))) == 4  # start + 3 iterations
    rows = (tmp_path / "run" / "log.csv").read_text().strip().splitlines()
    header, last = rows[0].split(","), rows[-1].split(",")
    assert len(rows) == 4  # header + 3 iterations
    for name in ["eval_rule", "eval_reference", "entropy", "approx_kl"]:
        assert np.isfinite(float(last[header.index(name)]))
