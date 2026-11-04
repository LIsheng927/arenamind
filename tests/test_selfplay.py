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


def saved_tiny_model(path):
    save_checkpoint(path, tiny_model(), PPOConfig(hidden=32), step=0)
    return str(path)


def test_fixed_opponents_get_their_share_and_never_change(tmp_path):
    fixed = saved_tiny_model(tmp_path / "fixed.pt")
    env = ArenaEnv(4000, seed=3)
    cfg = SelfPlayConfig(hidden=32, p_rule=0.1, p_latest=0.4, p_fixed=0.3, past_slots=2,
                         fixed=(fixed,))
    pool = OpponentPool(env, cfg, tiny_model(), np.random.default_rng(3))
    kinds = [pool.kind(g) for g in range(env.num_envs)]
    for kind, share in [("rule", 0.1), ("latest", 0.4), ("past", 0.2), ("fixed", 0.3)]:
        assert abs(kinds.count(kind) / env.num_envs - share) < 0.03
    before = pool.models[-1].actor[0].weight.clone()
    newer = tiny_model()
    with torch.no_grad():
        for p in newer.parameters():
            p.add_(1.0)
    pool.add(newer)  # refreshes the copies of the policy...
    assert torch.equal(pool.models[0].actor[0].weight, newer.actor[0].weight)
    assert torch.equal(pool.models[-1].actor[0].weight, before)  # ...but not the fixed opponent


def test_an_exploiter_plays_nothing_but_its_fixed_opponent(tmp_path):
    fixed = saved_tiny_model(tmp_path / "main.pt")
    env = ArenaEnv(500, seed=4)
    cfg = SelfPlayConfig(hidden=32, p_rule=0.0, p_latest=0.0, p_fixed=1.0, fixed=(fixed,))
    pool = OpponentPool(env, cfg, tiny_model(), np.random.default_rng(4))
    assert {pool.kind(g) for g in range(env.num_envs)} == {"fixed"}
    pool.on_done(env, np.ones(env.num_envs, dtype=bool))  # new games: still only the fixed one
    assert {pool.kind(g) for g in range(env.num_envs)} == {"fixed"}


def test_a_fixed_share_needs_a_fixed_opponent():
    with pytest.raises(ValueError):
        OpponentPool(ArenaEnv(4, seed=0), SelfPlayConfig(hidden=32, p_fixed=0.2), tiny_model(),
                     np.random.default_rng(0))


def test_seed_pool_starts_with_an_earlier_runs_snapshots(tmp_path):
    folder = tmp_path / "pool"
    for i in range(3):
        saved_tiny_model(folder / f"iter_{i:05d}.pt")
    cfg = SelfPlayConfig(hidden=32, seed_pool=(str(folder),))
    pool = OpponentPool(ArenaEnv(8, seed=5), cfg, tiny_model(), np.random.default_rng(5))
    assert len(pool.snapshots) == 4  # three loaded + the current policy


def test_seed_pool_can_combine_several_runs(tmp_path):
    for run, n in (("a", 3), ("b", 2)):
        for i in range(n):
            saved_tiny_model(tmp_path / run / f"iter_{i:05d}.pt")
    cfg = SelfPlayConfig(hidden=32, seed_pool=(str(tmp_path / "a"), str(tmp_path / "b")))
    pool = OpponentPool(ArenaEnv(8, seed=6), cfg, tiny_model(), np.random.default_rng(6))
    assert len(pool.snapshots) == 6  # 3 + 2 loaded + the current policy


def test_seed_pool_refuses_a_folder_without_snapshots(tmp_path):
    (tmp_path / "empty").mkdir()
    cfg = SelfPlayConfig(hidden=32, seed_pool=(str(tmp_path / "empty"),))
    with pytest.raises(ValueError):
        OpponentPool(ArenaEnv(8, seed=7), cfg, tiny_model(), np.random.default_rng(7))


def test_tiny_exploiter_run_completes(tmp_path):
    main = saved_tiny_model(tmp_path / "main.pt")
    cfg = SelfPlayConfig(total_steps=2 * 16 * 32, num_envs=16, rollout_steps=32, epochs=1,
                         minibatch_size=128, hidden=32, device="cpu", eval_every=1, eval_games=8,
                         p_rule=0.0, p_latest=0.0, p_fixed=1.0, fixed=(main,), reference=main)
    train_selfplay(cfg, tmp_path / "exploiter")
    rows = (tmp_path / "exploiter" / "log.csv").read_text().strip().splitlines()
    header, last = rows[0].split(","), rows[-1].split(",")
    assert np.isfinite(float(last[header.index("eval_reference")]))
    assert float(last[header.index("games")]) >= 0
