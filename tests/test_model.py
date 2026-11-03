"""Checks for the network and the PPO update. Skipped when PyTorch is not installed."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from arena.config import GameConfig  # noqa: E402
from arena.env import ArenaEnv  # noqa: E402
from arena.obs import observe  # noqa: E402
from rl.model import ActorCritic  # noqa: E402
from rl.ppo import PPOConfig, load_policy, ppo_update, save_checkpoint, train  # noqa: E402

CFG = GameConfig()


def make_model():
    torch.manual_seed(0)
    return ActorCritic(CFG.obs.obs_dim, CFG.action_sizes, hidden=32)


def test_shapes_and_valid_actions():
    model = make_model()
    obs = torch.as_tensor(observe(ArenaEnv(16, seed=0))[:, 0])
    actions, logp, value = model.act(obs)
    assert actions.shape == (16, 4) and logp.shape == (16,) and value.shape == (16,)
    assert (actions >= 0).all()
    assert (actions < torch.tensor(CFG.action_sizes)).all()


def test_untrained_policy_is_close_to_uniform():
    model = make_model()
    obs = torch.as_tensor(observe(ArenaEnv(8, seed=0))[:, 0])
    for head, size in zip(model.heads(obs), CFG.action_sizes):
        assert torch.allclose(head.probs, torch.full_like(head.probs, 1 / size), atol=0.02)


def test_action_log_prob_is_sum_of_heads():
    model = make_model()
    obs = torch.as_tensor(observe(ArenaEnv(4, seed=0))[:, 0])
    actions, logp, _ = model.act(obs)
    heads = model.heads(obs)
    manual = sum(torch.log(h.probs.gather(1, actions[:, i:i + 1]).squeeze(1)) for i, h in enumerate(heads))
    assert torch.allclose(logp, manual, atol=1e-5)
    # An untrained policy is almost uniform: log(1/9 * 1/5 * 1/2 * 1/2) = log(1/180).
    assert torch.allclose(logp, torch.full_like(logp, math.log(1 / 180)), atol=0.15)


def shoot_probability_after_update(clip):
    """Same observation everywhere; 'shoot' has advantage +1 and 'hold' has -1."""
    model = make_model()
    obs = torch.as_tensor(observe(ArenaEnv(1, seed=0))[:, 0]).repeat(512, 1)
    actions = torch.zeros(512, 4, dtype=torch.int64)
    actions[:, 1] = 2  # no turn
    actions[:256, 2] = 1  # first half shot, second half held fire
    with torch.no_grad():
        logp, _, _ = model.evaluate(obs, actions)
    adv = torch.cat([torch.ones(256), -torch.ones(256)])
    batch = {"obs": obs, "actions": actions, "logp": logp, "advantages": adv,
             "returns": torch.zeros(512)}
    before = model.heads(obs[:1])[2].probs[0, 1].item()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    cfg = PPOConfig(epochs=10, minibatch_size=128, clip=clip, entropy_coef=0.0)
    stats = ppo_update(model, opt, batch, cfg)
    return before, model.heads(obs[:1])[2].probs[0, 1].item(), stats


def test_ppo_update_moves_probability_towards_good_actions():
    before, after, stats = shoot_probability_after_update(clip=0.2)
    assert after > before + 0.03
    assert stats["clip_frac"] > 0


def test_clip_limits_how_far_one_batch_can_push():
    _, with_clip, _ = shoot_probability_after_update(clip=0.2)
    _, without_clip, _ = shoot_probability_after_update(clip=100.0)  # effectively no limit
    assert without_clip > with_clip + 0.03


def test_checkpoint_round_trip(tmp_path):
    model = make_model()
    save_checkpoint(tmp_path / "m.pt", model, PPOConfig(hidden=32), step=123)
    loaded = load_policy(tmp_path / "m.pt")
    obs = torch.as_tensor(observe(ArenaEnv(4, seed=1))[:, 0])
    assert torch.allclose(model.heads(obs)[0].probs, loaded.heads(obs)[0].probs)


def test_tiny_training_run_completes(tmp_path):
    cfg = PPOConfig(total_steps=2 * 16 * 32, num_envs=16, rollout_steps=32, epochs=2,
                    minibatch_size=128, hidden=32, device="cpu", save_every=1)
    train(cfg, tmp_path)
    assert (tmp_path / "model.pt").exists()
    rows = (tmp_path / "log.csv").read_text().strip().splitlines()
    assert len(rows) == 3  # header + 2 iterations
    assert np.isfinite([float(x) for x in rows[-1].split(",")[9:14]]).all()
