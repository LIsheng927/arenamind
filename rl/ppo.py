"""PPO training against a fixed opponent.

One iteration:
  1. play rollout_steps steps in every one of num_envs games with the current
     policy, recording observations, actions, log-probabilities, rewards,
     done flags and the critic's values;
  2. compute GAE advantages and critic targets;
  3. go over that batch `epochs` times in shuffled minibatches, each time
     minimising  clipped policy loss + value loss - entropy bonus;
  4. throw the batch away and start again.

The learner always plays player 0 (blue). The map is point-symmetric and the
observation is egocentric, so this loses nothing.
"""

from __future__ import annotations

import csv
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from arena.bots import BOTS
from arena.config import DEFAULT_CONFIG, GameConfig
from arena.env import ArenaEnv
from arena.obs import observe

from .gae import compute_gae
from .model import ActorCritic


@dataclass(frozen=True)
class PPOConfig:
    total_steps: int = 20_000_000  # game steps of experience (one per game per tick)
    num_envs: int = 256
    rollout_steps: int = 128  # per game, so 256 x 128 = 32,768 samples per iteration
    epochs: int = 4
    minibatch_size: int = 4096
    learning_rate: float = 3e-4
    anneal_lr: bool = True  # decrease the learning rate linearly to 0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 128
    opponent: str = "rule"
    seed: int = 0
    device: str = "auto"  # "auto", "cpu" or "cuda"
    save_every: int = 50  # iterations between checkpoints

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.rollout_steps


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def save_checkpoint(path: Path, model: ActorCritic, cfg: PPOConfig, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "ppo": asdict(cfg), "step": step,
                "obs_dim": model.actor[0].in_features, "action_sizes": model.action_sizes}, path)


def load_policy(path: str | Path, device: str = "cpu") -> ActorCritic:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ActorCritic(ckpt["obs_dim"], tuple(ckpt["action_sizes"]), ckpt["ppo"]["hidden"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def ppo_update(model: ActorCritic, optimizer: torch.optim.Optimizer, batch: dict,
               cfg: PPOConfig) -> dict:
    """The clipped PPO update on one batch. Returns averaged statistics."""
    n = batch["obs"].shape[0]
    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_frac": []}
    for _ in range(cfg.epochs):
        order = torch.randperm(n, device=batch["obs"].device)
        for start in range(0, n, cfg.minibatch_size):
            idx = order[start:start + cfg.minibatch_size]
            logp, entropy, value = model.evaluate(batch["obs"][idx], batch["actions"][idx])

            adv = batch["advantages"][idx]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # same scale every minibatch

            log_ratio = logp - batch["logp"][idx]
            ratio = log_ratio.exp()  # new probability / old probability
            unclipped = ratio * adv
            clipped = ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = 0.5 * (value - batch["returns"][idx]).pow(2).mean()
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(entropy.mean().item())
                stats["approx_kl"].append(((ratio - 1) - log_ratio).mean().item())
                stats["clip_frac"].append(((ratio - 1).abs() > cfg.clip).float().mean().item())
    return {k: float(np.mean(v)) for k, v in stats.items()}


def train(cfg: PPOConfig, run_dir: Path, game_cfg: GameConfig = DEFAULT_CONFIG) -> ActorCritic:
    device = pick_device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    env = ArenaEnv(cfg.num_envs, game_cfg, seed=cfg.seed)
    opponent = BOTS[cfg.opponent](env, 1, cfg.seed + 1)
    model = ActorCritic(game_cfg.obs.obs_dim, game_cfg.action_sizes, cfg.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=1e-5)

    T, N = cfg.rollout_steps, cfg.num_envs
    buf_obs = np.zeros((T, N, game_cfg.obs.obs_dim), dtype=np.float32)
    buf_act = np.zeros((T, N, 4), dtype=np.int64)
    buf_logp = np.zeros((T, N), dtype=np.float32)
    buf_val = np.zeros((T, N), dtype=np.float32)
    buf_rew = np.zeros((T, N), dtype=np.float32)
    buf_done = np.zeros((T, N), dtype=np.float32)

    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(run_dir / "log.csv", "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["step", "steps_per_sec", "games", "win_rate", "loss_rate", "draw_rate",
                  "mean_game_steps", "hits_per_game", "hit_by_per_game", "policy_loss",
                  "value_loss", "entropy", "approx_kl", "clip_frac", "learning_rate"])
    print(f"device: {device}, batch: {cfg.batch_size} samples per iteration")

    recent = deque(maxlen=500)  # (outcome, length, hits, hit_by) of recently finished games
    hits = np.zeros((N, 2))
    obs = observe(env)[:, 0]
    iterations = cfg.total_steps // cfg.batch_size
    step, start_time = 0, time.time()

    for it in range(1, iterations + 1):
        if cfg.anneal_lr:
            lr = cfg.learning_rate * (1 - (it - 1) / iterations)
            for group in optimizer.param_groups:
                group["lr"] = lr

        # 1. Collect a rollout.
        model.eval()
        for t in range(T):
            obs_t = torch.as_tensor(obs, device=device)
            actions, logp, value = model.act(obs_t)
            buf_obs[t], buf_act[t] = obs, actions.cpu().numpy()
            buf_logp[t], buf_val[t] = logp.cpu().numpy(), value.cpu().numpy()

            both = np.stack([buf_act[t], opponent.act(env)], axis=1)
            reward, done, info = env.step(both)
            opponent.on_done(env, done)
            buf_rew[t], buf_done[t] = reward[:, 0], done

            hits += np.rint(info["damage_dealt"] / game_cfg.weapon.damage)
            for g in np.flatnonzero(done):
                recent.append((info["outcome"][g, 0], info["episode_length"][g], hits[g, 0], hits[g, 1]))
                hits[g] = 0
            obs = observe(env)[:, 0]
        step += cfg.batch_size

        # 2. Advantages and critic targets.
        with torch.no_grad():
            last_values = model.value(torch.as_tensor(obs, device=device)).cpu().numpy()
        advantages, returns = compute_gae(buf_rew, buf_val, buf_done, last_values,
                                          cfg.gamma, cfg.gae_lambda)

        # 3. Learn from the batch.
        model.train()
        batch = {
            "obs": torch.as_tensor(buf_obs.reshape(-1, buf_obs.shape[-1]), device=device),
            "actions": torch.as_tensor(buf_act.reshape(-1, 4), device=device),
            "logp": torch.as_tensor(buf_logp.reshape(-1), device=device),
            "advantages": torch.as_tensor(advantages.reshape(-1), dtype=torch.float32, device=device),
            "returns": torch.as_tensor(returns.reshape(-1), dtype=torch.float32, device=device),
        }
        stats = ppo_update(model, optimizer, batch, cfg)

        # 4. Report.
        sps = step / (time.time() - start_time)
        if recent:
            res = np.array(recent)
            win, loss, draw = (res[:, 0] > 0).mean(), (res[:, 0] < 0).mean(), (res[:, 0] == 0).mean()
            length, h, hb = res[:, 1].mean(), res[:, 2].mean(), res[:, 3].mean()
        else:
            win = loss = draw = length = h = hb = float("nan")
        lr_now = optimizer.param_groups[0]["lr"]
        log.writerow([step, round(sps), len(recent), win, loss, draw, length, h, hb,
                      stats["policy_loss"], stats["value_loss"], stats["entropy"],
                      stats["approx_kl"], stats["clip_frac"], lr_now])
        log_file.flush()
        if it == 1 or it % 10 == 0 or it == iterations:
            print(f"step {step:>10,} | {sps:7,.0f} steps/s | win {win:5.1%} loss {loss:5.1%} "
                  f"draw {draw:5.1%} | hits {h:4.2f} hit-by {hb:4.2f} len {length:5.0f} | "
                  f"entropy {stats['entropy']:.2f} kl {stats['approx_kl']:.4f} "
                  f"clip {stats['clip_frac']:.2f}")
        if it % cfg.save_every == 0 or it == iterations:
            save_checkpoint(run_dir / "model.pt", model, cfg, step)

    log_file.close()
    return model
