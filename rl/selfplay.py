"""PPO with self-play: the policy trains against copies of itself.

    python -m rl.selfplay --name selfplay --steps 40000000 --reference runs/first/model.pt
    python -m rl.selfplay --name sp-smoke --steps 300000      # quick check that it runs

Training against one fixed bot teaches a policy that bot's habits (stage 1:
99% against the rule bot, 9% against the camper bot). Here the opponents come
from a pool that grows as the policy improves:

  * slot 0, "rule":   the rule bot, so the basics are not forgotten;
  * slot 1, "latest": a frozen copy of the policy, refreshed every
                      `snapshot_every` iterations;
  * slots 2..,"past": older copies, drawn at random from every snapshot so far
                      whenever the pool is refreshed, so that the policy keeps
                      beating the styles it has already left behind.

Each game draws its slot when it starts (p_rule, p_latest, the rest shared by
the past slots). A game in progress keeps its slot when the pool is refreshed,
so it may switch to a newer copy halfway through; that happens at most once
every `snapshot_every` iterations.

Every snapshot is saved to runs/<name>/pool/ for the league to rate later.
Every `eval_every` iterations the current policy also plays fixed reference
opponents (the rule bot, and optionally an earlier model). The pool changes
all the time, so its win rates cannot show progress; the references never
change, so their numbers can be compared across the whole run.

The camper, rusher, kiter and idle bots are never used here. They are kept for
testing what the policy learned, like a held-out test set.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from arena.bots import RuleBot
from arena.config import DEFAULT_CONFIG
from arena.env import ArenaEnv
from arena.obs import observe

from .gae import compute_gae
from .model import ActorCritic
from .policy_bot import PolicyBot
from .ppo import PPOConfig, load_policy, pick_device, ppo_update, save_checkpoint

KINDS = ("rule", "latest", "past")


@dataclass(frozen=True)
class SelfPlayConfig(PPOConfig):
    total_steps: int = 40_000_000
    snapshot_every: int = 10  # iterations between copies added to the pool
    past_slots: int = 3  # older copies in play at any time
    p_rule: float = 0.2  # share of games against the rule bot
    p_latest: float = 0.5  # share against the latest copy; the rest go to past copies
    eval_every: int = 50  # iterations between games against the fixed references
    eval_games: int = 200
    reference: str = ""  # checkpoint of an earlier policy to measure against ("" = none)
    init: str = ""  # start from this checkpoint instead of from scratch ("" = scratch)


class OpponentPool:
    """Decides who every game is played against, and plays those opponents."""

    def __init__(self, env: ArenaEnv, cfg: SelfPlayConfig, model: ActorCritic,
                 rng: np.random.Generator) -> None:
        if cfg.past_slots < 1 or not 0 <= cfg.p_rule + cfg.p_latest <= 1:
            raise ValueError("need at least one past slot and p_rule + p_latest <= 1")
        self.rng = rng
        self.rule = RuleBot(env, 1, cfg.seed + 1)
        device = next(model.parameters()).device
        # Slot k >= 1 is played by self.models[k - 1].
        self.models = [ActorCritic(model.actor[0].in_features, model.action_sizes, cfg.hidden).to(device)
                       for _ in range(1 + cfg.past_slots)]
        past = (1.0 - cfg.p_rule - cfg.p_latest) / cfg.past_slots
        self.probs = np.array([cfg.p_rule, cfg.p_latest] + [past] * cfg.past_slots)
        self.snapshots: list[dict] = []
        self.add(model)
        self.slot = self.rng.choice(len(self.probs), size=env.num_envs, p=self.probs)

    def add(self, model: ActorCritic) -> None:
        """Copy the current weights into the pool and deal the slots out again."""
        self.snapshots.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        self.models[0].load_state_dict(self.snapshots[-1])
        for m in self.models[1:]:
            m.load_state_dict(self.snapshots[self.rng.integers(len(self.snapshots))])
        for m in self.models:
            m.eval()

    def kind(self, game: int) -> str:
        return KINDS[min(int(self.slot[game]), 2)]

    def act(self, env: ArenaEnv, obs: torch.Tensor) -> np.ndarray:
        """Actions of every game's opponent; obs = player 1's observations."""
        actions = self.rule.act(env)  # in every game, so its memory stays up to date
        for k, m in enumerate(self.models, start=1):
            games = np.flatnonzero(self.slot == k)
            if games.size:
                a, _, _ = m.act(obs[torch.as_tensor(games, device=obs.device)])
                actions[games] = a.cpu().numpy()
        return actions

    def on_done(self, env: ArenaEnv, done: np.ndarray) -> None:
        self.rule.on_done(env, done)
        self.slot[done] = self.rng.choice(len(self.probs), size=int(done.sum()), p=self.probs)


def score_against(model: ActorCritic, make_opponent, games: int, seed: int) -> float:
    """Wins plus half the draws, as a share of `games`, playing as blue."""
    env = ArenaEnv(games, seed=seed)
    bots = [PolicyBot(env, 0, model=model), make_opponent(env)]
    finished = np.zeros(games, dtype=bool)
    outcome = np.zeros(games)
    while not finished.all():
        _, done, info = env.step(np.stack([b.act(env) for b in bots], axis=1))
        new = done & ~finished
        outcome[new] = info["outcome"][new, 0]
        finished |= done
        for b in bots:
            b.on_done(env, done)
    return float(((outcome > 0) + 0.5 * (outcome == 0)).mean())


def train_selfplay(cfg: SelfPlayConfig, run_dir: Path) -> ActorCritic:
    game_cfg = DEFAULT_CONFIG
    device = pick_device(cfg.device)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    env = ArenaEnv(cfg.num_envs, game_cfg, seed=cfg.seed)
    model = ActorCritic(game_cfg.obs.obs_dim, game_cfg.action_sizes, cfg.hidden).to(device)
    if cfg.init:
        model.load_state_dict(torch.load(cfg.init, map_location=device, weights_only=False)["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=1e-5)
    pool = OpponentPool(env, cfg, model, rng)
    reference = load_policy(cfg.reference).to(device) if cfg.reference else None

    T, N = cfg.rollout_steps, cfg.num_envs
    buf_obs = np.zeros((T, N, game_cfg.obs.obs_dim), dtype=np.float32)
    buf_act = np.zeros((T, N, 4), dtype=np.int64)
    buf_logp = np.zeros((T, N), dtype=np.float32)
    buf_val = np.zeros((T, N), dtype=np.float32)
    buf_rew = np.zeros((T, N), dtype=np.float32)
    buf_done = np.zeros((T, N), dtype=np.float32)

    run_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(run_dir / "pool" / "iter_00000.pt", model, cfg, 0)
    log_file = open(run_dir / "log.csv", "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["step", "steps_per_sec", "games", "win_rate", "loss_rate", "draw_rate",
                  "win_vs_rule", "win_vs_latest", "win_vs_past", "mean_game_steps", "snapshots",
                  "eval_rule", "eval_reference", "policy_loss", "value_loss", "entropy",
                  "approx_kl", "clip_frac", "learning_rate"])
    print(f"device: {device}, batch: {cfg.batch_size} samples per iteration, "
          f"reference: {cfg.reference or 'none'}, start: {cfg.init or 'from scratch'}")

    recent = deque(maxlen=500)  # (opponent kind, outcome, length) of recently finished games
    obs_all = observe(env)
    iterations = cfg.total_steps // cfg.batch_size
    step, start_time = 0, time.time()
    eval_rule = eval_ref = float("nan")

    for it in range(1, iterations + 1):
        if cfg.anneal_lr:
            for group in optimizer.param_groups:
                group["lr"] = cfg.learning_rate * (1 - (it - 1) / iterations)

        # 1. Collect a rollout: the learner is blue, each game's opponent comes from the pool.
        for t in range(T):
            obs_t = torch.as_tensor(obs_all[:, 0], device=device)
            actions, logp, value = model.act(obs_t)
            buf_obs[t], buf_act[t] = obs_all[:, 0], actions.cpu().numpy()
            buf_logp[t], buf_val[t] = logp.cpu().numpy(), value.cpu().numpy()

            opponent = pool.act(env, torch.as_tensor(obs_all[:, 1], device=device))
            reward, done, info = env.step(np.stack([buf_act[t], opponent], axis=1))
            buf_rew[t], buf_done[t] = reward[:, 0], done
            for g in np.flatnonzero(done):  # before on_done deals the game a new opponent
                recent.append((pool.kind(g), info["outcome"][g, 0], info["episode_length"][g]))
            pool.on_done(env, done)
            obs_all = observe(env)
        step += cfg.batch_size

        # 2. Advantages and critic targets, 3. the PPO update: exactly as in stage 1.
        with torch.no_grad():
            last_values = model.value(torch.as_tensor(obs_all[:, 0], device=device)).cpu().numpy()
        advantages, returns = compute_gae(buf_rew, buf_val, buf_done, last_values,
                                          cfg.gamma, cfg.gae_lambda)
        batch = {
            "obs": torch.as_tensor(buf_obs.reshape(-1, buf_obs.shape[-1]), device=device),
            "actions": torch.as_tensor(buf_act.reshape(-1, 4), device=device),
            "logp": torch.as_tensor(buf_logp.reshape(-1), device=device),
            "advantages": torch.as_tensor(advantages.reshape(-1), dtype=torch.float32, device=device),
            "returns": torch.as_tensor(returns.reshape(-1), dtype=torch.float32, device=device),
        }
        model.train()
        stats = ppo_update(model, optimizer, batch, cfg)
        model.eval()

        # 4. Grow the pool.
        if it % cfg.snapshot_every == 0:
            pool.add(model)
            save_checkpoint(run_dir / "pool" / f"iter_{it:05d}.pt", model, cfg, step)

        # 5. Measure against opponents that never change.
        if it % cfg.eval_every == 0 or it == iterations:
            eval_rule = score_against(model, lambda e: RuleBot(e, 1, it), cfg.eval_games, seed=it)
            if reference is not None:
                eval_ref = score_against(model, lambda e: PolicyBot(e, 1, model=reference),
                                         cfg.eval_games, seed=it)

        # 6. Report.
        sps = step / (time.time() - start_time)
        res = list(recent)
        outcome = np.array([r[1] for r in res], dtype=float) if res else np.array([np.nan])
        by_kind = {k: np.array([r[1] for r in res if r[0] == k], dtype=float) for k in KINDS}
        win_vs = {k: float((v > 0).mean()) if v.size else float("nan") for k, v in by_kind.items()}
        length = float(np.mean([r[2] for r in res])) if res else float("nan")
        lr_now = optimizer.param_groups[0]["lr"]
        log.writerow([step, round(sps), len(res), (outcome > 0).mean(), (outcome < 0).mean(),
                      (outcome == 0).mean(), win_vs["rule"], win_vs["latest"], win_vs["past"],
                      length, len(pool.snapshots), eval_rule, eval_ref, stats["policy_loss"],
                      stats["value_loss"], stats["entropy"], stats["approx_kl"],
                      stats["clip_frac"], lr_now])
        log_file.flush()
        if it == 1 or it % 10 == 0 or it == iterations:
            print(f"step {step:>11,} | {sps:6,.0f}/s | win vs rule {win_vs['rule']:5.1%} "
                  f"latest {win_vs['latest']:5.1%} past {win_vs['past']:5.1%} | "
                  f"eval rule {eval_rule:5.1%} ref {eval_ref:5.1%} | len {length:4.0f} | "
                  f"pool {len(pool.snapshots):3d} | entropy {stats['entropy']:.2f} "
                  f"kl {stats['approx_kl']:.4f}")
        if it % cfg.save_every == 0 or it == iterations:
            save_checkpoint(run_dir / "model.pt", model, cfg, step)

    log_file.close()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="selfplay", help="folder name under runs/")
    parser.add_argument("--steps", type=int, default=SelfPlayConfig.total_steps)
    parser.add_argument("--reference", default="", help="earlier model to measure against")
    parser.add_argument("--init", default="", help="start from this model instead of from scratch")
    parser.add_argument("--p-rule", type=float, default=SelfPlayConfig.p_rule)
    parser.add_argument("--p-latest", type=float, default=SelfPlayConfig.p_latest)
    parser.add_argument("--past-slots", type=int, default=SelfPlayConfig.past_slots)
    parser.add_argument("--snapshot-every", type=int, default=SelfPlayConfig.snapshot_every)
    parser.add_argument("--num-envs", type=int, default=SelfPlayConfig.num_envs)
    parser.add_argument("--seed", type=int, default=SelfPlayConfig.seed)
    parser.add_argument("--device", default=SelfPlayConfig.device, choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    cfg = SelfPlayConfig(total_steps=args.steps, reference=args.reference, init=args.init,
                         p_rule=args.p_rule, p_latest=args.p_latest, past_slots=args.past_slots,
                         snapshot_every=args.snapshot_every, num_envs=args.num_envs,
                         seed=args.seed, device=args.device)
    train_selfplay(cfg, Path("runs") / args.name)


if __name__ == "__main__":
    main()
