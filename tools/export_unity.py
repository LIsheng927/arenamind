"""Export a trained policy for the Unity version of the game (unity/).

    python -m tools.export_unity runs/style2/model.pt --ladder runs/ladder2/ladder.json

Writes into --out (unity/Assets/ArenaMind/Resources by default):

    policy.onnx      the actor network (written by tools/onnx_mlp.py): the
                     observation of arena/obs.py and the personality sliders of
                     rl/style.py in, 18 scores out (move 9, turn 5, shoot 2,
                     dash 2)
    arenamind.json   what the game needs besides the network: the size of each
                     action head, the personalities and difficulty levels to
                     offer with their ratings, how the levels did against each
                     other (checks.csv next to the ladder), and check cases:
                     inputs from real games together with the scores PyTorch
                     gives for them. The game runs the check cases through
                     Sentis when it starts, so a broken export shows at once
                     instead of as an odd NPC; the menu ArenaMind > Play the
                     levels against each other repeats the level games in Unity.

The levels come from tools/difficulty_ladder.py (--ladder): each one is the
same network deciding only every N steps (rl/difficulty.py). Without --ladder
the game offers full strength only. The personalities are the aggression
slider at -1, 0 and 1; their ratings at each level come from the ladder's
--also settings.

Needs PyTorch. If onnxruntime is installed, the file is also run here and
compared with PyTorch.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Callable

import numpy as np

from arena.bots import BOTS
from arena.config import GameConfig
from arena.env import ArenaEnv
from arena.obs import observe
from rl.personality import parse_settings
from rl.style import SLIDERS
from tools.onnx_mlp import Layers, forward, mlp_model

OUT = Path("unity/Assets/ArenaMind/Resources")
# (name, settings): the personalities the game offers, in menu order
PERSONALITIES = (("careful", "aggression=-1"), ("neutral", ""), ("aggressive", "aggression=1"))
TOLERANCE = 1e-4  # largest difference in a score allowed between PyTorch and the export


def actor_layers(state: dict) -> Layers:
    """The actor's linear layers (actor.0, actor.2, ...) from a model's state dict, as float32."""
    index = sorted({int(k.split(".")[1]) for k in state
                    if k.startswith("actor.") and k.endswith(".weight")})
    return [(np.asarray(state[f"actor.{i}.weight"], dtype=np.float32),
             np.asarray(state[f"actor.{i}.bias"], dtype=np.float32)) for i in index]


def personalities(n_sliders: int) -> list[dict]:
    """The personalities to offer; a policy trained without sliders has its own play only."""
    if n_sliders == 0:
        return [{"name": "neutral", "settings": "", "sliders": []}]
    if n_sliders != len(SLIDERS):
        raise ValueError(f"the policy takes {n_sliders} extra inputs; sliders are {SLIDERS}")
    out = []
    for name, settings in PERSONALITIES:
        _, personality = parse_settings(settings)
        out.append({"name": name, "settings": settings,
                    "sliders": [float(v) for v in personality.sliders()]})
    return out


def levels(ladder: dict | None, offered: list[dict]) -> list[dict]:
    """The difficulty levels, easiest first, each with its rating for every personality (0 = not rated)."""
    if ladder is None:
        return [{"name": "full", "every": 1.0, "elo": [0.0] * len(offered)}]
    if ladder.get("knob", "every") != "every":
        raise ValueError(f"the game only has the 'every' knob; this ladder turns {ladder['knob']!r}")

    def rating(level: str, settings: str) -> float:
        if settings:
            rows = [a for a in ladder.get("also", []) if a["level"] == level and a["settings"] == settings]
        else:
            rows = [lv for lv in ladder["levels"] if lv["name"] == level]
        return float(rows[0]["elo"]) if rows else 0.0

    return [{"name": lv["name"], "every": float(lv["level"]),
             "elo": [rating(lv["name"], p["settings"]) for p in offered]}
            for lv in sorted(ladder["levels"], key=lambda lv: lv["level"], reverse=True)]


def level_pairs(checks_csv: Path) -> list[dict]:
    """Neighbouring levels against each other in Python (checks.csv of tools/difficulty_ladder.py)."""
    if not checks_csv.exists():
        return []
    with checks_csv.open(newline="") as f:
        return [{"stronger": row["stronger"], "weaker": row["weaker"], "actual": float(row["actual"]),
                 "noise": float(row["noise"])} for row in csv.DictReader(f) if row["neighbours"] == "True"]


def check_inputs(count: int, offered: list[dict], seed: int = 0) -> np.ndarray:
    """Inputs from a real game (rule bot against rusher), each with one personality's sliders."""
    env = ArenaEnv(1, seed=seed)
    bots = [BOTS["rule"](env, 0, seed), BOTS["rusher"](env, 1, seed + 1)]
    seen = []
    for t in range(40 * count):
        if t % 20 == 0:
            seen.extend(observe(env)[0])  # both players
        actions = np.stack([bot.act(env) for bot in bots], axis=1)
        _, done, _ = env.step(actions)
        for bot in bots:
            bot.on_done(env, done)
    obs = np.array(seen[:count], dtype=np.float32)
    sliders = np.array([offered[k % len(offered)]["sliders"] for k in range(count)], dtype=np.float32)
    return np.concatenate([obs, sliders.reshape(count, -1)], axis=1)


def export(layers: Layers, reference: Callable[[np.ndarray], np.ndarray], out: Path,
           checkpoint: str = "", ladder: dict | None = None, checks: int = 12, seed: int = 0,
           pairs: list[dict] | None = None) -> dict:
    """Write policy.onnx and arenamind.json; `reference` gives the true scores for some inputs."""
    cfg = GameConfig()
    n_in, n_out = layers[0][0].shape[1], layers[-1][0].shape[0]
    obs_dim = cfg.obs.obs_dim
    if n_out != sum(cfg.action_sizes) or n_in < obs_dim:
        raise ValueError(f"a policy for this game takes at least {obs_dim} inputs and gives "
                         f"{sum(cfg.action_sizes)} scores, not {n_in} and {n_out}")
    offered = personalities(n_in - obs_dim)
    inputs = check_inputs(checks, offered, seed)
    expected = np.asarray(reference(inputs), dtype=np.float64)
    report = {"numpy": float(np.abs(forward(layers, inputs) - expected).max())}
    if report["numpy"] > TOLERANCE:
        raise ValueError(f"the exported layers give scores up to {report['numpy']:.2g} away from "
                         "the policy's; is it a stack of linear layers with tanh between them?")

    sizes = ",".join(str(s) for s in cfg.action_sizes)
    model = mlp_model(layers, metadata={"checkpoint": checkpoint, "observation": str(obs_dim),
                                        "sliders": ",".join(SLIDERS[:n_in - obs_dim]),
                                        "action_sizes": sizes})
    try:
        import onnxruntime
    except ImportError:
        report["onnxruntime"] = None
    else:
        session = onnxruntime.InferenceSession(model, providers=["CPUExecutionProvider"])
        got = session.run(None, {"obs": inputs})[0]
        report["onnxruntime"] = float(np.abs(got - expected).max())
        if report["onnxruntime"] > TOLERANCE:
            raise ValueError(f"onnxruntime gives scores up to {report['onnxruntime']:.2g} away "
                             "from the policy's")

    out.mkdir(parents=True, exist_ok=True)
    (out / "policy.onnx").write_bytes(model)
    data = {
        "checkpoint": checkpoint,
        "inputs": n_in,
        "observation": obs_dim,
        "actionSizes": list(cfg.action_sizes),
        "sliders": list(SLIDERS[:n_in - obs_dim]),
        "personalities": offered,
        "levels": levels(ladder, offered),
        "pairs": pairs or [],
        "checks": [{"input": [float(v) for v in x], "logits": [float(np.float32(v)) for v in y]}
                   for x, y in zip(inputs, expected)],
    }
    (out / "arenamind.json").write_text(json.dumps(data, indent=1, allow_nan=False) + "\n")
    report["bytes"] = len(model)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint")
    parser.add_argument("--ladder", type=Path, help="ladder.json of tools/difficulty_ladder.py")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--checks", type=int, default=12, help="check cases for the game to run")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch

    from rl.ppo import load_policy

    model = load_policy(args.checkpoint)
    actor = model.actor
    kinds = [type(m).__name__ for m in actor]
    if kinds[1::2] != ["Tanh"] * (len(kinds) // 2) or set(kinds[::2]) != {"Linear"}:
        raise SystemExit(f"expected linear layers with tanh between them, got {kinds}")
    ladder = json.loads(args.ladder.read_text()) if args.ladder else None
    pairs = level_pairs(args.ladder.parent / "checks.csv") if args.ladder else []
    if ladder and Path(ladder["checkpoint"]) != Path(args.checkpoint):
        print(f"note: the ladder was built for {ladder['checkpoint']}, not {args.checkpoint}")

    def reference(inputs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return actor(torch.as_tensor(inputs)).numpy()

    try:
        report = export(actor_layers(model.state_dict()), reference, args.out, args.checkpoint,
                        ladder, args.checks, args.seed, pairs)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    data = json.loads((args.out / "arenamind.json").read_text())
    print(f"policy.onnx: {data['inputs']} inputs -> {sum(data['actionSizes'])} scores, "
          f"{report['bytes'] / 1024:.0f} KB")
    print(f"largest difference from PyTorch over {args.checks} check cases: numpy "
          f"{report['numpy']:.1e}, onnxruntime "
          + (f"{report['onnxruntime']:.1e}" if report["onnxruntime"] is not None else "not installed"))
    names = [p["name"] for p in data["personalities"]]
    for lv in data["levels"]:
        rated = ", ".join(f"{n} {e:.0f}" for n, e in zip(names, lv["elo"]) if e)
        print(f"  {lv['name']:<4} every {lv['every']:<5g} {rated}")
    if pairs:
        print("in Python, each level against the next weaker one: "
              + ", ".join(f"{p['stronger']}-{p['weaker']} {p['actual']:.1%}" for p in pairs))
    print(f"wrote {args.out / 'policy.onnx'} and {args.out / 'arenamind.json'}")


if __name__ == "__main__":
    main()
