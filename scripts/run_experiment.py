"""Run one fresh, seeded training + greedy evaluation for a MountainCar agent.

Standalone experiment runner shared by the two agents of this repository
(``qlearning`` and ``dqn``). It trains with a fixed seed, evaluates the greedy
policy (no exploration) on a disjoint, separately seeded range, and writes
per-episode CSVs, a JSON summary and the FINAL checkpoint into a brand-new
output directory.

Usage (from the repository root):

    uv run --frozen python scripts/run_experiment.py qlearning \
        --episodes 2000 --seed 42 --eval-episodes 100 \
        --output-dir results/exp-qlearning-seed42

    uv run --frozen python scripts/run_experiment.py dqn \
        --episodes 700 --seed 42 --eval-episodes 100 \
        --output-dir results/exp-dqn-seed42

Contract highlights:

- ``--episodes`` and ``--output-dir`` are required and ``--output-dir`` must be a
  brand-new path (an existing file, empty directory or symlink is rejected with
  ``mkdir(exist_ok=False)`` before anything is written).
- Counts and seed values/ranges (NumPy accepted seeds ``[0, 2**32)``) and the
  disjointness of the training/evaluation seed ranges are validated BEFORE any
  side effect or training.
- The checkpoint is always the FINAL trained policy, never a "best" selection.
- On failure the partially written directory is left untouched; the next attempt
  must use a new ``--output-dir`` (no resumption flag).
"""
import argparse
import csv
import importlib.metadata
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from mountain_car.agents import DQNAgent, QLearningAgent

ENV_ID = "MountainCar-v0"

# NumPy accepts integer seeds in [0, 2**32).
MAX_SEED = 2**32 - 1
DEFAULT_TRAIN_SEED = 42
DEFAULT_EVAL_EPISODES = 100

# Column schema shared by the training and evaluation per-episode CSVs.
CSV_FIELDS = ("episode", "seed", "return", "steps", "terminated", "truncated", "epsilon", "time")

# Agent name -> (class, checkpoint extension, initial hyperparameter attributes).
_QL_HPARAMS = ("n_bins", "lr", "gamma", "epsilon_end", "epsilon_decay")
_DQN_HPARAMS = (
    "lr",
    "gamma",
    "epsilon_end",
    "epsilon_decay",
    "batch_size",
    "buffer_capacity",
    "target_update_freq",
    "hidden",
    "repeat_prob",
)
AGENTS = {
    "qlearning": (QLearningAgent, ".pkl", _QL_HPARAMS),
    "dqn": (DQNAgent, ".pt", _DQN_HPARAMS),
}


class UsageError(Exception):
    """Invalid CLI arguments detected after argparse parsing."""


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"must be an integer, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _require_seed_range(name: str, start: int, count: int) -> int:
    """Validate ``start .. start + count - 1`` fits NumPy's accepted seed range."""
    end = start + count - 1
    if start < 0 or end > MAX_SEED:
        raise UsageError(
            f"{name} seed range [{start}, {end}] exceeds NumPy's accepted seeds 0..{MAX_SEED}"
        )
    return end


def resolve_and_validate(args: argparse.Namespace) -> tuple[int, int]:
    """Resolve the effective seed pair and reject invalid or overlapping ranges.

    The default evaluation seed is ``train seed + episodes``: the first seed
    outside the training range, so the two ranges never share a seed. Ranges are
    also allowed to be disjoint with evaluation before training.
    """
    train_seed = args.seed
    eval_seed = args.eval_seed if args.eval_seed is not None else train_seed + args.episodes
    train_end = _require_seed_range("training", train_seed, args.episodes)
    eval_end = _require_seed_range("evaluation", eval_seed, args.eval_episodes)
    if not (train_end < eval_seed or eval_end < train_seed):
        raise UsageError(
            f"training seeds [{train_seed}, {train_end}] and evaluation seeds "
            f"[{eval_seed}, {eval_end}] overlap; pass --eval-seed outside the training range"
        )
    return train_seed, eval_seed


def _write_episode_csv(path: Path, rows: list[dict]) -> None:
    """Write per-episode rows with the shared CSV schema (flags as 0/1)."""
    try:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
            writer.writeheader()
            for row in rows:
                out = dict(row)
                out["terminated"] = int(out["terminated"])
                out["truncated"] = int(out["truncated"])
                writer.writerow(out)
    except OSError as exc:
        raise RuntimeError(f"could not write episode CSV {path}: {exc}") from exc


def evaluate(agent, eval_seed: int, eval_episodes: int) -> list[dict]:
    """Play ``eval_episodes`` greedy episodes, each reset with a distinct seed.

    ``epsilon`` is reported as 0: an effective greedy policy, independent of the
    agent's training exploration rate. ``terminated`` marks flag reached (not
    ``reward == -200``). The evaluation environment is closed in ``finally`` so
    an error still releases it.
    """
    rows: list[dict] = []
    env = gym.make(ENV_ID)
    try:
        for episode in range(1, eval_episodes + 1):
            episode_seed = eval_seed + episode - 1
            started = time.perf_counter()
            obs, _ = env.reset(seed=episode_seed)
            total = 0.0
            steps = 0
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(int(action))
                total += float(reward)
                steps += 1
            rows.append(
                {
                    "episode": episode,
                    "seed": episode_seed,
                    "return": total,
                    "steps": steps,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "epsilon": 0,
                    "time": time.perf_counter() - started,
                }
            )
    finally:
        env.close()
    return rows


def _statistics(rows: list[dict]) -> dict:
    """Plain-JSON statistics recomputable from a per-episode CSV (population std)."""
    returns = np.asarray([float(row["return"]) for row in rows], dtype=float)
    steps = np.asarray([float(row["steps"]) for row in rows], dtype=float)
    return {
        "episodes": len(rows),
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "return_min": float(np.min(returns)),
        "return_max": float(np.max(returns)),
        "success_count": sum(1 for row in rows if bool(row["terminated"])),
        "steps_mean": float(np.mean(steps)),
    }


def _write_summary(path: Path, summary: dict) -> None:
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
    except OSError as exc:
        raise RuntimeError(f"could not write summary {path}: {exc}") from exc


def run(args: argparse.Namespace) -> int:
    train_seed, eval_seed = resolve_and_validate(args)
    cls, suffix, hparam_attrs = AGENTS[args.agent]
    output_dir = Path(args.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except (FileExistsError, NotADirectoryError) as exc:
        raise UsageError(
            f"output directory already exists (or is not a directory): {output_dir}. "
            "The runner only writes into a brand-new directory; choose a new --output-dir."
        ) from exc

    if args.agent == "dqn":
        # The constructor seed fixes weight initialization; train() re-seeds the
        # exploration/replay RNGs, so both must share the training seed.
        agent = cls(ENV_ID, seed=train_seed)
    else:
        agent = cls(ENV_ID)

    # Hyperparameters (incl. epsilon_start) captured BEFORE any training.
    hparams = {"epsilon_start": float(agent.epsilon)}
    hparams.update({name: getattr(agent, name) for name in hparam_attrs})

    print(
        f"[run_experiment] agent={args.agent} env={ENV_ID} "
        f"train={args.episodes} eval={args.eval_episodes} "
        f"seeds=({train_seed}, {eval_seed}) -> {output_dir}"
    )

    train_rows: list[dict] = []
    train_started = time.perf_counter()
    agent.train(
        total_episodes=args.episodes,
        seed=train_seed,
        episode_callback=train_rows.append,
    )
    train_duration = time.perf_counter() - train_started
    training_csv = output_dir / "training_episodes.csv"
    _write_episode_csv(training_csv, train_rows)

    eval_started = time.perf_counter()
    eval_rows = evaluate(agent, eval_seed, args.eval_episodes)
    eval_duration = time.perf_counter() - eval_started
    eval_csv = output_dir / "evaluation_episodes.csv"
    _write_episode_csv(eval_csv, eval_rows)

    checkpoint = output_dir / f"final_{args.agent}{suffix}"
    agent.save(checkpoint)

    summary = {
        "config": {
            "env": ENV_ID,
            "agent": args.agent,
            "episodes": args.episodes,
            "eval_episodes": args.eval_episodes,
            "train_seed": train_seed,
            "eval_seed": eval_seed,
            "fresh_run": True,
            "checkpoint": checkpoint.name,
            "checkpoint_selection": "final",
        },
        "hyperparameters_initial": hparams,
        "versions": {
            "python": platform.python_version(),
            "gymnasium": importlib.metadata.version("gymnasium"),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "durations_seconds": {"training": train_duration, "evaluation": eval_duration},
        "statistics": {
            "training": _statistics(train_rows),
            "evaluation": _statistics(eval_rows),
        },
    }
    _write_summary(output_dir / "summary.json", summary)

    success = summary["statistics"]["evaluation"]["success_count"]
    print(
        f"[run_experiment] done: eval success {success}/{args.eval_episodes}, "
        f"checkpoint={checkpoint.name}, summary written last"
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_experiment",
        description=(
            "Run one fresh, seeded qlearning/dqn training plus greedy evaluation and "
            "write per-episode CSVs, statistics and the FINAL checkpoint into a "
            "brand-new output directory."
        ),
    )
    parser.add_argument("agent", choices=tuple(AGENTS), help="agent to train and evaluate")
    parser.add_argument(
        "--episodes",
        type=_positive_int,
        required=True,
        help="training episodes (positive, required)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_TRAIN_SEED,
        help=f"training seed (default: {DEFAULT_TRAIN_SEED})",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help=(
            "first evaluation seed (default: train seed + episodes, the first seed "
            "outside the training range)"
        ),
    )
    parser.add_argument(
        "--eval-episodes",
        type=_positive_int,
        default=DEFAULT_EVAL_EPISODES,
        help=f"greedy evaluation episodes (default: {DEFAULT_EVAL_EPISODES})",
    )
    parser.add_argument("--output-dir", required=True, help="brand-new directory for the run output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception:  # noqa: BLE001 - CLI boundary: report any failure, keep partial run
        # Preserve the partial run: the output directory is always fresh, so the
        # next attempt must use a new --output-dir (no resumption flag).
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
