"""Focused behavioral tests for scripts/run_experiment.py.

Stdlib ``unittest`` only (plus the project's own numpy/gymnasium dependencies,
already used by the agent suite). Agents and environments are tiny fakes except
for two real end-to-end smoke runs with 2 + 2 episodes; no expensive campaigns
are trained. The tests exercise observable behavior and never read the runner's
source text.
"""
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest import mock

import numpy as np

_RUNNER_PATH = Path(__file__).resolve().parent.parent / "scripts" / "run_experiment.py"
_spec = importlib.util.spec_from_file_location("run_experiment", _RUNNER_PATH)
assert _spec is not None and _spec.loader is not None
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


class _FakeQLearning:
    """Records construction/train arguments and emits deterministic events."""

    instances: ClassVar[list["_FakeQLearning"]] = []

    def __init__(self, env_id: str) -> None:
        self.env_id = env_id
        # Values the runner reads as "initial hyperparameters".
        self.n_bins, self.lr = 4, 0.1
        self.gamma, self.epsilon_end, self.epsilon_decay = 0.99, 0.01, 0.9995
        self.epsilon = 1.0
        self.training_episodes = 0
        self.seed: int | None = None
        self.predict_deterministic: bool | None = None
        _FakeQLearning.instances.append(self)

    def train(self, total_episodes, log_interval=100, *, seed=None, episode_callback=None):
        self.train_args = (total_episodes, seed)
        for episode in range(1, total_episodes + 1):
            terminated = episode % 2 == 1
            event = {
                "episode": episode,
                "seed": None if seed is None else seed + episode - 1,
                "return": float(-110.0 + episode),
                "steps": 100 + episode,
                "terminated": terminated,
                "truncated": not terminated,
                "epsilon": 0.5,
                "time": 0.01,
            }
            if episode_callback is not None:
                episode_callback(event)
            self.training_episodes += 1
        return []

    def predict(self, obs, *, deterministic=True):
        self.predict_deterministic = deterministic
        return 2, None

    def save(self, path: Path) -> None:
        path.write_bytes(b"fake-ql")


class _FakeDQN(_FakeQLearning):
    """DQN-shaped fake: the constructor takes the training seed like DQNAgent."""

    def __init__(self, env_id: str, *, seed: int | None = None) -> None:
        super().__init__(env_id)
        self.seed = seed
        self.lr, self.gamma = 1e-3, 0.99
        self.epsilon_end, self.epsilon_decay = 0.01, 0.995
        self.batch_size, self.buffer_capacity = 64, 100_000
        self.target_update_freq, self.hidden = 10, 128
        self.repeat_prob = 0.9

    def save(self, path: Path) -> None:
        path.write_bytes(b"fake-dqn")


class _FakeEvalEnv:
    """One step per episode; the result comes from a per-episode behaviors list."""

    def __init__(self, behaviors) -> None:
        self.behaviors = list(behaviors)
        self.reset_seeds: list[int | None] = []
        self.episode_index = 0
        self.closed = False

    def reset(self, *, seed: int | None = None):
        self.reset_seeds.append(seed)
        return np.array([-0.5, 0.0], dtype=np.float32), {}

    def step(self, _action: int):
        self.episode_index += 1
        terminated, truncated = self.behaviors[self.episode_index - 1]
        return np.array([-0.5, 0.0], dtype=np.float32), -1.0, terminated, truncated, {}

    def close(self) -> None:
        self.closed = True


class _ExplodingPredict(_FakeQLearning):
    def predict(self, obs, *, deterministic=True):
        raise RuntimeError("policy evaluation exploded")


class _CsvResult:
    def __init__(self, fieldnames, rows) -> None:
        self.fieldnames = fieldnames
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)


class ExperimentRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._original_agent_items = list(runner.AGENTS.items())
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.output = Path(self._temp.name) / "exp"

    def tearDown(self) -> None:
        runner.AGENTS.clear()
        runner.AGENTS.update(dict(self._original_agent_items))

    def patch_agents(self, ql: type | None = None, dqn: type | None = None) -> None:
        """Swap the runner's agent registry entries for fakes (in place)."""
        patched = {}
        for name, (_, suffix, attrs) in self._original_agent_items:
            fake = (dqn or _FakeDQN) if name == "dqn" else (ql or _FakeQLearning)
            patched[name] = (fake, suffix, attrs)
        runner.AGENTS.clear()
        runner.AGENTS.update(patched)
        _FakeQLearning.instances.clear()

    def run_with_fake_env(self, argv, behaviors):
        env = _FakeEvalEnv(behaviors)
        with mock.patch.object(runner.gym, "make", return_value=env):
            code = runner.main(argv)
        return code, env

    # ── constructor seeding ────────────────────────────────────────────

    def test_qlearning_constructor_gets_env_id_without_seed(self) -> None:
        self.patch_agents()
        code, _ = self.run_with_fake_env(
            ["qlearning", "--episodes", "3", "--eval-episodes", "2",
             "--seed", "7", "--output-dir", str(self.output)],
            [(True, False), (True, False)],
        )
        self.assertEqual(code, 0)
        [agent] = _FakeQLearning.instances
        self.assertIsInstance(agent, _FakeQLearning)
        self.assertEqual(agent.env_id, runner.ENV_ID)
        self.assertEqual(agent.train_args, (3, 7))

    def test_dqn_constructor_and_train_receive_the_train_seed(self) -> None:
        self.patch_agents()
        code, _ = self.run_with_fake_env(
            ["dqn", "--episodes", "4", "--eval-episodes", "2",
             "--seed", "7", "--eval-seed", "50", "--output-dir", str(self.output)],
            [(True, False), (False, True)],
        )
        self.assertEqual(code, 0)
        [agent] = _FakeQLearning.instances
        self.assertIsInstance(agent, _FakeDQN)
        self.assertEqual(agent.seed, 7, "constructor seed must be the training seed")
        self.assertEqual(agent.train_args, (4, 7))

    # ── evaluation semantics ───────────────────────────────────────────

    def test_evaluation_is_greedy_seeded_and_uses_termination_not_reward(self) -> None:
        self.patch_agents()
        code, env = self.run_with_fake_env(
            ["qlearning", "--episodes", "3", "--eval-episodes", "4",
             "--seed", "10", "--eval-seed", "20", "--output-dir", str(self.output)],
            [(True, False), (False, True), (True, False), (False, True)],
        )
        self.assertEqual(code, 0)
        [agent] = _FakeQLearning.instances
        self.assertTrue(agent.predict_deterministic, "evaluation must use the greedy policy")

        rows = self._read_csv(self.output / "evaluation_episodes.csv")
        self.assertEqual(env.reset_seeds, [20, 21, 22, 23])
        self.assertEqual([r["seed"] for r in rows], [20, 21, 22, 23])
        # Flag reached is reported from `terminated`, never from reward == -200.
        self.assertEqual([r["terminated"] for r in rows], [1, 0, 1, 0])
        self.assertEqual([r["truncated"] for r in rows], [0, 1, 0, 1])
        # Evaluation reports an effective greedy policy: epsilon 0 everywhere,
        # regardless of the training exploration rate (0.5 in the fake events).
        self.assertEqual([r["epsilon"] for r in rows], [0, 0, 0, 0])

    def test_default_eval_seed_is_the_first_seed_outside_the_training_range(self) -> None:
        self.patch_agents()
        code, env = self.run_with_fake_env(
            ["qlearning", "--episodes", "5", "--eval-episodes", "3",
             "--seed", "10", "--output-dir", str(self.output)],
            [(True, False), (False, True), (True, False)],
        )
        self.assertEqual(code, 0)
        self.assertEqual(env.reset_seeds, [15, 16, 17])
        summary = self._summary()
        self.assertEqual(summary["config"]["eval_seed"], 15)
        training = self._read_csv(self.output / "training_episodes.csv")
        self.assertEqual([r["seed"] for r in training], [10, 11, 12, 13, 14])

    # ── seed/count validation before any side effect ───────────────────

    def test_seed_range_disjointness_at_the_boundary(self) -> None:
        self.patch_agents()
        # Boundary: evaluation starting exactly at train_seed + episodes is disjoint.
        code, _ = self.run_with_fake_env(
            ["qlearning", "--episodes", "5", "--eval-episodes", "2",
             "--seed", "10", "--eval-seed", "15", "--output-dir", str(self.output)],
            [(True, False), (False, True)],
        )
        self.assertEqual(code, 0)
        # Evaluation fully before training is also disjoint (fresh dir).
        other = Path(self._temp.name) / "exp-before"
        code, _ = self.run_with_fake_env(
            ["qlearning", "--episodes", "5", "--eval-episodes", "4",
             "--seed", "10", "--eval-seed", "0", "--output-dir", str(other)],
            [(True, False), (False, True), (True, False), (False, True)],
        )
        self.assertEqual(code, 0)

    def test_overlapping_seed_ranges_rejected_before_directory_creation(self) -> None:
        self.patch_agents()
        cases = [
            ["qlearning", "--episodes", "5", "--eval-episodes", "1",
             "--seed", "10", "--eval-seed", "14"],  # shares seed 14 with training
            ["qlearning", "--episodes", "5", "--eval-episodes", "5",
             "--seed", "10", "--eval-seed", "6"],  # eval 6..10 overlaps training at 10
            ["qlearning", "--episodes", "5", "--eval-episodes", "2",
             "--seed", "10", "--eval-seed", "10"],  # identical start
        ]
        for extra in cases:
            with self.subTest(extra=extra):
                argv = extra + ["--output-dir", str(self.output)]
                self.assertEqual(runner.main(argv), 2)
                self.assertFalse(self.output.exists())

    def test_seeds_outside_numpy_range_rejected_before_directory_creation(self) -> None:
        self.patch_agents()
        cases = [
            (5, ["--seed", "-1"]),
            (5, ["--seed", str(2**32)]),
            (5, ["--eval-seed", "-5"]),
            (5, ["--eval-seed", str(2**32 - 1), "--eval-episodes", "2"]),
            (5, ["--seed", str(2**32 - 3)]),  # training range ends beyond 2**32-1
            (10, ["--seed", str(2**32 - 10)]),  # derived eval seed 2**32 overflows
        ]
        for episodes, extra in cases:
            with self.subTest(extra=extra):
                argv = (["qlearning", "--episodes", str(episodes)] + extra
                        + ["--output-dir", str(self.output)])
                self.assertEqual(runner.main(argv), 2)
                self.assertFalse(self.output.exists())

    def test_nonpositive_counts_rejected_before_directory_creation(self) -> None:
        self.patch_agents()
        cases = [
            ["--episodes", "0"],
            ["--episodes", "-3"],
            ["--episodes", "not-a-number"],
            ["qlearning", "--episodes", "2", "--eval-episodes", "0"],
            ["qlearning", "--episodes", "2", "--eval-episodes", "-1"],
            ["qlearning", "--eval-episodes", "2"],  # missing required --episodes
        ]
        for argv in cases:
            with self.subTest(argv=argv):
                full = argv if argv[0] == "qlearning" else ["qlearning"] + argv
                full += ["--output-dir", str(self.output)]
                with self.assertRaises(SystemExit):
                    runner.main(full)
                self.assertFalse(self.output.exists())

    # ── fresh, isolated output directory ───────────────────────────────

    def test_existing_destination_never_touched(self) -> None:
        self.patch_agents()

        def attempt(destination: Path) -> int:
            return runner.main(["qlearning", "--episodes", "2", "--seed", "1",
                                "--eval-seed", "100", "--output-dir", str(destination)])

        with self.subTest(case="existing file"):
            leaf = Path(self._temp.name) / "file_leaf"
            leaf.write_bytes(b"precious\x00bytes")
            self.assertEqual(attempt(leaf), 2)
            self.assertEqual(leaf.read_bytes(), b"precious\x00bytes")

        with self.subTest(case="existing empty directory"):
            leaf = Path(self._temp.name) / "dir_leaf"
            leaf.mkdir()
            self.assertEqual(attempt(leaf), 2)
            self.assertEqual(sorted(p.name for p in leaf.iterdir()), [])

        with self.subTest(case="existing symlink"):
            target = Path(self._temp.name) / "target_leaf"
            target.mkdir()
            leaf = Path(self._temp.name) / "link_leaf"
            leaf.symlink_to(target, target_is_directory=True)
            self.assertEqual(attempt(leaf), 2)
            self.assertEqual(sorted(p.name for p in target.iterdir()), [])

    # ── artifacts: CSVs and summary ────────────────────────────────────

    def test_output_schema_nonempty_and_summary_statistics_agree(self) -> None:
        self.patch_agents()
        code, _ = self.run_with_fake_env(
            ["qlearning", "--episodes", "4", "--eval-episodes", "4",
             "--seed", "5", "--eval-seed", "20", "--output-dir", str(self.output)],
            [(True, False), (False, True), (True, False), (False, True)],
        )
        self.assertEqual(code, 0)

        training = self._read_csv(self.output / "training_episodes.csv")
        evaluation = self._read_csv(self.output / "evaluation_episodes.csv")
        self.assertEqual(training.fieldnames, list(runner.CSV_FIELDS))
        self.assertEqual(evaluation.fieldnames, list(runner.CSV_FIELDS))
        self.assertEqual(len(training), 4)
        self.assertEqual(len(evaluation), 4)
        self.assertEqual([r["seed"] for r in training], [5, 6, 7, 8])
        self.assertEqual([r["return"] for r in training], [-109.0, -108.0, -107.0, -106.0])
        self.assertEqual([r["steps"] for r in training], [101, 102, 103, 104])

        summary = self._summary()
        config = summary["config"]
        self.assertEqual(config["env"], "MountainCar-v0")
        self.assertEqual(config["agent"], "qlearning")
        self.assertEqual((config["episodes"], config["eval_episodes"]), (4, 4))
        self.assertEqual((config["train_seed"], config["eval_seed"]), (5, 20))
        self.assertTrue(config["fresh_run"])
        self.assertEqual(config["checkpoint"], "final_qlearning.pkl")
        self.assertEqual(config["checkpoint_selection"], "final")
        self.assertEqual(summary["hyperparameters_initial"]["epsilon_start"], 1.0)
        self.assertEqual(summary["hyperparameters_initial"]["n_bins"], 4)

        checkpoint = self.output / config["checkpoint"]
        self.assertTrue(checkpoint.exists())
        self.assertEqual(checkpoint.read_bytes(), b"fake-ql")

        # Statistics must be recomputable from the CSVs alone (population std).
        for phase, rows in (("training", training), ("evaluation", evaluation)):
            stats = summary["statistics"][phase]
            returns = [float(r["return"]) for r in rows]
            steps = [float(r["steps"]) for r in rows]
            self.assertEqual(stats["episodes"], len(rows))
            self.assertEqual(stats["return_mean"], float(np.mean(returns)))
            self.assertEqual(stats["return_std"], float(np.std(returns)))
            self.assertEqual(stats["return_min"], float(min(returns)))
            self.assertEqual(stats["return_max"], float(max(returns)))
            self.assertEqual(stats["steps_mean"], float(np.mean(steps)))
            self.assertEqual(stats["success_count"], sum(bool(int(r["terminated"])) for r in rows))

        # summary.json is written last.
        self.assertGreaterEqual(
            (self.output / "summary.json").stat().st_mtime_ns,
            (self.output / "evaluation_episodes.csv").stat().st_mtime_ns,
        )

    def test_summary_uses_native_json_numbers(self) -> None:
        self.patch_agents()
        code, _ = self.run_with_fake_env(
            ["dqn", "--episodes", "3", "--eval-episodes", "2",
             "--seed", "9", "--eval-seed", "40", "--output-dir", str(self.output)],
            [(True, False), (True, False)],
        )
        self.assertEqual(code, 0)
        summary = self._summary()
        numeric_paths = [
            ("config", "episodes"),
            ("config", "eval_episodes"),
            ("config", "train_seed"),
            ("config", "eval_seed"),
            ("hyperparameters_initial", "epsilon_start"),
            ("hyperparameters_initial", "batch_size"),
            ("durations_seconds", "training"),
            ("durations_seconds", "evaluation"),
            ("statistics", "training", "return_mean"),
            ("statistics", "training", "return_std"),
            ("statistics", "training", "return_min"),
            ("statistics", "training", "return_max"),
            ("statistics", "training", "success_count"),
            ("statistics", "training", "steps_mean"),
            ("statistics", "evaluation", "return_mean"),
            ("statistics", "evaluation", "success_count"),
        ]
        for path in numeric_paths:
            value = summary
            for part in path:
                value = value[part]
            self.assertNotIsInstance(value, str, f"{path} must be a native number")
            self.assertIsInstance(value, (int, float))
        self.assertIsInstance(summary["config"]["fresh_run"], bool)
        self.assertIsInstance(summary["durations_seconds"]["training"], float)

    # ── failure behavior ───────────────────────────────────────────────

    def test_eval_error_closes_env_and_preserves_partial_run(self) -> None:
        self.patch_agents(ql=_ExplodingPredict)
        env = _FakeEvalEnv([(True, False)])
        with mock.patch.object(runner.gym, "make", return_value=env):
            code = runner.main(["qlearning", "--episodes", "3", "--eval-episodes", "1",
                                "--seed", "5", "--eval-seed", "40",
                                "--output-dir", str(self.output)])
        self.assertEqual(code, 1)
        self.assertTrue(env.closed, "evaluation environment must be closed on error")
        self.assertTrue((self.output / "training_episodes.csv").exists())
        self.assertFalse((self.output / "evaluation_episodes.csv").exists())
        self.assertFalse((self.output / "summary.json").exists())
        self.assertFalse(list(self.output.glob("final_*")))

    # ── real tiny end-to-end smoke runs ────────────────────────────────

    def test_real_qlearning_tiny_run_end_to_end(self) -> None:
        code = runner.main(["qlearning", "--episodes", "2", "--eval-episodes", "2",
                            "--seed", "3", "--eval-seed", "1000",
                            "--output-dir", str(self.output)])
        self.assertEqual(code, 0)
        self.assertEqual(len(self._read_csv(self.output / "training_episodes.csv")), 2)
        checkpoint = self.output / "final_qlearning.pkl"
        self.assertTrue(checkpoint.exists())
        self.assertGreater(checkpoint.stat().st_size, 0)
        self.assertEqual(self._summary()["config"]["checkpoint"], "final_qlearning.pkl")

    def test_real_dqn_tiny_run_end_to_end(self) -> None:
        code = runner.main(["dqn", "--episodes", "2", "--eval-episodes", "2",
                            "--seed", "5", "--eval-seed", "1000",
                            "--output-dir", str(self.output)])
        self.assertEqual(code, 0)
        self.assertEqual(len(self._read_csv(self.output / "training_episodes.csv")), 2)
        checkpoint = self.output / "final_dqn.pt"
        self.assertTrue(checkpoint.exists())
        self.assertGreater(checkpoint.stat().st_size, 0)
        self.assertEqual(self._summary()["config"]["checkpoint"], "final_dqn.pt")

    # ── helpers ────────────────────────────────────────────────────────

    def _read_csv(self, path: Path):
        with open(path, encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            rows = [dict(row) for row in reader]
        parsed = [
            {
                key: float(value) if key in ("return", "epsilon", "time")
                else int(value)
                for key, value in row.items()
            }
            for row in rows
        ]
        return _CsvResult(fieldnames, parsed)

    def _summary(self) -> dict:
        return json.loads((self.output / "summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
