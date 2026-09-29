"""Regression tests for the seeded train APIs and core RL semantics.

The suite uses only the standard library's ``unittest`` plus the project's
own dependencies (numpy, torch). No long training is performed: environments
are replaced by tiny deterministic fakes and deterministic tensor runs are
forced onto the CPU.
"""
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from mountain_car.agents.dqn import DQNAgent, QNetwork
from mountain_car.agents.qlearning import QLearningAgent


class _OneStepEnv:
    """Terminates immediately after one step and records that ``close`` ran."""

    def __init__(self) -> None:
        self.closed = False

    def reset(self, *, seed: int | None = None):
        return np.array([-0.5, 0.0], dtype=np.float32), {}

    def step(self, _action: int):
        return np.array([-0.5, 0.0], dtype=np.float32), -1.0, True, False, {}

    def close(self) -> None:
        self.closed = True


class _FlaggedExitEnv:
    """Shared tiny env: episode 1 ends by termination, later ones by truncation.

    Every episode lasts exactly one step, so both the ``steps`` callback field
    and ``return`` are known up front (-1.0) no matter what actions the agent
    samples.
    """

    def __init__(self) -> None:
        self.reset_seeds: list[int | None] = []
        self.episode_index = 0

    def reset(self, *, seed: int | None = None):
        self.reset_seeds.append(seed)
        self.episode_index += 1
        return np.array([-0.5, 0.0], dtype=np.float32), {}

    def step(self, _action: int):
        observation = np.array([-0.5, 0.0], dtype=np.float32)
        if self.episode_index == 1:
            return observation, -1.0, True, False, {}
        return observation, -1.0, False, True, {}

    def close(self) -> None:
        pass


def _without_time(events: list[dict]) -> list[dict]:
    return [{key: value for key, value in event.items() if key != "time"} for event in events]


class QLearningAgentTest(unittest.TestCase):
    def make_agent(self, **kwargs) -> QLearningAgent:
        return QLearningAgent("MountainCar-v0", n_bins=4, **kwargs)

    # ── numeric TD updates ───────────────────────────────────────────

    def test_terminal_update_uses_reward_without_bootstrapping(self) -> None:
        agent = self.make_agent(lr=0.25, gamma=0.9)
        state, next_state = (0, 0), (1, 1)
        agent.q_table[state] = np.array([2.0, 0.0, 0.0])
        agent.q_table[next_state] = np.array([100.0, 0.0, 0.0])

        agent._update(state, action=0, reward=3.0, next_state=next_state, terminated=True)

        self.assertAlmostEqual(agent.q_table[state][0], 2.25)

    def test_nonterminal_update_bootstraps_from_the_best_next_action(self) -> None:
        agent = self.make_agent(lr=0.5, gamma=0.9)
        state, next_state = (0, 0), (1, 1)
        agent.q_table[state] = np.array([0.0, 0.0, 1.0])
        agent.q_table[next_state] = np.array([3.0, 5.0, 4.0])

        agent._update(state, action=2, reward=-1.0, next_state=next_state, terminated=False)

        self.assertAlmostEqual(agent.q_table[state][2], 2.25)

    # ── greedy evaluation ────────────────────────────────────────────

    def test_deterministic_selection_exploits_without_sampling_exploration(self) -> None:
        agent = self.make_agent(epsilon_start=1.0)
        state = (1, 2)
        agent.q_table[state] = np.array([0.0, 4.0, 1.0])

        with patch("mountain_car.agents.qlearning.np.random.random") as random_value:
            action = agent.select_action(state, deterministic=True)

        self.assertEqual(action, 1)
        random_value.assert_not_called()

    # ── seeded training and callbacks ────────────────────────────────

    def test_seeded_training_is_reproducible_excluding_elapsed_time(self) -> None:
        def run_once():
            agent = self.make_agent()
            events: list[dict] = []
            env = _FlaggedExitEnv()
            with patch("mountain_car.agents.qlearning.gym.make", return_value=env):
                returns = agent.train( 3, log_interval=10, seed=123, episode_callback=events.append)
            return returns, _without_time(events), env.reset_seeds

        first, second = run_once(), run_once()

        self.assertEqual(first, second)
        self.assertEqual(first[2], [123, 124, 125])
        self.assertEqual([event["epsilon"] for event in first[1]], [1.0, 0.9995, 0.9995**2])
        self.assertEqual(
            set(first[1][0]),
            {"episode", "seed", "return", "steps", "terminated", "truncated", "epsilon"},
        )
        for index, (event, ret) in enumerate(zip(first[1], first[0])):
            self.assertEqual(event["episode"], index + 1)
            self.assertEqual(event["return"], ret)
            self.assertEqual(event["steps"], 1)
        self.assertEqual([(event["terminated"], event["truncated"]) for event in first[1]],
                         [(True, False), (False, True), (False, True)])

    def test_callback_exception_still_closes_the_environment(self) -> None:
        agent = self.make_agent()

        def boom(_event: dict) -> None:
            raise RuntimeError("callback failed")

        env = _OneStepEnv()
        with (
            patch("mountain_car.agents.qlearning.gym.make", return_value=env),
            self.assertRaises(RuntimeError),
        ):
            agent.train(1, log_interval=10, episode_callback=boom)

        self.assertTrue(env.closed)

    # ── legacy call signature ────────────────────────────────────────

    def test_legacy_positional_train_call_still_works(self) -> None:
        agent = self.make_agent()
        env = _OneStepEnv()
        with patch("mountain_car.agents.qlearning.gym.make", return_value=env):
            returns = agent.train(3, log_interval=10)

        self.assertEqual(len(returns), 3)
        self.assertTrue(all(ret == -1.0 for ret in returns))
        self.assertTrue(env.closed)

    def test_train_args_after_log_interval_are_keyword_only(self) -> None:
        agent = self.make_agent()

        with self.assertRaises(TypeError):
            agent.train(3, 10, 42)


class _LinearQNetwork(QNetwork):
    """Small controllable network for numeric Bellman-target tests."""

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.linear = nn.Linear(2, 3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class DQNAgentTest(unittest.TestCase):
    def make_agent(self, **kwargs) -> DQNAgent:
        with patch("mountain_car.agents.dqn.torch.cuda.is_available", return_value=False):
            return DQNAgent("MountainCar-v0", **kwargs)

    # ── DQN learning step ────────────────────────────────────────────

    def test_bellman_target_masks_terminal_bootstrap_and_keeps_truncation_bootstrap(self) -> None:
        agent = self.make_agent(lr=0.1, gamma=0.5, batch_size=2, hidden=4)
        online = _LinearQNetwork().to(agent.device)
        target = _LinearQNetwork().to(agent.device)
        agent.q_net = online
        agent.target_net = target
        with torch.no_grad():
            online.linear.weight.zero_()
            target.linear.weight.copy_(
                torch.tensor([[2.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], device=agent.device)
            )

        state = np.array([1.0, 0.0], dtype=np.float32)
        agent.buffer.push(state, 1, 1.0, state, True)   # terminal: target is 1.0
        agent.buffer.push(state, 2, 1.0, state, False)  # nonterminal: target is 1.0 + 0.5 * 2

        loss = agent._learn()

        self.assertAlmostEqual(loss, 2.5)
        self.assertTrue(any(p.grad is not None for p in agent.q_net.parameters()))
        self.assertTrue(all(p.grad is None for p in agent.target_net.parameters()))

    # ── greedy evaluation and repeat_prob exploration ────────────────

    def test_deterministic_selection_is_greedy_despite_active_exploration(self) -> None:
        agent = self.make_agent(epsilon_start=1.0)
        state = np.array([-0.5, 0.0], dtype=np.float32)
        with torch.no_grad():
            expected = int(
                agent.q_net(torch.as_tensor(state, device=agent.device).unsqueeze(0))
                .argmax(dim=1)
                .item()
            )
        agent._last_action = 2

        with (
            patch("mountain_car.agents.dqn.random.random") as random_value,
            patch("mountain_car.agents.dqn.random.randrange") as random_action,
        ):
            action = agent.select_action(state, deterministic=True)

        self.assertEqual(action, expected)
        random_value.assert_not_called()
        random_action.assert_not_called()

    def test_repeat_prob_repeats_last_action_only_when_the_roll_succeeds(self) -> None:
        agent = self.make_agent(epsilon_start=1.0, repeat_prob=0.9)
        state = np.array([-0.5, 0.0], dtype=np.float32)

        with (
            patch("mountain_car.agents.dqn.random.random", side_effect=[0.0, 0.0, 0.0, 0.0, 0.95]),
            patch("mountain_car.agents.dqn.random.randrange", side_effect=[3, 1]) as randrange,
        ):
            actions = [agent.select_action(state) for _ in range(3)]

        self.assertEqual(actions, [3, 3, 1])
        self.assertEqual(randrange.call_count, 2)

    # ── constructor seed and weight initialization ───────────────────

    def test_same_constructor_seed_yields_identical_fresh_weights(self) -> None:
        first = self.make_agent(seed=7)
        second = self.make_agent(seed=7)
        different = self.make_agent(seed=99)

        for name in first.q_net.state_dict():
            self.assertTrue(
                torch.equal(first.q_net.state_dict()[name], second.q_net.state_dict()[name])
            )

        identical_everywhere = all(
            torch.equal(first.q_net.state_dict()[name], different.q_net.state_dict()[name])
            for name in first.q_net.state_dict()
        )
        self.assertFalse(identical_everywhere)

    def test_train_seed_does_not_reinitialize_existing_weights(self) -> None:
        agent = self.make_agent(seed=5)
        before = {name: tensor.clone() for name, tensor in agent.q_net.state_dict().items()}
        env = _OneStepEnv()

        with patch("mountain_car.agents.dqn.gym.make", return_value=env):
            agent.train(1, log_interval=10, seed=9)

        for name, tensor in before.items():
            self.assertTrue(torch.equal(tensor, agent.q_net.state_dict()[name]))

    # ── seeded training and callbacks ────────────────────────────────

    def test_seeded_training_is_reproducible_excluding_elapsed_time(self) -> None:
        def run_once():
            agent = self.make_agent(
                epsilon_start=1.0, epsilon_end=0.0, epsilon_decay=0.5, seed=321
            )
            events: list[dict] = []
            env = _FlaggedExitEnv()
            with patch("mountain_car.agents.dqn.gym.make", return_value=env):
                returns = agent.train(3, log_interval=10, seed=123, episode_callback=events.append)
            return returns, _without_time(events), env.reset_seeds

        first, second = run_once(), run_once()

        self.assertEqual(first, second)
        self.assertEqual(first[2], [123, 124, 125])
        self.assertEqual([event["epsilon"] for event in first[1]], [1.0, 0.5, 0.25])
        self.assertEqual(
            set(first[1][0]),
            {"episode", "seed", "return", "steps", "terminated", "truncated", "epsilon"},
        )
        for index, (event, ret) in enumerate(zip(first[1], first[0], strict=True)):
            self.assertEqual(event["episode"], index + 1)
            self.assertEqual(event["return"], ret)
            self.assertEqual(event["steps"], 1)
        self.assertEqual([(event["terminated"], event["truncated"]) for event in first[1]],
                         [(True, False), (False, True), (False, True)])

    def test_callback_exception_still_closes_the_environment(self) -> None:
        agent = self.make_agent(seed=1)

        def boom(_event: dict) -> None:
            raise RuntimeError("callback failed")

        env = _OneStepEnv()
        with (
            patch("mountain_car.agents.dqn.gym.make", return_value=env),
            self.assertRaises(RuntimeError),
        ):
            agent.train(1, log_interval=10, episode_callback=boom)

        self.assertTrue(env.closed)

    # ── legacy call signatures ───────────────────────────────────────

    def test_legacy_positional_train_call_still_works(self) -> None:
        agent = self.make_agent(seed=1)
        env = _OneStepEnv()
        with patch("mountain_car.agents.dqn.gym.make", return_value=env):
            returns = agent.train(2, log_interval=10)

        self.assertEqual(len(returns), 2)
        self.assertTrue(all(ret == -1.0 for ret in returns))
        self.assertTrue(env.closed)

    def test_constructor_and_train_seed_are_keyword_only(self) -> None:
        with self.assertRaises(TypeError):
            DQNAgent("MountainCar-v0", 42)

        agent = self.make_agent(seed=1)
        with self.assertRaises(TypeError):
            agent.train(2, 10, 42)


if __name__ == "__main__":
    unittest.main()
