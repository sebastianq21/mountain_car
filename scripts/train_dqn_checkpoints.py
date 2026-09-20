"""Entrena el DQN 2500 episodios evaluando la política congelada cada 100 y guardando el mejor modelo.

Escribe el mejor modelo en saves/dqn_best.pt. Uso (el log de este experimento es results/dqn_progress.log):

    uv run python -u scripts/train_dqn_checkpoints.py | tee results/dqn_progress.log
"""
from pathlib import Path

import gymnasium as gym
import numpy as np

from mountain_car.agents.dqn import DQNAgent
from mountain_car.cli import _run_episode

agent = DQNAgent("MountainCar-v0")
env_eval = gym.make("MountainCar-v0")
best = -1e9

for block in range(1, 26):
    rewards = agent.train(total_episodes=100, log_interval=100)
    res = [_run_episode(env_eval, agent) for _ in range(30)]
    ev = float(np.mean([r[0] for r in res]))
    ok = sum(r[2] for r in res)
    print(
        f"ep {block * 100:>4} | entrenamiento (prom. 100 ep):  {np.mean(rewards):7.1f}"
        f" | evaluacion congelada:  {ev:7.1f} | llegan {ok:>2}/30",
        flush=True,
    )
    if ev > best:
        best = ev
        agent.save(Path("saves/dqn_best.pt"))
        print(f"   -> nuevo mejor modelo (eval {ev:.1f}) guardado", flush=True)
