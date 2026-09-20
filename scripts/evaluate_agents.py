"""Evalúa (política codiciosa, 100 episodios) los agentes guardados en saves/ y escribe results/evaluacion_final.json."""
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from mountain_car.cli import AGENTS, ENV_ID, _run_episode

N_EPISODES = 100


def evaluate(name: str) -> dict:
    cls, path = AGENTS[name]
    agent = cls.load(path)
    env = gym.make(ENV_ID)
    results = [_run_episode(env, agent) for _ in range(N_EPISODES)]
    env.close()
    rewards = [reward for reward, _, _ in results]
    return {
        "training_episodes": agent.training_episodes,
        "eval_episodes": N_EPISODES,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "reached_flag": int(sum(reached for _, _, reached in results)),
    }


if __name__ == "__main__":
    out = {name: evaluate(name) for name in AGENTS}
    Path("results").mkdir(exist_ok=True)
    Path("results/evaluacion_final.json").write_text(json.dumps(out, indent=2))
    for name, m in out.items():
        print(
            f"{name:<10} entrenado {m['training_episodes']:>6} ep. | "
            f"recompensa {m['mean_reward']:.1f} +/- {m['std_reward']:.1f} | "
            f"llega a la bandera {m['reached_flag']}/{m['eval_episodes']}"
        )
