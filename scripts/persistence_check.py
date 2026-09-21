"""Mide cuántos de 300 episodios llegan a la bandera con exploración puramente aleatoria pero con persistencia p.

p = probabilidad de repetir la acción anterior; con p = 0 es el dado independiente de la ε-greedy estándar.
Uso: uv run python scripts/persistence_check.py | tee results/persistence_check.log
"""
import random

import gymnasium as gym

random.seed(0)
env = gym.make("MountainCar-v0")

for p in [0.0, 0.5, 0.8, 0.9, 0.95]:
    llegadas = 0
    for episodio in range(300):
        env.reset(seed=episodio)
        ultima = None
        terminated = truncated = False
        while not (terminated or truncated):
            if ultima is not None and random.random() < p:
                accion = ultima
            else:
                accion = random.randrange(3)
            ultima = accion
            _, _, terminated, truncated, _ = env.step(accion)
        llegadas += terminated
    print(f"p = {p:<4} -> {llegadas:>3} de 300 llegan a la bandera")
