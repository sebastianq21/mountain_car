import gymnasium as gym

env = gym.make("MountainCar-v0")
llegadas = 0

for episodio in range(300):
    env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        accion = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(accion)
    if terminated:
        llegadas += 1

print(f"Episodios que llegaron a la bandera: {llegadas} de 300")