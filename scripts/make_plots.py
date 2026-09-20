"""Genera las gráficas de evidencia (PNG) en results/ a partir de los logs y de evaluacion_final.json."""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

RESULTS = Path("results")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8983"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "legend.frameon": False,
    }
)


def read_train_log(path: Path) -> tuple[list[int], list[float]]:
    pts = []
    for line in path.read_text().splitlines():
        m = re.match(r"Episode (\d+)/\d+ \| Avg Reward: (-?[\d.]+)", line)
        if m:
            pts.append((int(m[1]), float(m[2])))
    return [p[0] for p in pts], [p[1] for p in pts]


def read_progress_log(path: Path) -> tuple[list[int], list[float], list[float]]:
    rows = []
    pattern = (
        r"ep\s+(\d+) \| entrenamiento \(prom\. 100 ep\):\s+(-?[\d.]+)"
        r" \| evaluacion congelada:\s+(-?[\d.]+) \| llegan\s+(\d+)/30"
    )
    for line in path.read_text().splitlines():
        m = re.match(pattern, line)
        if m:
            rows.append((int(m[1]), float(m[2]), float(m[3])))
    return [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]


def reference_lines(ax, note110_left: bool = False) -> None:
    ax.axhline(-200, color=GRAY, lw=1, ls=(0, (4, 3)))
    ax.axhline(-110, color=GRAY, lw=1, ls=(0, (4, 3)))
    tf = ax.get_yaxis_transform()
    ax.text(0.99, -200 + 1.5, "-200: nunca llega a la bandera", transform=tf, ha="right", va="bottom", color=INK2, fontsize=8.5)
    x110, ha110 = (0.01, "left") if note110_left else (0.99, "right")
    ax.text(x110, -110 + 1.5, "-110: umbral de «resuelto»", transform=tf, ha=ha110, va="bottom", color=INK2, fontsize=8.5)


def save(fig, name: str) -> None:
    fig.savefig(RESULTS / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("guardado", RESULTS / name)


final = json.loads((RESULTS / "evaluacion_final.json").read_text())
ql, dqn = final["qlearning"], final["dqn"]
ql_x, ql_y = read_train_log(RESULTS / "qlearning_train.log")
base_x, base_y = read_train_log(RESULTS / "dqn_baseline_1000.log")
dqn_x, dqn_train, dqn_eval = read_progress_log(RESULTS / "dqn_progress.log")


def summary(m: dict) -> str:
    return f"{m['mean_reward']:.1f} ± {m['std_reward']:.1f} · {m['reached_flag']}/{m['eval_episodes']} llegan a la bandera"


# 1) Q-Learning ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(ql_x, ql_y, color=BLUE, lw=2)
ax.set_ylim(-212, -95)
ax.set_xlim(0, max(ql_x))
reference_lines(ax)
ax.set_title("Q-Learning tabular en MountainCar-v0")
ax.set_xlabel("Episodios de entrenamiento")
ax.set_ylabel("Recompensa promedio (ventana de 100 episodios)")
ax.text(
    0.02, 0.83,
    f"Evaluación final ({ql['eval_episodes']} episodios, sin exploración):\n{summary(ql)}",
    transform=ax.transAxes, va="top", color=INK, fontsize=9.5,
)
save(fig, "fig_qlearning.png")

# 2) DQN ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(base_x, base_y, color=GRAY, lw=2, label="DQN estándar (ε-greedy)")
ax.plot(dqn_x, dqn_train, color=ORANGE, lw=2, label="DQN persistente: entrenamiento")
ax.plot(dqn_x, dqn_eval, color=ORANGE, lw=1.5, ls=(0, (4, 2)), marker="o", ms=4, mfc=SURFACE, mew=1.5,
        label="DQN persistente: evaluación (30 ep.)")
best_i = max(range(len(dqn_eval)), key=lambda i: dqn_eval[i])
ax.plot(dqn_x[best_i], dqn_eval[best_i], marker="*", ms=14, color=ORANGE, mec=SURFACE, mew=1.5, ls="none", zorder=5)
ax.annotate(
    f"Mejor modelo (episodio {dqn_x[best_i]})\n{summary(dqn)}",
    xy=(dqn_x[best_i], dqn_eval[best_i] + 3), xytext=(dqn_x[best_i], -70),
    ha="center", va="center", color=INK, fontsize=9,
    arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8},
)
ax.set_ylim(-212, -60)
ax.set_xlim(0, 2500)
reference_lines(ax, note110_left=True)
ax.set_title("DQN en MountainCar-v0: el efecto de la exploración persistente")
ax.set_xlabel("Episodios de entrenamiento")
ax.set_ylabel("Recompensa promedio")
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.1), fontsize=8.5)
save(fig, "fig_dqn.png")

# 3) Comparación --------------------------------------------------------------
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True, gridspec_kw={"width_ratios": [2.2, 1]})
ax.plot(ql_x, ql_y, color=BLUE, lw=2, label="Q-Learning tabular")
ax.plot(dqn_x, dqn_train, color=ORANGE, lw=2, label="DQN")
ax.set_xscale("log")
ax.set_xlim(90, 60000)
ax.set_xticks([100, 300, 1000, 3000, 10000])
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", ".")))
ax.minorticks_off()
ax.set_ylim(-212, -80)
reference_lines(ax)
ax.text(ql_x[-1] * 1.08, ql_y[-1], "Q-Learning", color=INK2, va="center", fontsize=9)
ax.text(dqn_x[-1] * 1.08, dqn_train[-1], "DQN", color=INK2, va="center", fontsize=9)
ax.set_title("Velocidad de aprendizaje")
ax.set_xlabel("Episodios de entrenamiento (escala logarítmica)")
ax.set_ylabel("Recompensa promedio (ventana de 100 episodios)")
ax.legend(loc="upper left", fontsize=9)

for i, (m, color, name) in enumerate([(ql, BLUE, "Q-Learning"), (dqn, ORANGE, "DQN")]):
    ax2.errorbar(i, m["mean_reward"], yerr=m["std_reward"], fmt="o", ms=8, color=color, ecolor=color, elinewidth=2, capsize=5)
    ax2.text(i + 0.12, m["mean_reward"], f"{m['mean_reward']:.1f}\n{m['reached_flag']}/{m['eval_episodes']} llegan",
             ha="left", va="center", color=INK, fontsize=9)
ax2.axhline(-110, color=GRAY, lw=1, ls=(0, (4, 3)))
ax2.set_xlim(-0.6, 2.1)
ax2.set_xticks([0, 1], ["Q-Learning", "DQN"])
ax2.grid(False)
ax2.grid(True, axis="y")
ax2.set_title("Evaluación final")
ax2.set_xlabel("Media ± desv. (100 episodios)")
fig.suptitle("Q-Learning tabular vs. DQN en MountainCar-v0", x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.tight_layout()
save(fig, "fig_comparacion.png")
