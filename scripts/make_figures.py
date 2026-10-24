"""Figures for the report."""
import glob
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rlrec.config import RUNS, REPORTS

plt.rcParams.update({"figure.dpi": 140, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})


def load(name):
    p = RUNS / name
    return json.loads(p.read_text()) if p.exists() else None


# ---------------------------------------------------------------- pareto
fin = load("final_test.json")
if fin:
    R, n = fin["results"], fin["n_users"]
    def rpu(v):
        return v.get("reward_per_user", v["hits"] * v["realised_reward"] / n)

    kl_runs = {}
    for k, v in R.items():
        if k.startswith("RL, kl") and "seed" not in k and "return" not in k:
            kl = float(k.replace("RL, kl", ""))
            kl_runs[kl] = v
    base = R["базовая GPT-2"]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    xs = [base["ndcg@10"]] + [kl_runs[k]["ndcg@10"] for k in sorted(kl_runs)]
    ys = [rpu(base)] + [rpu(kl_runs[k]) for k in sorted(kl_runs)]
    extras = [("best-of-50 по reward", "#d62728", "s", "best-of-50"),
              ("популярность", "#7f7f7f", "^", "популярность"),
              ("самые досматриваемые", "#8c564b", "v", "самые досматриваемые"),
              ("RL, return_kl0.05", "#9467bd", "D", "RL на возвратах")]

    for ax, zoom in zip(axes, (False, True)):
        ax.plot(xs[1:], ys[1:], "-o", color="#1f77b4", ms=5, lw=1.4, zorder=3,
                label="RL при разных KL")
        ax.scatter([xs[0]], [ys[0]], c="#2ca02c", s=90, marker="*", zorder=5,
                   label="базовая модель")
        ax.axhline(ys[0], color="#2ca02c", ls=":", lw=1)
        if not zoom:
            for name, col, mk, lab in extras:
                if name in R:
                    ax.scatter(R[name]["ndcg@10"], rpu(R[name]), c=col, marker=mk,
                               s=55, zorder=4)
                    ax.annotate(lab, (R[name]["ndcg@10"], rpu(R[name])), fontsize=7.5,
                                textcoords="offset points", xytext=(7, 2))
            ax.set_title("все подходы", fontsize=9.5)
            ax.legend(fontsize=7.5, loc="lower right")
        else:
            for k, x, y in zip(sorted(kl_runs), xs[1:], ys[1:]):
                dy = 7 if k >= 1 else -12
                ax.annotate(f"KL={k:g}", (x, y), fontsize=8,
                            textcoords="offset points", xytext=(0, dy), ha="center")
            ax.annotate("базовая", (xs[0], ys[0]), fontsize=8, fontweight="bold",
                        textcoords="offset points", xytext=(-6, 8), ha="right")
            lo = min(ys[1:] + [ys[0]]); hi = max(ys[1:] + [ys[0]])
            ax.set_ylim(lo - (hi - lo) * 0.18, hi + (hi - lo) * 0.30)
            ax.set_xlim(min(xs) - 0.004, max(xs) + 0.006)
            ax.set_title("рабочая область: выше пунктира RL выигрывает", fontsize=9.5)
        ax.set_xlabel("NDCG@10 на тесте — релевантность")
    axes[0].set_ylabel("доставленное время просмотра на пользователя")
    fig.suptitle("Компромисс между релевантностью и вознаграждением", fontsize=11)
    fig.tight_layout(); fig.savefig(REPORTS / "fig_pareto.png"); plt.close(fig)

# -------------------------------------------------- the hacking signature
d = load("rl_kl0.json")
if d:
    ck = d["checkpoints"]; n = d["args"]["eval_users"]
    steps = [c["step"] for c in ck]
    pred = [c["predicted_reward"] for c in ck]
    real = [c["realised_reward"] for c in ck]
    rp = [c["hits"] * c["realised_reward"] / n for c in ck]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.4, 5.6), sharex=True,
                                   gridspec_kw={"height_ratios": [1.3, 1]})
    ax1.plot(steps, pred, "-o", ms=4, color="#d62728", label="обещает reward-модель")
    ax1.plot(steps, real, "-o", ms=4, color="#2ca02c", label="наблюдается на отложенной неделе")
    ax1.fill_between(steps, real, pred, color="#d62728", alpha=0.12, lw=0)
    for s, p_, r_ in [(steps[-1], pred[-1], real[-1])]:
        ax1.annotate(f"разрыв {p_-r_:+.3f}", (s, (p_ + r_) / 2), fontsize=8,
                     ha="right", color="#d62728")
    ax1.set_ylabel("доля досмотра")
    ax1.set_title("Вознаграждение растёт, доставленная ценность падает\n"
                  "обучение без штрафа KL", fontsize=10)
    ax1.legend(fontsize=8)
    ax2.plot(steps, rp, "-o", ms=4, color="#1f77b4")
    ax2.axhline(rp[0], color="k", ls=":", lw=1)
    ax2.annotate("уровень исходной модели", (steps[-1], rp[0]), fontsize=7.5,
                 ha="right", va="bottom")
    ax2.set_xlabel("шаг GRPO")
    ax2.set_ylabel("доставлено на пользователя")
    fig.tight_layout(); fig.savefig(REPORTS / "fig_hacking.png"); plt.close(fig)

# ------------------------------------------------------------- rollout
ro = load("rollout.json")
if ro:
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax2 = ax.twinx(); ax2.grid(False)
    cmap = plt.get_cmap("tab10")
    show = [k for k in ro["policies"] if k in ("базовая", "RL kl0.01", "RL early60")]
    for i, name in enumerate(show):
        rows = ro["policies"][name]
        st = [r["step"] for r in rows]
        ax.plot(st, [r["predicted_reward"] for r in rows], "-o", ms=4,
                color=cmap(i), label=f"{name}: обещанная награда")
        ax2.plot(st, [r["verifiable_share"] for r in rows], "--s", ms=4,
                 color=cmap(i), alpha=0.75)
    ax.set_xlabel("шаг генерации вперёд")
    ax.set_ylabel("обещанная награда (сплошная)")
    ax2.set_ylabel("доля совпавших с реальностью (пунктир)")
    ax.set_title("Чем дальше модель планирует, тем меньше это можно проверить\n"
                 "обещанная награда при этом не меняется", fontsize=10)
    ax.legend(fontsize=7.5, loc="center left")
    fig.tight_layout(); fig.savefig(REPORTS / "fig_rollout.png"); plt.close(fig)

# ------------------------------------------------------- collapse vs KL
if fin:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    kls = sorted(kl_runs)
    ax.plot(kls, [kl_runs[k]["coverage"] for k in kls], "-o", color="#ff7f0e",
            label="после RL")
    ax.axhline(base["coverage"], color="#1f77b4", ls="--", lw=1.2, label="базовая модель")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("коэффициент KL")
    ax.set_ylabel("доля каталога в топ-10")
    ax.set_title("Схлопывание выдачи: что остаётся от каталога", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(REPORTS / "fig_collapse.png"); plt.close(fig)

print("готово:", ", ".join(p.name for p in sorted(REPORTS.glob("*.png"))))
