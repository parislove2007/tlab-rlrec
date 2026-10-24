"""Assembles README.md from the written sections and the figures."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from rlrec.config import REPORTS, ROOT

FIGS = [
    ("fig_pareto.png", "Компромисс между релевантностью и доставленной ценностью. "
     "Справа увеличена рабочая область: выше пунктира RL выигрывает у базовой модели."),
    ("fig_hacking.png", "Вверху — расхождение обещанного и наблюдаемого вознаграждения. "
     "Внизу — доставленная ценность: растёт до 50-го шага, затем падает ниже исходной."),
    ("fig_collapse.png", "Какая доля каталога остаётся в выдаче при разном штрафе KL."),
    ("fig_rollout.png", "При генерации вперёд обещанная награда не меняется, "
     "а доля совпавшего с реальностью падает вдвое."),
]
ORDER = ["part0_summary.md", "part1_setup.md", "part2_reward.md", "part3_results.md",
         "part4_tradeoff.md", "part5_longterm.md", "part6_limits.md"]

head = ("# RL-дообучение последовательной рекомендательной модели\n\n"
        "Отчёт по заданию T-Lab 2026, RecSys, задание 2. Датасет — KION от МТС.\n")
parts = []
for f in ORDER:
    p = REPORTS / f
    if not p.exists():
        continue
    t = p.read_text().rstrip()
    if t.startswith("# "):
        t = "\n".join(t.splitlines()[3:]).lstrip()
    parts.append(t)
    if f == "part3_results.md":
        figs = ["\n## Рисунки\n"]
        for name, cap in FIGS:
            if (REPORTS / name).exists():
                figs.append(f"![{cap}](reports/{name})\n\n*{cap}*\n")
        parts.append("\n".join(figs).rstrip())

body = head + "\n\n" + "\n\n".join(parts) + "\n"


def slug(t):
    keep = "".join(c for c in t.lower() if c.isalnum() or c in " -")
    return keep.strip().replace(" ", "-")


toc = ["## Содержание", ""]
for line in body.splitlines():
    if line.startswith("## ") and line[3:] != "Содержание":
        toc.append(f"- [{line[3:]}](#{slug(line[3:])})")
    elif line.startswith("### "):
        toc.append(f"  - [{line[4:]}](#{slug(line[4:])})")
lines = body.splitlines()
cut = next(i for i, l in enumerate(lines) if l.startswith("## "))
body = "\n".join(lines[:cut] + toc + [""] + lines[cut:]) + "\n"
(ROOT / "README.md").write_text(body)
print(f"README.md: {len(body.splitlines())} строк")
