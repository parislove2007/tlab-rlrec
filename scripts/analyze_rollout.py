"""What happens when the model is asked to plan five steps ahead.

Only the first generated item can be checked against reality. From the second
step on the context contains items the user never actually watched, so the
model is reasoning about a history that did not happen. The reward model will
happily keep scoring that trajectory, and it will usually score it higher --
which is the point: the number keeps rising exactly where it stops being
verifiable. This script measures both curves side by side.
"""
import argparse
import glob
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch

from rlrec import data, evaluate, model as M, rl
from rlrec.config import RUNS, PAD, SEED, MAX_LEN, N_SPECIAL

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=5)
ap.add_argument("--users", type=int, default=4000)
ap.add_argument("--split", default="test")
ap.add_argument("--greedy", action="store_true")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--out", default="rollout.json")
a = ap.parse_args()

torch.set_num_threads(a.threads)
seqs = data.load_sequences()
vocab = data.vocab_size()
ctx0, tgt, seen0, _, rewmap = data.eval_contexts(seqs, a.split)
if len(ctx0) > a.users:
    sel = np.random.default_rng(SEED).choice(len(ctx0), a.users, replace=False)
    ctx0 = ctx0[sel]
    tgt = [tgt[i] for i in sel]
    seen0 = [seen0[i] for i in sel]
    rewmap = [rewmap[i] for i in sel]
print(f"{len(ctx0)} пользователей, {a.steps} шагов вперёд", flush=True)

rk = torch.load(RUNS / "reward.pt", weights_only=False)
rm = M.RewardModel(vocab, **rk["cfg"]); rm.load_state_dict(rk["state"]); rm.eval()


def shift_append(ctx, item):
    """Appends a generated item, dropping the oldest token when full."""
    lens = (ctx != PAD).sum(1)
    out = ctx.clone()
    full = lens >= ctx.size(1)
    if full.any():
        out[full] = torch.roll(out[full], -1, dims=1)
        lens = torch.where(full, lens - 1, lens)
    out[torch.arange(len(ctx)), lens] = item
    return out


@torch.no_grad()
def rollout(policy, name):
    policy.eval()
    ctx = ctx0.clone()
    seen = [set(s) for s in seen0]
    rows = []
    for step in range(1, a.steps + 1):
        logits = rl.mask_invalid(M.item_scores(policy, ctx), seen)
        if a.greedy:
            item = logits.argmax(1)
        else:
            item = torch.multinomial(torch.softmax(logits, -1), 1).squeeze(1)
        pred = torch.sigmoid(rm(None, item.unsqueeze(1), h=rm.encode(ctx))).squeeze(1)

        it = item.numpy()
        counts = np.bincount(it, minlength=vocab).astype(float)
        p = counts[counts > 0] / counts.sum()
        # the share of generated items the user really did consume in the window:
        # this is the only honest check, and it is available for every step
        verifiable = float(np.mean([int(i in t) for i, t in zip(it.tolist(), tgt)]))
        rows.append({
            "step": step,
            "predicted_reward": float(pred.mean()),
            "verifiable_share": verifiable,
            "distinct_items": int((counts > 0).sum()),
            "entropy": float(-(p * np.log(p)).sum()),
            "share_of_top100": float(np.sort(counts)[::-1][:100].sum() / counts.sum()),
        })
        for i, t in enumerate(it.tolist()):
            seen[i].add(t)
        ctx = shift_append(ctx, item)
    print(f"  {name}", flush=True)
    for r in rows:
        print(f"    шаг {r['step']}: предсказанная награда {r['predicted_reward']:.4f} "
              f"| совпало с реальностью {r['verifiable_share']:.4f} "
              f"| разных айтемов {r['distinct_items']} энтропия {r['entropy']:.2f}", flush=True)
    return rows


out = {"n_users": len(ctx0), "steps": a.steps, "greedy": a.greedy, "policies": {}}
for path in [RUNS / "sft.pt"] + sorted(pathlib.Path(p) for p in glob.glob(str(RUNS / "rl_*.pt"))):
    if not path.exists():
        continue
    ck = torch.load(path, weights_only=False)
    pol = M.build_policy(vocab, **ck["cfg"]); pol.load_state_dict(ck["state"])
    name = "базовая" if path.name == "sft.pt" else path.stem.replace("rl_", "RL ")
    out["policies"][name] = rollout(pol, name)

(RUNS / a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(f"\nсохранено в {a.out}")
