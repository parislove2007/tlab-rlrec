"""Trains the reward model: how much of a recommended item will the user watch?

Two baselines are measured alongside it, because a reward model that only
memorises each item's average watch fraction is not modelling the user at all,
and the RL stage would then be optimising item popularity under another name.
"""
import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from rlrec import data, model as M
from rlrec.config import RUNS, PAD, SEED

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--max-rows", type=int, default=1_500_000)
ap.add_argument("--layers", type=int, default=3)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--out", default="reward.pt")
a = ap.parse_args()

torch.manual_seed(SEED)
torch.set_num_threads(a.threads)
seqs = data.load_sequences()
vocab = data.vocab_size()

t0 = time.time()
Xtr, Itr, Rtr, _ = data.reward_rows(seqs, "train", max_rows=a.max_rows)
Xva, Iva, Rva, _ = data.reward_rows(seqs, "val", max_rows=200_000)
print(f"обучение {len(Itr)}, валидация {len(Iva)} примеров ({time.time()-t0:.0f} с)", flush=True)


def report(pred, true, tag):
    pred, true = np.asarray(pred), np.asarray(true)
    m = {"mae": float(np.abs(pred - true).mean()),
         "rmse": float(np.sqrt(((pred - true) ** 2).mean())),
         "spearman": float(spearmanr(pred, true).statistic)}
    lab = (true >= 0.5).astype(int)
    if 0 < lab.sum() < len(lab):
        m["auc_watched_half"] = float(roc_auc_score(lab, pred))
    print(f"  {tag:22} mae {m['mae']:.4f} rmse {m['rmse']:.4f} "
          f"spearman {m['spearman']:+.3f} auc {m.get('auc_watched_half', float('nan')):.4f}",
          flush=True)
    return m


out = {"n_train": len(Itr), "n_val": len(Iva), "baselines": {}, "history": []}

# baseline 1: the global mean watch fraction
gm = float(Rtr.mean())
out["baselines"]["global_mean"] = report(np.full(len(Rva), gm), Rva.numpy(), "среднее по всем")

# baseline 2: the item's own average watch fraction, smoothed towards the mean
sums = np.zeros(vocab); cnts = np.zeros(vocab)
np.add.at(sums, Itr.numpy(), Rtr.numpy())
np.add.at(cnts, Itr.numpy(), 1.0)
item_mean = (sums + 20 * gm) / (cnts + 20)
out["baselines"]["item_mean"] = report(item_mean[Iva.numpy()], Rva.numpy(), "среднее по айтему")

rm = M.RewardModel(vocab, n_layer=a.layers)
opt = torch.optim.AdamW(rm.parameters(), lr=a.lr, weight_decay=0.01)
lengths = (Xtr != PAD).sum(1).numpy()
order = np.argsort(lengths, kind="stable")
buckets = [order[i:i + a.batch] for i in range(0, len(order), a.batch)]
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr,
                                            total_steps=a.epochs * len(buckets),
                                            pct_start=0.05)
rng = np.random.default_rng(SEED)


@torch.no_grad()
def predict(X, I, batch=512):
    rm.eval()
    out = []
    for i in range(0, len(I), batch):
        z = rm(X[i:i + batch], I[i:i + batch])
        out.append(torch.sigmoid(z))
    return torch.cat(out).numpy()


best = -1
for ep in range(1, a.epochs + 1):
    rm.train(); rng.shuffle(buckets)
    t0, tot, n = time.time(), 0.0, 0
    for bi, idx in enumerate(buckets):
        ln = int(lengths[idx].max()) + 1
        z = rm(Xtr[idx, :ln], Itr[idx])
        loss = F.binary_cross_entropy_with_logits(z, Rtr[idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        tot += float(loss.detach()) * len(idx); n += len(idx)
        if bi % 500 == 0:
            print(f"  эпоха {ep} шаг {bi}/{len(buckets)} loss {tot/max(1,n):.4f} "
                  f"({time.time()-t0:.0f} с)", flush=True)
    m = report(predict(Xva, Iva), Rva.numpy(), f"модель, эпоха {ep}")
    m["epoch"] = ep; m["train_loss"] = tot / max(1, n)
    m["seconds"] = round(time.time() - t0, 1)
    out["history"].append(m)
    if m["spearman"] > best:
        best = m["spearman"]
        torch.save({"state": rm.state_dict(), "vocab": vocab,
                    "cfg": {"n_layer": a.layers}}, RUNS / a.out)

out["best_val_spearman"] = best
(RUNS / "reward_history.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(f"\nлучший spearman на валидации {best:+.3f}, модель в {RUNS / a.out}")
