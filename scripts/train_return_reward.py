"""A second reward model, for a long-horizon signal.

Watch fraction says how much of one item was consumed. It says nothing about
whether the person came back. On KION the day-level timestamps support exactly
one long-horizon question that can be answered offline: after this event, did
the user have another one within a week? That is what this model predicts, and
the two signals turn out to be almost uncorrelated, which is the point.
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
ap.add_argument("--max-rows", type=int, default=1_200_000)
ap.add_argument("--horizon", type=int, default=7)
ap.add_argument("--layers", type=int, default=3)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--out", default="reward_return.pt")
a = ap.parse_args()

torch.manual_seed(SEED); torch.set_num_threads(a.threads)
seqs = data.load_sequences()
vocab = data.vocab_size()

t0 = time.time()
Xtr, Itr, Ltr, _ = data.return_rows(seqs, "train", a.horizon, a.max_rows)
Xva, Iva, Lva, _ = data.return_rows(seqs, "val", a.horizon, 200_000)
print(f"обучение {len(Itr)}, валидация {len(Iva)}; доля возвратов "
      f"{float(Ltr.mean()):.3f} / {float(Lva.mean()):.3f} ({time.time()-t0:.0f} с)", flush=True)

# how related is this to the watch-fraction signal at all
_, Iw, Rw, _ = data.reward_rows(seqs, "val", max_rows=200_000)
n = min(len(Rw), len(Lva))
corr = float(np.corrcoef(Lva[:n].numpy(), Rw[:n].numpy())[0, 1])
print(f"корреляция возврата и доли досмотра: {corr:+.3f}", flush=True)

out = {"n_train": len(Itr), "n_val": len(Iva), "horizon_days": a.horizon,
       "return_rate_train": float(Ltr.mean()), "return_rate_val": float(Lva.mean()),
       "corr_with_watch_fraction": corr, "baselines": {}, "history": []}

base = float(Ltr.mean())
auc_base = 0.5
out["baselines"]["constant"] = {"auc": auc_base, "brier": float(((Lva - base) ** 2).mean())}

sums = np.zeros(vocab); cnts = np.zeros(vocab)
np.add.at(sums, Itr.numpy(), Ltr.numpy()); np.add.at(cnts, Itr.numpy(), 1.0)
item_rate = (sums + 20 * base) / (cnts + 20)
pi = item_rate[Iva.numpy()]
out["baselines"]["item_rate"] = {"auc": float(roc_auc_score(Lva.numpy(), pi)),
                                 "brier": float(((Lva.numpy() - pi) ** 2).mean())}
print(f"  baseline по айтему: auc {out['baselines']['item_rate']['auc']:.4f}", flush=True)

rm = M.RewardModel(vocab, n_layer=a.layers)
opt = torch.optim.AdamW(rm.parameters(), lr=a.lr, weight_decay=0.01)
lengths = (Xtr != PAD).sum(1).numpy()
order = np.argsort(lengths, kind="stable")
buckets = [order[i:i + a.batch] for i in range(0, len(order), a.batch)]
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr,
                                            total_steps=a.epochs * len(buckets), pct_start=0.05)
rng = np.random.default_rng(SEED)


@torch.no_grad()
def predict(X, I, batch=512):
    rm.eval()
    return torch.cat([torch.sigmoid(rm(X[i:i + batch], I[i:i + batch]))
                      for i in range(0, len(I), batch)]).numpy()


best = -1
for ep in range(1, a.epochs + 1):
    rm.train(); rng.shuffle(buckets)
    t0, tot, n_ = time.time(), 0.0, 0
    for bi, idx in enumerate(buckets):
        ln = int(lengths[idx].max()) + 1
        loss = F.binary_cross_entropy_with_logits(rm(Xtr[idx, :ln], Itr[idx]), Ltr[idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        tot += float(loss.detach()) * len(idx); n_ += len(idx)
        if bi % 500 == 0:
            print(f"  эпоха {ep} шаг {bi}/{len(buckets)} loss {tot/max(1,n_):.4f}", flush=True)
    p = predict(Xva, Iva)
    m = {"epoch": ep, "train_loss": tot / max(1, n_),
         "auc": float(roc_auc_score(Lva.numpy(), p)),
         "brier": float(((Lva.numpy() - p) ** 2).mean()),
         "spearman": float(spearmanr(p, Lva.numpy()).statistic),
         "seconds": round(time.time() - t0, 1)}
    out["history"].append(m)
    print(f"эпоха {ep}: auc {m['auc']:.4f} brier {m['brier']:.4f}", flush=True)
    if m["auc"] > best:
        best = m["auc"]
        torch.save({"state": rm.state_dict(), "vocab": vocab, "cfg": {"n_layer": a.layers}},
                   RUNS / a.out)

out["best_val_auc"] = best
(RUNS / "reward_return_history.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(f"\nлучший auc {best:.4f}")
