"""Trains the base recommender: GPT-2 on next-item prediction.

Nothing here reads anything at or after the training cut, so the model that the
RL stage starts from has never seen the validation or test weeks.
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

from rlrec import data, evaluate, model as M
from rlrec.config import RUNS, PAD, SEED, MAX_LEN

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--embd", type=int, default=128)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--out", default="sft.pt")
a = ap.parse_args()

torch.manual_seed(SEED)
torch.set_num_threads(a.threads)

seqs = data.load_sequences()
x, y, mask, _ = data.training_tensors(seqs)
lengths = mask.sum(1).numpy()
vocab = data.vocab_size()
print(f"{len(x)} последовательностей, {int(mask.sum())} токенов в лоссе, словарь {vocab}", flush=True)

# Batching by length: the median history is 9 items, so padding everything to
# 32 would spend most of the compute on padding.
order = np.argsort(lengths, kind="stable")
buckets = [order[i:i + a.batch] for i in range(0, len(order), a.batch)]

ctx_v, tgt_v, seen_v, _, _ = data.eval_contexts(seqs, "val")
print(f"валидация: {len(ctx_v)} пользователей", flush=True)

policy = M.build_policy(vocab, n_embd=a.embd, n_layer=a.layers)
opt = torch.optim.AdamW(policy.parameters(), lr=a.lr, weight_decay=0.01)
steps = a.epochs * len(buckets)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps,
                                            pct_start=0.05)

rng = np.random.default_rng(SEED)
history, best = [], -1
for ep in range(1, a.epochs + 1):
    policy.train()
    rng.shuffle(buckets)
    t0, tot, seen_tok = time.time(), 0.0, 0
    for bi, idx in enumerate(buckets):
        n = int(lengths[idx].max()) + 1
        xb, yb, mb = x[idx, :n], y[idx, :n], mask[idx, :n]
        logits = policy(input_ids=xb, attention_mask=(xb != PAD).long()).logits
        loss = F.cross_entropy(logits[mb], yb[mb])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        tot += float(loss) * int(mb.sum()); seen_tok += int(mb.sum())
        if bi % 200 == 0:
            print(f"  эпоха {ep} шаг {bi}/{len(buckets)} loss {tot/max(1,seen_tok):.4f} "
                  f"({time.time()-t0:.0f} с)", flush=True)

    scores = M.item_scores(policy, ctx_v)
    topk = evaluate.topk_from_scores(scores, seen_v, k=10)
    m = evaluate.ranking_metrics(topk, tgt_v, k=10)
    m.update(evaluate.catalogue_metrics(topk, vocab, k=10))
    m["epoch"] = ep
    m["train_loss"] = tot / max(1, seen_tok)
    m["seconds"] = round(time.time() - t0, 1)
    history.append(m)
    print(f"эпоха {ep}: loss {m['train_loss']:.4f} ndcg@10 {m['ndcg@10']:.4f} "
          f"recall@10 {m['recall@10']:.4f} покрытие {m['coverage']:.3f} "
          f"({m['seconds']} с)", flush=True)
    if m["ndcg@10"] > best:
        best = m["ndcg@10"]
        torch.save({"state": policy.state_dict(), "vocab": vocab,
                    "cfg": {"n_embd": a.embd, "n_layer": a.layers}}, RUNS / a.out)

(RUNS / "sft_history.json").write_text(json.dumps(
    {"args": vars(a), "history": history, "best_val_ndcg": best}, indent=1, ensure_ascii=False))
print(f"\nлучший val ndcg@10 {best:.4f}, модель в {RUNS / a.out}")
