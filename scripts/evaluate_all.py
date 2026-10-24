"""The one script that reads the test weeks.

Every arm is scored under the same protocol: the same users, the same blocked
history, the same k. Two of the arms are there to keep the RL honest — a
best-of-N reranker that reaches for the same reward with no training at all,
and plain popularity, which is what a collapsed policy degenerates into.
"""
import argparse
import glob
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import polars as pl
import torch

from rlrec import data, evaluate, model as M
from rlrec.config import RUNS, DATA, N_SPECIAL, SEED

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test")
ap.add_argument("--users", type=int, default=20000)
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--rerank-pool", type=int, default=50)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--out", default="final_test.json")
a = ap.parse_args()

torch.set_num_threads(a.threads)
seqs = data.load_sequences()
vocab = data.vocab_size()
ctx, tgt, seen, users, rewmap = data.eval_contexts(seqs, a.split)
if len(ctx) > a.users:
    sel = np.random.default_rng(SEED).choice(len(ctx), a.users, replace=False)
    ctx, users = ctx[sel], users[sel]
    tgt = [tgt[i] for i in sel]
    seen = [seen[i] for i in sel]
    rewmap = [rewmap[i] for i in sel]
print(f"{a.split}: {len(ctx)} пользователей", flush=True)

rk = torch.load(RUNS / "reward.pt", weights_only=False)
rm = M.RewardModel(vocab, **rk["cfg"]); rm.load_state_dict(rk["state"]); rm.eval()

items_tbl = pl.read_parquet(DATA / "items.parquet")
tok_of = {int(k): v for k, v in json.loads((DATA / "vocab.json").read_text())["tok"].items()}
is_series = np.zeros(vocab); year = np.zeros(vocab)
for iid, ct, yr in zip(items_tbl["item_id"], items_tbl["content_type"], items_tbl["release_year"]):
    t = tok_of.get(int(iid))
    if t is not None:
        is_series[t] = 1.0 if ct == "series" else 0.0
        year[t] = float(yr)

# how often each item was consumed in the training period, for the popularity arm
ev_pop = np.zeros(vocab)
for toks, n in zip(seqs["toks"].to_list(), seqs["n_train"].to_list()):
    for t in toks[:n]:
        ev_pop[t] += 1


def score_arm(name, topk, seconds):
    m = evaluate.ranking_metrics(topk, tgt, k=a.k)
    m.update(evaluate.catalogue_metrics(topk, vocab, k=a.k))
    m["predicted_reward"] = float(evaluate.predicted_reward(rm, ctx, topk).mean())
    m.update(evaluate.realised_reward(topk, rewmap, k=a.k))
    flat = topk[:, :a.k].reshape(-1).numpy()
    m["share_series"] = float(is_series[flat].mean())
    m["mean_release_year"] = float(year[flat][year[flat] > 0].mean())
    m["median_train_popularity"] = float(np.median(ev_pop[flat]))
    m["seconds"] = round(seconds, 1)
    print(f"  {name:26} ndcg {m[f'ndcg@{a.k}']:.4f} recall {m[f'recall@{a.k}']:.4f} "
          f"| предсказ.награда {m['predicted_reward']:.4f} наблюд. {m['realised_reward']:.4f} "
          f"| покрытие {m['coverage']:.3f} сериалов {m['share_series']:.2f}", flush=True)
    return m, topk


results, tops = {}, {}


def eval_policy(name, path):
    ck = torch.load(path, weights_only=False)
    pol = M.build_policy(vocab, **ck["cfg"]); pol.load_state_dict(ck["state"]); pol.eval()
    t0 = time.time()
    scores = M.item_scores(pol, ctx)
    topk = evaluate.topk_from_scores(scores, seen, k=a.k)
    results[name], tops[name] = score_arm(name, topk, time.time() - t0)
    return pol, scores


# --- trivial references -------------------------------------------------
t0 = time.time()
pop_scores = torch.from_numpy(np.tile(np.log1p(ev_pop), (len(ctx), 1))).float()
results["популярность"], tops["популярность"] = score_arm(
    "популярность", evaluate.topk_from_scores(pop_scores, seen, k=a.k), time.time() - t0)

# the far end of the direction RL pushes in: show everyone the ten items with
# the highest average watch fraction and nothing else
t0 = time.time()
sums = np.zeros(vocab); cnts = np.zeros(vocab)
for toks, rs, n in zip(seqs["toks"].to_list(), seqs["rewards"].to_list(),
                       seqs["n_train"].to_list()):
    for t, r in zip(toks[:n], rs[:n]):
        sums[t] += r; cnts[t] += 1
item_watch = (sums + 20 * 0.46) / (cnts + 20)
item_watch[cnts < 50] = 0.0            # otherwise it is a long tail of noise
greedy = torch.from_numpy(np.tile(item_watch, (len(ctx), 1))).float()
results["самые досматриваемые"], tops["самые досматриваемые"] = score_arm(
    "самые досматриваемые", evaluate.topk_from_scores(greedy, seen, k=a.k), time.time() - t0)

t0 = time.time()
rnd = torch.from_numpy(np.random.default_rng(SEED).random((len(ctx), vocab))).float()
results["случайный выбор"], tops["случайный выбор"] = score_arm(
    "случайный выбор", evaluate.topk_from_scores(rnd, seen, k=a.k), time.time() - t0)

# --- the base model and the reranker that needs no RL -------------------
sft, sft_scores = eval_policy("базовая GPT-2", RUNS / "sft.pt")

t0 = time.time()
pool = evaluate.topk_from_scores(sft_scores, seen, k=a.rerank_pool)
with torch.no_grad():
    chosen = []
    for i in range(0, len(ctx), 256):
        h = rm.encode(ctx[i:i + 256])
        z = rm(None, pool[i:i + 256], h=h)
        order = torch.argsort(z, dim=1, descending=True)[:, :a.k]
        chosen.append(torch.gather(pool[i:i + 256], 1, order))
    rerank = torch.cat(chosen)
results[f"best-of-{a.rerank_pool} по reward"], tops["best_of_n"] = score_arm(
    f"best-of-{a.rerank_pool} по reward", rerank, time.time() - t0)

# --- every RL run that exists ------------------------------------------
for f in sorted(glob.glob(str(RUNS / "rl_*.pt"))):
    name = "RL, " + pathlib.Path(f).stem.replace("rl_", "")
    eval_policy(name, f)

# how much the arms overlap with the base model's own list
base = tops["базовая GPT-2"]
for name, tk in tops.items():
    if name == "базовая GPT-2":
        continue
    ov = np.mean([len(set(x.tolist()) & set(y.tolist())) / a.k
                  for x, y in zip(tk[:, :a.k], base[:, :a.k])])
    key = [k for k in results if k.startswith(name.split(",")[0])]
    for kk in results:
        if results[kk] is not None and kk.startswith(name[:12]):
            results[kk]["overlap_with_base"] = float(ov)

(RUNS / a.out).write_text(json.dumps(
    {"split": a.split, "n_users": len(ctx), "k": a.k, "results": results},
    indent=1, ensure_ascii=False))
print(f"\nсохранено в {a.out}")
