"""RL fine-tuning of the recommender, and the measurement that matters.

The policy is nudged towards items the reward model likes. Whether that is an
improvement or an exploit is decided by comparing two numbers at every
checkpoint: the reward the model *predicts* for what it recommends, and the
reward *actually observed* on the held-out week for those of its
recommendations the user really consumed. A widening gap between the two is
reward hacking, measured rather than asserted.
"""
import argparse
import copy
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch

from rlrec import data, evaluate, model as M, rl
from rlrec.config import RUNS, PAD, SEED, MAX_LEN, BOS, MIN_HISTORY

ap = argparse.ArgumentParser()
ap.add_argument("--kl", type=float, default=0.05)
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--group", type=int, default=8)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--eval-every", type=int, default=50)
ap.add_argument("--eval-users", type=int, default=6000)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--sft", default="sft.pt")
ap.add_argument("--reward", default="reward.pt")
ap.add_argument("--seed", type=int, default=SEED)
ap.add_argument("--tag", default=None)
a = ap.parse_args()
tag = a.tag or f"kl{a.kl:g}"

torch.manual_seed(a.seed)
torch.set_num_threads(a.threads)

seqs = data.load_sequences()
vocab = data.vocab_size()

ck = torch.load(RUNS / a.sft, weights_only=False)
policy = M.build_policy(vocab, **ck["cfg"]); policy.load_state_dict(ck["state"])
ref = copy.deepcopy(policy)
for p in ref.parameters():
    p.requires_grad_(False)
ref.eval()

rk = torch.load(RUNS / a.reward, weights_only=False)
rm = M.RewardModel(vocab, **rk["cfg"]); rm.load_state_dict(rk["state"])
for p in rm.parameters():
    p.requires_grad_(False)
rm.eval()

# RL prompts come from the training period only; the validation week is never
# used to drive the updates, only to watch what they do.
train_pool = seqs.filter(data.pl.col("n_train") >= MIN_HISTORY)
toks = train_pool["toks"].to_list()
ntr = train_pool["n_train"].to_list()
rl_ctx, rl_seen = [], []
for t, n in zip(toks, ntr):
    rl_ctx.append([BOS] + t[max(0, n - MAX_LEN + 1):n])
    rl_seen.append(set(t[:n]))
rl_ctx = torch.from_numpy(data._pad(rl_ctx, MAX_LEN))
print(f"{len(rl_ctx)} контекстов для RL, словарь {vocab}", flush=True)

ctx_v, tgt_v, seen_v, _, rew_v = data.eval_contexts(seqs, "val")
if len(ctx_v) > a.eval_users:
    sel = np.random.default_rng(SEED).choice(len(ctx_v), a.eval_users, replace=False)
    ctx_v = ctx_v[sel]
    tgt_v = [tgt_v[i] for i in sel]
    seen_v = [seen_v[i] for i in sel]
    rew_v = [rew_v[i] for i in sel]
print(f"валидация на {len(ctx_v)} пользователях", flush=True)

items_tbl = data.pl.read_parquet(data.DATA / "items.parquet")
vj = json.loads((data.DATA / "vocab.json").read_text())["tok"]
tok_of = {int(k): v for k, v in vj.items()}
dur_of = np.zeros(vocab)
is_series = np.zeros(vocab)
for iid, ct, yr in zip(items_tbl["item_id"], items_tbl["content_type"],
                       items_tbl["release_year"]):
    t = tok_of.get(int(iid))
    if t is not None:
        is_series[t] = 1.0 if ct == "series" else 0.0
        dur_of[t] = float(yr)


def checkpoint(step):
    scores = M.item_scores(policy, ctx_v)
    topk = evaluate.topk_from_scores(scores, seen_v, k=10)
    m = evaluate.ranking_metrics(topk, tgt_v, k=10)
    m.update(evaluate.catalogue_metrics(topk, vocab, k=10))
    m["predicted_reward"] = float(evaluate.predicted_reward(rm, ctx_v, topk).mean())
    m.update(evaluate.realised_reward(topk, rew_v, k=10))
    flat = topk.reshape(-1).numpy()
    m["share_series"] = float(is_series[flat].mean())
    m["mean_release_year"] = float(dur_of[flat][dur_of[flat] > 0].mean())
    m["step"] = step
    return m


opt = torch.optim.AdamW(policy.parameters(), lr=a.lr)
rng = np.random.default_rng(a.seed)
hist = [checkpoint(0)]
b = hist[0]
print(f"старт: ndcg {b['ndcg@10']:.4f} покрытие {b['coverage']:.3f} "
      f"предсказанная награда {b['predicted_reward']:.4f} "
      f"наблюдаемая {b['realised_reward']:.4f} попаданий {b['hits']}", flush=True)

t0 = time.time()
trace = []
for step in range(1, a.steps + 1):
    idx = rng.choice(len(rl_ctx), a.batch, replace=False)
    policy.train()
    loss, info = rl.grpo_step(policy, ref, rm, rl_ctx[idx], [rl_seen[i] for i in idx],
                              k=a.group, kl_coef=a.kl, temperature=a.temperature)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    info["step"] = step
    trace.append(info)
    if step % a.eval_every == 0 or step == a.steps:
        m = checkpoint(step)
        hist.append(m)
        print(f"шаг {step}: награда в группе {info['reward_mean']:.4f} KL {info['kl']:.4f} "
              f"| ndcg {m['ndcg@10']:.4f} покрытие {m['coverage']:.3f} "
              f"предсказ. {m['predicted_reward']:.4f} наблюд. {m['realised_reward']:.4f} "
              f"({time.time()-t0:.0f} с)", flush=True)

torch.save({"state": policy.state_dict(), "vocab": vocab, "cfg": ck["cfg"]},
           RUNS / f"rl_{tag}.pt")
(RUNS / f"rl_{tag}.json").write_text(json.dumps(
    {"args": vars(a), "checkpoints": hist, "trace": trace}, indent=1, ensure_ascii=False))
print(f"\nсохранено в rl_{tag}.json")
