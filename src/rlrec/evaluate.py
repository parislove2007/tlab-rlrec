import math

import numpy as np
import torch

from .config import N_SPECIAL


def topk_from_scores(scores, seen, k=10, block_seen=True):
    """Top-k items per row, with the user's own history removed.

    KION holds one row per (user, item), so an item already watched can never
    be the next one; leaving it in the ranking would only measure the model's
    ability to repeat itself.
    """
    s = scores.clone()
    s[:, :N_SPECIAL] = -1e9
    if block_seen:
        for i, hist in enumerate(seen):
            if hist:
                s[i, list(hist)] = -1e9
    return torch.topk(s, k, dim=1).indices


def ranking_metrics(topk, targets, k=10):
    rec, ndcg, ap, hit = [], [], [], []
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    for row, tgt in zip(topk.tolist(), targets):
        if not tgt:
            continue
        rel = np.array([1.0 if t in tgt else 0.0 for t in row[:k]])
        n = len(tgt)
        ideal = disc[: min(n, k)].sum()
        ndcg.append(float((rel * disc).sum() / ideal))
        rec.append(float(rel.sum() / n))
        prec = np.cumsum(rel) / np.arange(1, k + 1)
        ap.append(float((prec * rel).sum() / min(n, k)))
        hit.append(float(rel.sum() > 0))
    return {f"recall@{k}": float(np.mean(rec)), f"ndcg@{k}": float(np.mean(ndcg)),
            f"map@{k}": float(np.mean(ap)), f"hit@{k}": float(np.mean(hit)),
            "users": len(rec)}


def catalogue_metrics(topk, vocab, k=10):
    """Is the model recommending the same handful of items to everyone?"""
    flat = topk[:, :k].reshape(-1).numpy()
    counts = np.bincount(flat, minlength=vocab).astype(np.float64)
    shown = counts > 0
    p = counts[shown] / counts.sum()
    ent = float(-(p * np.log(p)).sum())
    srt = np.sort(counts)
    cum = np.cumsum(srt)
    gini = float((len(srt) + 1 - 2 * (cum / cum[-1]).sum()) / len(srt))
    top100 = float(np.sort(counts)[::-1][:100].sum() / counts.sum())
    return {"coverage": float(shown.sum() / (vocab - N_SPECIAL)),
            "distinct_items": int(shown.sum()),
            "entropy": ent, "entropy_norm": ent / math.log(max(2, int(shown.sum()))),
            "gini": gini, "share_of_top100": top100}


@torch.no_grad()
def predicted_reward(reward_model, ctx, topk, batch=256):
    """Mean reward the reward model expects from what the policy recommends."""
    reward_model.eval()
    vals = []
    for i in range(0, len(ctx), batch):
        h = reward_model.encode(ctx[i:i + batch])
        z = reward_model(None, topk[i:i + batch], h=h)
        vals.append(torch.sigmoid(z).mean(1))
    return torch.cat(vals)


def realised_reward(topk, target_rewards, k=10):
    """Reward actually observed on the held-out week.

    Two numbers, and the difference between them matters. `realised_reward`
    averages over the recommendations the user actually consumed, which is what
    the reward model can be wrong about -- but its denominator shrinks as the
    policy loses hits, so a policy that recommends fewer but safer items looks
    better than it is. `reward_per_user` divides by the number of users
    instead, which is fixed, and is therefore the honest measure of how much
    watch time the recommendations actually delivered."""
    hits, total = 0, 0.0
    for row, rmap in zip(topk[:, :k].tolist(), target_rewards):
        for t in row:
            if t in rmap:
                total += rmap[t]
                hits += 1
    n = max(1, len(target_rewards))
    return {"realised_reward": (total / hits) if hits else float("nan"),
            "hits": hits,
            "hit_rate_per_user": hits / n,
            "reward_per_user": total / n}
