import json

import numpy as np
import polars as pl
import torch

from .config import DATA, PAD, BOS, MAX_LEN, TRAIN_END, VAL_END, MIN_HISTORY, SEED


def load_sequences():
    return pl.read_parquet(DATA / "sequences.parquet")


def vocab_size():
    return json.loads((DATA / "vocab.json").read_text())["vocab_size"]


def _pad(rows, length):
    out = np.full((len(rows), length), PAD, dtype=np.int64)
    for i, r in enumerate(rows):
        r = r[-length:]
        out[i, : len(r)] = r
    return out


def training_tensors(seqs, min_events=2):
    """Each user contributes one window of their training-period history.

    The target is the next token at every position, which is the ordinary
    language-modelling objective; padding is masked out of the loss.
    """
    use = seqs.filter(pl.col("n_train") >= min_events)
    rows = [t[:n] for t, n in zip(use["toks"].to_list(), use["n_train"].to_list())]
    x = _pad([[BOS] + r for r in rows], MAX_LEN)
    y = _pad([r + [PAD] for r in rows], MAX_LEN)
    mask = (y != PAD)
    return (torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask),
            use["user_id"].to_numpy())


def reward_rows(seqs, split="train", max_rows=None, seed=SEED):
    """(history, item, reward) triples. The history is everything strictly
    before the item, so the target event never leaks into its own context."""
    hist, items, rew, users = [], [], [], []
    for uid, toks, rs, n_tr, n_val in zip(
        seqs["user_id"].to_list(), seqs["toks"].to_list(), seqs["rewards"].to_list(),
        seqs["n_train"].to_list(), seqs["n_val"].to_list()
    ):
        if split == "train":
            rng = range(1, n_tr)
        elif split == "val":
            rng = range(n_tr, n_tr + n_val)
        else:
            rng = range(n_tr + n_val, len(toks))
        for k in rng:
            if k == 0:
                continue
            hist.append([BOS] + toks[max(0, k - MAX_LEN + 1):k])
            items.append(toks[k])
            rew.append(rs[k])
            users.append(uid)
    if max_rows and len(items) > max_rows:
        keep = np.random.default_rng(seed).choice(len(items), max_rows, replace=False)
        keep.sort()
        hist = [hist[i] for i in keep]
        items = [items[i] for i in keep]
        rew = [rew[i] for i in keep]
        users = [users[i] for i in keep]
    return (torch.from_numpy(_pad(hist, MAX_LEN)),
            torch.tensor(items, dtype=torch.long),
            torch.tensor(rew, dtype=torch.float32),
            np.asarray(users))


def eval_contexts(seqs, split="val"):
    """Context and the set of items the user actually consumed in the window."""
    keep = seqs.filter(
        (pl.col("n_train") >= MIN_HISTORY)
        & ((pl.col("n_val") > 0) if split == "val" else (pl.col("n_test") > 0))
    )
    ctx, targets, seen, users, target_rewards = [], [], [], [], []
    for uid, toks, rs, n_tr, n_val in zip(
        keep["user_id"].to_list(), keep["toks"].to_list(), keep["rewards"].to_list(),
        keep["n_train"].to_list(), keep["n_val"].to_list()
    ):
        cut = n_tr if split == "val" else n_tr + n_val
        end = n_tr + n_val if split == "val" else len(toks)
        ctx.append([BOS] + toks[max(0, cut - MAX_LEN + 1):cut])
        targets.append(set(toks[cut:end]))
        target_rewards.append({t: r for t, r in zip(toks[cut:end], rs[cut:end])})
        seen.append(set(toks[:cut]))
        users.append(uid)
    return (torch.from_numpy(_pad(ctx, MAX_LEN)), targets, seen,
            np.asarray(users), target_rewards)


def return_rows(seqs, split="train", horizon=7, max_rows=None, seed=SEED):
    """A long-horizon target: did the user come back within `horizon` days?

    Watch fraction is a within-session signal and says nothing about whether the
    service kept the person. KION's day-level timestamps do support one
    long-horizon question that can be answered offline: after consuming this
    item, did the user have any further event inside the next week? It is a
    proxy, not retention, and the limitation is stated in the report.
    """
    hist, items, lab, users = [], [], [], []
    for uid, toks, days, n_tr, n_val in zip(
        seqs["user_id"].to_list(), seqs["toks"].to_list(), seqs["days"].to_list(),
        seqs["n_train"].to_list(), seqs["n_val"].to_list()
    ):
        if split == "train":
            rng_ = range(1, n_tr)
            horizon_end = None
        elif split == "val":
            rng_ = range(n_tr, n_tr + n_val)
        else:
            rng_ = range(n_tr + n_val, len(toks))
        for k in rng_:
            d = days[k]
            came_back = any((days[j] - d).days <= horizon and days[j] > d
                            for j in range(k + 1, len(toks)))
            hist.append([BOS] + toks[max(0, k - MAX_LEN + 1):k])
            items.append(toks[k])
            lab.append(1.0 if came_back else 0.0)
            users.append(uid)
    if max_rows and len(items) > max_rows:
        keep = np.random.default_rng(seed).choice(len(items), max_rows, replace=False)
        keep.sort()
        hist = [hist[i] for i in keep]
        items = [items[i] for i in keep]
        lab = [lab[i] for i in keep]
        users = [users[i] for i in keep]
    return (torch.from_numpy(_pad(hist, MAX_LEN)),
            torch.tensor(items, dtype=torch.long),
            torch.tensor(lab, dtype=torch.float32),
            np.asarray(users))
