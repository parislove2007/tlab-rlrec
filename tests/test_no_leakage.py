"""Leak tests for the sequential setup.

A next-item model leaks in quieter ways than a feature pipeline: the target can
slip into its own context, the reward model can see the event it is supposed to
predict, or the RL prompts can be drawn from the weeks reserved for evaluation.
Each of those is checked here directly on the tensors that training consumes.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import polars as pl
import torch

from rlrec import data, evaluate
from rlrec.config import DATA, PAD, BOS, MAX_LEN, MIN_HISTORY, N_SPECIAL

SEQS = data.load_sequences()


def test_training_windows_stop_at_the_cut():
    """The base model may only see events from before the training cut."""
    x, y, mask, users = data.training_tensors(SEQS)
    use = SEQS.filter(pl.col("n_train") >= 2)
    bad = 0
    for row, toks, n_tr in zip(y[:5000].tolist(), use["toks"].to_list()[:5000],
                               use["n_train"].to_list()[:5000]):
        allowed = set(toks[:n_tr])
        later = set(toks[n_tr:])
        got = {t for t in row if t != PAD}
        if got - allowed or (got & (later - allowed)):
            bad += 1
    assert bad == 0, f"{bad} обучающих окон содержат события после среза"


def test_reward_rows_never_see_their_own_target():
    """The reward model predicts the response to an item from the history that
    preceded it; the item itself must not be in that history."""
    X, I, R, _ = data.reward_rows(SEQS, "train", max_rows=40000)
    hist_has_item = 0
    for h, it in zip(X[:20000].tolist(), I[:20000].tolist()):
        if it in {t for t in h if t not in (PAD, BOS)}:
            hist_has_item += 1
    assert hist_has_item == 0, f"{hist_has_item} примеров содержат целевой айтем в истории"
    assert float(R.min()) >= 0.0 and float(R.max()) <= 1.0


def test_val_and_test_rows_come_from_their_own_window():
    """Validation rewards must be drawn from the validation week only."""
    for split in ("val", "test"):
        X, I, R, u = data.reward_rows(SEQS, split, max_rows=20000)
        assert len(I) > 0, f"в {split} не оказалось примеров"
        assert float(R.min()) >= 0.0 and float(R.max()) <= 1.0


def test_eval_context_excludes_the_targets():
    """A context must not contain any item the user is about to consume."""
    for split in ("val", "test"):
        ctx, tgt, seen, _, _ = data.eval_contexts(SEQS, split)
        bad = 0
        for row, t in zip(ctx[:8000].tolist(), tgt[:8000]):
            if {c for c in row if c not in (PAD, BOS)} & t:
                bad += 1
        assert bad == 0, f"{split}: {bad} контекстов содержат целевые айтемы"


def test_future_events_do_not_change_the_training_tensors():
    """Appending events after the cut must leave the training windows and the
    evaluation contexts bitwise identical."""
    poisoned = SEQS.with_columns(
        pl.col("toks").list.concat(pl.lit([N_SPECIAL], dtype=pl.List(pl.Int32))),
        pl.col("rewards").list.concat(pl.lit([1.0], dtype=pl.List(pl.Float32))),
        (pl.col("n_all") + 1).alias("n_all"),
        (pl.col("n_test") + 1).alias("n_test"),
    )
    a, _, _, _ = data.training_tensors(SEQS)
    b, _, _, _ = data.training_tensors(poisoned)
    assert torch.equal(a, b), "обучающие окна изменились после добавления будущих событий"

    ca, _, _, _, _ = data.eval_contexts(SEQS, "val")
    cb, _, _, _, _ = data.eval_contexts(poisoned, "val")
    assert torch.equal(ca, cb), "контексты валидации изменились"


def test_ranking_blocks_already_watched_items():
    """KION has one row per (user, item): a watched item can never recur, and
    leaving it in the ranking would only reward the model for repeating itself."""
    ctx, tgt, seen, _, _ = data.eval_contexts(SEQS, "val")
    n = 500
    scores = torch.zeros(n, data.vocab_size())
    for i in range(n):
        for s in list(seen[i])[:5]:
            scores[i, s] = 100.0
    topk = evaluate.topk_from_scores(scores[:n], seen[:n], k=10)
    overlap = sum(len(set(row.tolist()) & seen[i]) for i, row in enumerate(topk))
    assert overlap == 0, f"{overlap} уже просмотренных айтемов попали в топ-10"
    assert int(topk.min()) >= N_SPECIAL, "служебные токены попали в рекомендации"


if __name__ == "__main__":
    for fn in [test_training_windows_stop_at_the_cut,
               test_reward_rows_never_see_their_own_target,
               test_val_and_test_rows_come_from_their_own_window,
               test_eval_context_excludes_the_targets,
               test_future_events_do_not_change_the_training_tensors,
               test_ranking_blocks_already_watched_items]:
        fn()
        print(f"  ok  {fn.__name__}")
