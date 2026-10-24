"""Turns the event log into per-user item sequences and reports what the
day-level timestamps do to the notion of a sequence."""
import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import polars as pl
from rlrec.config import DATA, RUNS, TRAIN_END, VAL_END, TEST_END, MIN_HISTORY, N_SPECIAL

ev = pl.read_parquet(DATA / "events.parquet")
print("events", ev.shape, ev["ts"].min(), "->", ev["ts"].max())

# how much of the ordering is actually knowable
per_day = ev.group_by("user_id", "ts").agg(pl.len().alias("n"))
share_multi = float((per_day["n"] > 1).sum() / per_day.height)
mean_multi = float(per_day.filter(pl.col("n") > 1)["n"].mean())
print(f"user-days with more than one event: {share_multi:.1%}, "
      f"average size of those days: {mean_multi:.2f}")

items = np.sort(ev["item_id"].unique().to_numpy())
tok = {int(i): k + N_SPECIAL for k, i in enumerate(items)}
inv = {v: k for k, v in tok.items()}
vocab_size = len(items) + N_SPECIAL
print("items", len(items), "vocab", vocab_size)

ev = ev.with_columns(
    pl.col("item_id").replace_strict(tok).cast(pl.Int32).alias("tok"),
    (pl.col("watched_pct").clip(0, 100) / 100.0).cast(pl.Float32).alias("reward"),
)

# Day granularity means within-day order is arbitrary; item_id makes it at
# least deterministic and reproducible.
ev = ev.sort(["user_id", "ts", "item_id"])

seqs = ev.group_by("user_id", maintain_order=True).agg(
    pl.col("tok").alias("toks"),
    pl.col("ts").alias("days"),
    pl.col("reward").alias("rewards"),
    pl.col("total_dur").alias("durs"),
)
print("users with any history:", seqs.height)

def split_at(cut):
    return pl.col("days").list.eval(pl.element() < cut).list.sum()

seqs = seqs.with_columns(
    pl.col("days").list.eval(pl.element() < TRAIN_END).list.sum().alias("n_train"),
    pl.col("days").list.eval(pl.element() < VAL_END).list.sum().alias("n_val_end"),
    pl.col("days").list.len().alias("n_all"),
)
seqs = seqs.with_columns(
    (pl.col("n_val_end") - pl.col("n_train")).alias("n_val"),
    (pl.col("n_all") - pl.col("n_val_end")).alias("n_test"),
)

usable = seqs.filter(pl.col("n_train") >= MIN_HISTORY)
print(f"users with >= {MIN_HISTORY} events before {TRAIN_END}: {usable.height}")
print("of them with a validation target:", int((usable['n_val'] > 0).sum()))
print("of them with a test target:", int((usable['n_test'] > 0).sum()))

seqs.write_parquet(DATA / "sequences.parquet")
(DATA / "vocab.json").write_text(json.dumps(
    {"tok": {str(k): v for k, v in tok.items()}, "vocab_size": vocab_size}))

stats = {
    "n_events": ev.height, "n_items": len(items), "vocab_size": vocab_size,
    "n_users_any": seqs.height, "n_users_usable": usable.height,
    "users_with_val_target": int((usable["n_val"] > 0).sum()),
    "users_with_test_target": int((usable["n_test"] > 0).sum()),
    "share_user_days_with_several_events": round(share_multi, 4),
    "mean_events_on_those_days": round(mean_multi, 3),
    "train_end": str(TRAIN_END), "val_end": str(VAL_END), "test_end": str(TEST_END),
    "history_len": {
        "median": float(usable["n_train"].median()),
        "p90": float(usable["n_train"].quantile(0.9)),
        "max": int(usable["n_train"].max()),
    },
    "reward": {
        "mean": float(ev["reward"].mean()), "median": float(ev["reward"].median()),
        "share_above_0.9": float((ev["reward"] >= 0.9).mean()),
    },
}
(RUNS / "data_stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
print(json.dumps(stats, indent=1, ensure_ascii=False))
