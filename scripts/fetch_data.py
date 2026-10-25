"""Downloads KION and builds the two tables the rest of the pipeline needs.

Only `events.parquet` and `items.parquet` are produced here; everything else
(`sequences.parquet`, `vocab.json`) is derived by `prepare_sequences.py`. The
script is idempotent and skips any step whose output already exists.
"""
import pathlib
import sys
import urllib.request
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import polars as pl

from rlrec.config import DATA

URL = "https://storage.yandexcloud.net/itmo-recsys-public-data/kion_train.zip"
RAW = DATA.parent / "data_raw"
WATCH_PCT = 30.0
FINISH_PCT = 90.0

DATA.mkdir(parents=True, exist_ok=True)
if (DATA / "events.parquet").exists() and (DATA / "items.parquet").exists():
    print("events.parquet и items.parquet уже на месте, скачивание не нужно")
    raise SystemExit(0)

RAW.mkdir(parents=True, exist_ok=True)
src = RAW / "kion_train"
if not src.exists():
    zp = RAW / "kion_train.zip"
    if not zp.exists():
        print(f"скачиваю {URL} ...", flush=True)
        urllib.request.urlretrieve(URL, zp)
    print("распаковываю ...", flush=True)
    with zipfile.ZipFile(zp) as z:
        z.extractall(RAW)

ev = (
    pl.read_csv(src / "interactions.csv", try_parse_dates=True)
    .rename({"last_watch_dt": "ts"})
    .with_columns(pl.col("watched_pct").fill_null(0.0), pl.col("total_dur").fill_null(0))
    .with_columns((pl.col("watched_pct") >= WATCH_PCT).alias("is_watch"),
                  (pl.col("watched_pct") >= FINISH_PCT).alias("is_finish"))
    .sort("ts")
)
ev.write_parquet(DATA / "events.parquet")
print("events.parquet:", ev.shape, ev["ts"].min(), "->", ev["ts"].max())

items = pl.read_csv(src / "items.csv")
gen = (
    items.select("item_id", pl.col("genres").fill_null("__na__").str.split(", "))
    .explode("genres").with_columns(pl.col("genres").str.strip_chars())
)
freq = gen.group_by("genres").agg(pl.len().alias("n"))
primary = (
    gen.join(freq, on="genres").sort("n", descending=True)
    .group_by("item_id").agg(pl.col("genres").first().alias("primary_genre"))
)
out = items.select(
    "item_id", "content_type",
    pl.col("release_year").fill_null(-1).cast(pl.Int32),
    pl.col("age_rating").fill_null(-1).cast(pl.Int32),
    pl.col("for_kids").fill_null(0).cast(pl.Int32),
    pl.col("genres").str.split(", ").list.len().cast(pl.Int32).alias("n_genres"),
    pl.col("title").str.len_chars().cast(pl.Int32).alias("title_len"),
    pl.col("description").fill_null("").str.len_chars().cast(pl.Int32).alias("descr_len"),
    pl.col("keywords").fill_null("").str.split(", ").list.len().cast(pl.Int32).alias("n_keywords"),
).join(primary, on="item_id", how="left").with_columns(
    pl.col("primary_genre").fill_null("__na__"))
out.write_parquet(DATA / "items.parquet")
print("items.parquet:", out.shape)
