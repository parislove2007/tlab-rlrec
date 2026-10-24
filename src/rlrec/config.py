from pathlib import Path
import datetime as dt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
for p in (RUNS, REPORTS):
    p.mkdir(parents=True, exist_ok=True)

# Same cuts as the feature-engineering study, so the two assignments are
# measured on the same held-out weeks.
TRAIN_END = dt.date(2021, 8, 2)
VAL_END = dt.date(2021, 8, 9)
TEST_END = dt.date(2021, 8, 23)

PAD, BOS = 0, 1
N_SPECIAL = 2

MAX_LEN = 32            # p90 of history length is 26, and cost grows with the square
MIN_HISTORY = 5         # a user needs this many events before the cut

SEED = 42
