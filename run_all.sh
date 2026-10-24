#!/usr/bin/env bash
# Full reproduction. About two and a half hours on an eight-core laptop.
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1
PY=${PY:-python}

$PY scripts/prepare_sequences.py
$PY tests/test_no_leakage.py

$PY scripts/train_sft.py --epochs 4
$PY scripts/train_reward.py --epochs 2
$PY scripts/diagnose_reward.py
$PY scripts/train_return_reward.py --epochs 2

for kl in 0.0 0.01 0.05 0.2 0.5 1.0 2.0 5.0; do
  $PY scripts/train_rl.py --kl $kl --steps 300
done
$PY scripts/train_rl.py --kl 0.05 --steps 60 --eval-every 20 --tag early60
$PY scripts/train_rl.py --kl 0.05 --steps 300 --reward reward_return.pt --tag return_kl0.05
# the same setting several times: anything smaller than this spread is noise
for sd in 1 2 3; do
  $PY scripts/train_rl.py --kl 2.0 --steps 300 --seed $sd --tag kl2_seed$sd
done

$PY scripts/evaluate_all.py --split test --users 20000
$PY scripts/analyze_rollout.py --steps 5 --users 4000 --split test
$PY scripts/make_figures.py
$PY scripts/make_report.py
$PY scripts/make_pdf.py
