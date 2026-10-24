"""How much of the reward model is personalisation, and how much is a lookup?

If its predictions are nearly a function of the item alone, then RL against it
is not personalising anything: it is re-ranking the catalogue by average
watchability, with extra steps. This measures that directly.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from scipy.stats import spearmanr

from rlrec import data, model as M
from rlrec.config import RUNS, SEED

torch.set_num_threads(8)
seqs = data.load_sequences()
vocab = data.vocab_size()
Xtr, Itr, Rtr, _ = data.reward_rows(seqs, "train", max_rows=400_000)
Xva, Iva, Rva, Uva = data.reward_rows(seqs, "val", max_rows=120_000)

ck = torch.load(RUNS / "reward.pt", weights_only=False)
rm = M.RewardModel(vocab, **ck["cfg"]); rm.load_state_dict(ck["state"]); rm.eval()


@torch.no_grad()
def predict(X, I, b=512):
    return torch.cat([torch.sigmoid(rm(X[i:i + b], I[i:i + b]))
                      for i in range(0, len(I), b)]).numpy()


p = predict(Xva, Iva)
gm = float(Rtr.mean())
sums = np.zeros(vocab); cnts = np.zeros(vocab)
np.add.at(sums, Itr.numpy(), Rtr.numpy()); np.add.at(cnts, Itr.numpy(), 1.0)
item_mean = (sums + 20 * gm) / (cnts + 20)
base = item_mean[Iva.numpy()]

# 1. how close the model is to the item lookup
r_model_item = float(np.corrcoef(p, base)[0, 1])

# 2. how much the model moves a single item across different users
df = {}
for it, pr in zip(Iva.numpy(), p):
    df.setdefault(it, []).append(pr)
spreads = np.array([np.std(v) for v in df.values() if len(v) >= 20])
within = float(spreads.mean())
between = float(np.std([np.mean(v) for v in df.values() if len(v) >= 20]))

# 3. does what is left after removing the item lookup predict anything?
resid_pred = p - base
resid_true = Rva.numpy() - base
rho_resid = float(spearmanr(resid_pred, resid_true).statistic)

out = {
    "corr_model_vs_item_lookup": round(r_model_item, 4),
    "within_item_sd_of_prediction": round(within, 4),
    "between_item_sd_of_prediction": round(between, 4),
    "personalisation_share_of_sd": round(within / (within + between), 4),
    "spearman_of_residuals": round(rho_resid, 4),
    "n_val": int(len(p)),
}
(RUNS / "reward_diagnosis.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps(out, indent=1, ensure_ascii=False))
print(f"\nкорреляция предсказаний со справочником по айтему: {r_model_item:.3f}")
print(f"разброс предсказаний для одного айтема между пользователями: {within:.4f}")
print(f"разброс между айтемами: {between:.4f}")
print(f"остаток после вычитания справочника предсказывает отклик со спирменом {rho_resid:+.3f}")
