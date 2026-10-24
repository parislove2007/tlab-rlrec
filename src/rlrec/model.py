import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model

from .config import MAX_LEN, PAD


def policy_config(vocab, n_embd=128, n_layer=4, n_head=4, dropout=0.1):
    return GPT2Config(
        vocab_size=vocab, n_positions=MAX_LEN, n_embd=n_embd, n_layer=n_layer,
        n_head=n_head, n_inner=4 * n_embd, resid_pdrop=dropout, embd_pdrop=dropout,
        attn_pdrop=dropout, bos_token_id=None, eos_token_id=None, pad_token_id=PAD,
    )


def build_policy(vocab, **kw):
    """The recommender itself: ordinary GPT-2, with item ids as tokens."""
    return GPT2LMHeadModel(policy_config(vocab, **kw))


class RewardModel(nn.Module):
    """Predicts how much of the item the user will actually consume.

    A GPT-2 trunk reads the history, and the last hidden state is combined with
    the candidate's embedding. The elementwise product is in there because the
    answer depends on the match between the two, not on either alone.
    """

    def __init__(self, vocab, n_embd=128, n_layer=3, n_head=4, dropout=0.1):
        super().__init__()
        self.trunk = GPT2Model(policy_config(vocab, n_embd, n_layer, n_head, dropout))
        self.item = nn.Embedding(vocab, n_embd)
        self.head = nn.Sequential(
            nn.Linear(3 * n_embd, 2 * n_embd), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * n_embd, 1),
        )
        self.item_bias = nn.Embedding(vocab, 1)
        nn.init.zeros_(self.item_bias.weight)

    def encode(self, hist):
        out = self.trunk(input_ids=hist, attention_mask=(hist != PAD).long())
        lengths = (hist != PAD).sum(1).clamp(min=1) - 1
        return out.last_hidden_state[torch.arange(len(hist)), lengths]

    def forward(self, hist, items, h=None):
        if h is None:
            h = self.encode(hist)
        if items.dim() == 2:                      # several candidates per history
            h = h.unsqueeze(1).expand(-1, items.size(1), -1)
        e = self.item(items)
        z = self.head(torch.cat([h, e, h * e], dim=-1)).squeeze(-1)
        return z + self.item_bias(items).squeeze(-1)


@torch.no_grad()
def item_scores(policy, ctx, batch=512):
    """Next-item logits for a batch of contexts."""
    policy.eval()
    out = []
    for i in range(0, len(ctx), batch):
        c = ctx[i:i + batch]
        lengths = (c != PAD).sum(1).clamp(min=1) - 1
        logits = policy(input_ids=c, attention_mask=(c != PAD).long()).logits
        out.append(logits[torch.arange(len(c)), lengths])
    return torch.cat(out)
