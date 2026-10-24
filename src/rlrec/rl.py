"""Group-relative policy optimisation for a next-item recommender.

For each user context the policy samples a group of K candidate items. The
reward model scores them, the advantage of each is its reward minus the mean of
its own group, and the policy is nudged towards the better half. No value
network is needed, which matters on a laptop.

A KL penalty towards the frozen starting policy is the only thing standing
between "higher predicted reward" and a model that recommends the same handful
of items to everyone, so its coefficient is the knob this study sweeps.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .config import PAD, N_SPECIAL


def last_logits(policy, ctx):
    lengths = (ctx != PAD).sum(1).clamp(min=1) - 1
    out = policy(input_ids=ctx, attention_mask=(ctx != PAD).long()).logits
    return out[torch.arange(len(ctx)), lengths]


def mask_invalid(logits, seen=None):
    """Specials are never valid; already-watched items cannot recur in KION."""
    logits = logits.clone()
    logits[:, :N_SPECIAL] = -1e9
    if seen is not None:
        for i, h in enumerate(seen):
            if h:
                logits[i, list(h)] = -1e9
    return logits


def sample_group(policy, ctx, seen, k, temperature=1.0):
    with torch.no_grad():
        logits = mask_invalid(last_logits(policy, ctx), seen)
        probs = torch.softmax(logits / temperature, dim=-1)
        items = torch.multinomial(probs, k, replacement=False)
    return items


def grpo_step(policy, ref_policy, reward_model, ctx, seen, k=8, kl_coef=0.05,
              temperature=1.0, clip=0.2, advantage_clip=4.0):
    """One optimisation step. Returns the loss and what happened inside it."""
    items = sample_group(policy, ctx, seen, k, temperature)

    with torch.no_grad():
        h = reward_model.encode(ctx)
        rewards = torch.sigmoid(reward_model(None, items, h=h))
        adv = rewards - rewards.mean(dim=1, keepdim=True)
        sd = rewards.std(dim=1, keepdim=True)
        adv = (adv / (sd + 1e-6)).clamp(-advantage_clip, advantage_clip)
        ref_logp = torch.log_softmax(mask_invalid(last_logits(ref_policy, ctx), seen), -1)
        old_logp = torch.log_softmax(mask_invalid(last_logits(policy, ctx), seen), -1)
        old_tok = old_logp.gather(1, items)

    logits = mask_invalid(last_logits(policy, ctx), seen)
    logp = torch.log_softmax(logits, dim=-1)
    tok_logp = logp.gather(1, items)

    ratio = torch.exp(tok_logp - old_tok)
    unclipped = ratio * adv
    clipped = ratio.clamp(1 - clip, 1 + clip) * adv
    pg = -torch.min(unclipped, clipped).mean()

    # forward KL over the full item distribution, not just the sampled ones:
    # the whole worry is about mass quietly moving to a few popular items
    p = logp.exp()
    kl = (p * (logp - ref_logp)).sum(-1).mean()

    loss = pg + kl_coef * kl
    info = {
        "reward_mean": float(rewards.mean()),
        "reward_max": float(rewards.max(dim=1).values.mean()),
        "kl": float(kl.detach()),
        "pg_loss": float(pg.detach()),
        "ratio_mean": float(ratio.mean().detach()),
        "entropy": float(-(p * logp).sum(-1).mean().detach()),
    }
    return loss, info
