"""Masked-diffusion sampling loop with structural early-stop (design doc §6-7).

Phase-1 reconstruction order = pre-order, left-to-right (gap-free), so the grammar's
`dangling==0` early-stop is exact. Validity is enforced in-loop via the grammar's
valid-token mask; sequences that never complete by `max_len` are repaired.
"""
import numpy as np
import torch


@torch.no_grad()
def diffusion_sample(denoiser, grammar, batch, max_len, dyn_tokens,
                     device, temperature=1.0):
    """Returns a list of `batch` complete pre-order token-id lists."""
    n_choices = grammar.L
    grid = torch.full((batch, max_len), denoiser.mask_id, dtype=torch.long, device=device)
    seqs = [[] for _ in range(batch)]
    done = [False] * batch

    for pos in range(max_len):
        if all(done):
            break
        t = torch.full((batch,), 1.0 - pos / max_len, device=device)        # noise level ↓
        logits = denoiser(grid, t, dyn_tokens)[:, pos, :]                    # (B, n_choices)
        for b in range(batch):
            if done[b]:
                continue
            mask = grammar.valid_token_mask(seqs[b])[:n_choices].copy()      # real-token mask
            # Force completion within max_len: when remaining slots are tight, terminals only.
            if (max_len - pos) <= grammar.dangling(seqs[b]):
                term_only = np.zeros(n_choices, dtype=bool)
                term_only[grammar.terminal_tokens] = True
                mask &= term_only
            valid = torch.as_tensor(mask, dtype=torch.bool, device=device)
            lg = logits[b].masked_fill(~valid, float("-inf"))
            if torch.isinf(lg).all():                                       # model killed all valid -> uniform
                probs = valid.float()
            else:
                probs = torch.softmax(lg / temperature, dim=-1)
            tok = int(torch.multinomial(probs, 1))
            seqs[b].append(tok)
            grid[b, pos] = tok
            if grammar.is_complete(seqs[b]):
                done[b] = True

    return [s if grammar.is_complete(s) else grammar.repair(s) for s in seqs]


def _revealed_before(grid_row, pos, L):
    """Real tokens (< L) at positions [0, pos), in position order, skipping gaps."""
    out = []
    for j in range(pos):
        tok = int(grid_row[j])
        if tok < L:
            out.append(tok)
    return out


def _decode(grid_row, grammar):
    """Trim the gap-free prefix to its minimal complete tree; else repair."""
    slots, out = 1, []
    for tok in grammar.contiguous_prefix(grid_row):
        out.append(tok)
        slots += grammar.arity(tok) - 1
        if slots == 0:
            return out
    return grammar.repair([int(t) for t in grid_row])      # _real() drops gaps + tail


@torch.no_grad()
def diffusion_sample_random(denoiser, grammar, batch, max_len, dyn_tokens,
                            device, temperature=1.0, rng=None, order=None):
    """Phase-3 any-order reverse sampler (design doc §6-7).

    Reveals masked positions in a RANDOM order (matching the random-subset masking
    the denoiser is trained on), rather than the semi-autoregressive pre-order of
    `diffusion_sample`. Validity is the grammar's lightweight mask over the
    gap-skipping revealed context; completion uses the no-interior-MASK guard
    (`is_complete_grid`); sequences that never complete by `max_len` are repaired.

    `rng`   : np.random.Generator for the per-sample reveal permutation (testable).
    `order` : optional list of per-sample position orderings (overrides `rng`).
    """
    n_choices = grammar.L
    mask_id = denoiser.mask_id
    grid = torch.full((batch, max_len), mask_id, dtype=torch.long, device=device)
    done = [False] * batch

    if order is None:
        if rng is None:
            rng = np.random.default_rng()
        order = [list(rng.permutation(max_len)) for _ in range(batch)]
    ptr = [0] * batch

    for step in range(max_len):
        if all(done):
            break
        t = torch.full((batch,), 1.0 - step / max_len, device=device)
        logits = denoiser(grid, t, dyn_tokens)                              # (B,L,n_choices)
        for b in range(batch):
            if done[b]:
                continue
            pos = int(order[b][ptr[b]])
            ptr[b] += 1
            ctx = _revealed_before(grid[b].tolist(), pos, n_choices)
            mask = grammar.valid_token_mask(ctx)[:n_choices].copy()          # lightweight, drop PAD
            if not mask.any():                                               # ctx already complete -> PAD-only;
                mask[grammar.terminal_tokens] = True                         # denoiser can't emit PAD, trim at decode
            valid = torch.as_tensor(mask, dtype=torch.bool, device=device)
            lg = logits[b, pos].masked_fill(~valid, float("-inf"))
            if torch.isinf(lg).all():                                        # model killed all valid -> uniform
                probs = valid.float()
            else:
                probs = torch.softmax(lg / temperature, dim=-1)
            grid[b, pos] = int(torch.multinomial(probs, 1))
            if grammar.is_complete_grid(grid[b].tolist()):                   # §7 guard (no interior MASK)
                done[b] = True

    return [_decode(grid[b].tolist(), grammar) for b in range(batch)]
