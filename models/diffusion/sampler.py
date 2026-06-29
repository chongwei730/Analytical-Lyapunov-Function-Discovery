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
