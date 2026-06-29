"""Unit tests for the Phase-3 random-order diffusion sampler (design doc §6-7).

Torch-based but CPU-only: a hand-built dummy denoiser supplies a fixed token
distribution so the sampler's reveal-order / completion-guard / repair logic is
exercised deterministically (design doc §10 "dummy-denoiser integration").

Token map (FakeLib):  0:'+'(2) 1:'*'(2) 2:'sin'(1) 3:'cos'(1) 4:'x1' 5:'x2' 6:'1' 7:'c'
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.diffusion.grammar import Grammar
from models.diffusion.sampler import diffusion_sample_random


class FakeLib:
    L = 8
    arities = np.array([2, 2, 1, 1, 0, 0, 0, 0], dtype=np.int64)
    terminal_tokens = np.array([4, 5, 6, 7], dtype=np.int64)
    trig_tokens = np.array([2, 3], dtype=np.int64)
    const_token = 7


class TreeDenoiser:
    """Prefers '+' at position 0 and 'x1' everywhere else -> builds (x1 + x1)."""
    def __init__(self, n_choices):
        self.n_choices = n_choices
        self.mask_id = n_choices

    def __call__(self, grid, t, dyn):
        B, L = grid.shape
        logits = torch.zeros(B, L, self.n_choices)
        logits[:, 0, 0] = 20.0      # '+'
        logits[:, 1:, 4] = 20.0     # 'x1'
        return logits


def G():
    return Grammar(FakeLib(), max_len=16)


def _is_tight_complete(grammar, seq):
    return grammar.is_complete(seq) and (len(seq) == 1 or not grammar.is_complete(seq[:-1]))


def test_random_sampler_returns_only_valid_complete():
    grammar, den = G(), TreeDenoiser(8)
    dyn = torch.zeros((5, 1), dtype=torch.long)
    seqs = diffusion_sample_random(den, grammar, 5, 16, dyn, device="cpu",
                                   rng=np.random.default_rng(0))
    assert len(seqs) == 5
    for s in seqs:
        assert _is_tight_complete(grammar, s), s        # complete and minimal (no trailing junk)


def test_random_sampler_builds_expected_tree():
    grammar, den = G(), TreeDenoiser(8)
    dyn = torch.zeros((4, 1), dtype=torch.long)
    seqs = diffusion_sample_random(den, grammar, 4, 16, dyn, device="cpu",
                                   rng=np.random.default_rng(1))
    for s in seqs:
        assert s == [0, 4, 4], s                        # (x1 + x1)


def test_guard_blocks_false_completion_out_of_order():
    """Reveal order [2,1,0]: positions 1,2 ('x1','x1') land before position 0.
    A naive `_real` check would see [x1] and stop; the §7 guard must not."""
    grammar, den = G(), TreeDenoiser(8)
    dyn = torch.zeros((1, 1), dtype=torch.long)
    seqs = diffusion_sample_random(den, grammar, 1, 16, dyn, device="cpu",
                                   order=[[2, 1, 0] + list(range(3, 16))])
    assert seqs[0] == [0, 4, 4], seqs[0]                 # NOT falsely truncated to [4]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} sampler tests passed.")
