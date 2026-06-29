"""Unit tests for the diffusion grammar (design doc §10). Torch-free: uses a fake library.

Token map:  0:'+'(2) 1:'*'(2) 2:'sin'(1) 3:'cos'(1) 4:'x1' 5:'x2' 6:'1' 7:'c'(const)
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.diffusion.grammar import Grammar


class FakeLib:
    L = 8
    arities = np.array([2, 2, 1, 1, 0, 0, 0, 0], dtype=np.int64)
    terminal_tokens = np.array([4, 5, 6, 7], dtype=np.int64)
    trig_tokens = np.array([2, 3], dtype=np.int64)
    const_token = 7
    EMPTY_PARENT = 0
    EMPTY_SIBLING = 0


def g(**kw):
    return Grammar(FakeLib(), max_len=16, **kw)


def test_is_complete():
    G = g()
    assert G.is_complete([4])              # x1
    assert G.is_complete([0, 4, 6])        # x1 + 1
    assert G.is_complete([2, 4])           # sin(x1)
    assert not G.is_complete([0, 4])       # + missing right child
    assert not G.is_complete([1, 0, 4])    # * + x1  (incomplete)


def test_early_stop_pad():
    G = g()
    padded = G.pad_after_complete([0, 4, 6, 1, 1, 1], length=8)   # trailing junk dropped
    assert padded[:3] == [0, 4, 6]
    assert all(p == G.PAD for p in padded[3:])
    assert len(padded) == 8


def test_repair_completes():
    G = g()
    fixed = G.repair([0, 4])               # '+ x1' -> complete with a terminal
    assert G.is_complete(fixed)
    assert fixed[:2] == [0, 4]


def test_valid_mask_pad_only_when_complete():
    G = g()
    m = G.valid_token_mask([0, 4, 6])      # complete -> only PAD
    assert m[G.PAD] and m[:G.L].sum() == 0
    m2 = G.valid_token_mask([0])           # incomplete -> PAD forbidden, real tokens ok
    assert not m2[G.PAD] and m2[:G.L].any()


def test_no_nested_trig():
    G = g(ban_nested_trig=True)
    m = G.valid_token_mask([2])            # inside sin(): trig children banned
    assert not m[2] and not m[3]
    assert m[4]                            # terminals fine


def test_max_const():
    G = g(max_const=1)
    m = G.valid_token_mask([0, 7])         # one const already used; next const banned
    assert not m[7]
    assert m[4]


def test_contiguous_prefix_stops_at_gap():
    G = g()
    # real tokens 0..7; anything >= L (PAD=8, mask_id=8, MASK=9) is a boundary
    assert G.contiguous_prefix([0, 4, 6, G.MASK, G.MASK]) == [0, 4, 6]
    assert G.contiguous_prefix([0, G.MASK, 4, 6]) == [0]          # stop at first gap
    assert G.contiguous_prefix([0, G.L, 4, 6]) == [0]            # mask_id == L is also a boundary
    assert G.contiguous_prefix([G.MASK, 4]) == []                # leading gap -> empty


def test_is_complete_grid_guard():
    G = g()
    # gappy grid: _real()-filtering would falsely see [0,4,6] (x1+1) as complete
    assert G.is_complete([0, G.MASK, 4, 6])             # OLD check: falsely True (documents the bug)
    assert not G.is_complete_grid([0, G.MASK, 4, 6])    # GUARD: interior gap -> not complete
    # genuinely complete contiguous prefix with masked tail -> complete
    assert G.is_complete_grid([0, 4, 6, G.MASK, G.MASK])
    # all-masked / leading gap -> not complete
    assert not G.is_complete_grid([G.MASK, G.MASK, G.MASK])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} grammar tests passed.")
