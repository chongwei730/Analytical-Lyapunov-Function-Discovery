"""Encoder-only masked-diffusion denoiser (design doc §5).

A single bidirectional transformer over [dynamics prefix | expression tokens]:
  * dynamics prefix : clean (never masked) conditioning token ids
  * expression body : token ids in 0..n_choices-1, or `mask_id` (= n_choices)
  * timestep t      : scalar noise level, conditioned via an additive embedding
Predicts clean-token logits (size n_choices) at every expression position at once.
"""
import torch
import torch.nn as nn


class MaskedDiffusionDenoiser(nn.Module):
    def __init__(self, n_choices, max_length, dyn_vocab_size,
                 d_model=128, nhead=4, nlayers=4, dim_ff=512, dropout=0.0):
        super().__init__()
        self.n_choices = n_choices
        self.mask_id = n_choices                      # extra input id for [MASK]
        self.max_length = max_length
        self.tok_emb = nn.Embedding(n_choices + 1, d_model)     # real tokens + MASK
        self.dyn_emb = nn.Embedding(dyn_vocab_size, d_model)    # dynamics prefix
        self.pos_emb = nn.Embedding(max_length, d_model)
        self.seg_emb = nn.Embedding(2, d_model)                 # 0=prefix, 1=expression
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.head = nn.Linear(d_model, n_choices)

    def forward(self, expr_tokens, t, dyn_tokens):
        """expr_tokens (B,L) long; t (B,) float in (0,1]; dyn_tokens (B,P) long -> logits (B,L,n_choices)."""
        B, L = expr_tokens.shape
        P = dyn_tokens.shape[1]
        dev = expr_tokens.device
        tcond = self.time_mlp(t.view(B, 1).float()).unsqueeze(1)             # (B,1,d)
        dyn = self.dyn_emb(dyn_tokens) \
            + self.seg_emb(torch.zeros(B, P, dtype=torch.long, device=dev)) + tcond
        pos = self.pos_emb(torch.arange(L, device=dev)).unsqueeze(0)
        expr = self.tok_emb(expr_tokens) + pos \
            + self.seg_emb(torch.ones(B, L, dtype=torch.long, device=dev)) + tcond
        h = self.encoder(torch.cat([dyn, expr], dim=1))                      # (B, P+L, d)
        return self.head(h[:, P:, :])                                        # (B, L, n_choices)
