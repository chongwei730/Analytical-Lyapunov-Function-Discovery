# Design: Discrete-Diffusion Expression Generator (config-selectable, for A/B vs the transformer)

Status: draft for review
Date: 2026-06-29
Owner: (project author)

## 1. Goal & success criteria

Add a **discrete (masked) diffusion** generator to this RL symbolic-regression framework as a
**config-selectable alternative** to the existing autoregressive transformer
(`TransformerTreeEncoderController`, `models/transformers2.py`). The purpose is a **clean A/B study**:
does a diffusion generator discover analytical Lyapunov functions better than the autoregressive
transformer, on the existing 13 benchmark systems?

**Success = a controlled comparison** reporting, per system, **success rate** (valid Lyapunov function
found / 5 seeds), **wall-clock to success**, and **expression quality** (reward `r`, complexity), with
the diffusion arm sharing the reward, benchmark, verification, and training loop with the transformer arm.

### Reference papers
- **Architecture** — MDLM, "Simple and Effective Masked Diffusion Language Models" (arXiv:2406.07524):
  encoder-only masked/absorbing diffusion; Rao-Blackwellized weighted masked-cross-entropy objective.
- **RL algorithm** — RL-D², "RL with Discrete Diffusion Policies for Combinatorial Action Spaces"
  (arXiv:2509.22963): fit the diffusion policy to a Policy-Mirror-Descent target via distributional
  matching; Forward-KL (advantage-weighted ELBO) and Reverse-KL (clipped per-step ratio) variants.
- **Tricks** — "Diffusion-Based Symbolic Regression" (arXiv:2505.24776): mask-diffusion over expression
  tokens, token-wise GRPO, Long-Short-Term risk-seeking buffer, structural-completion termination + repair.

## 2. Non-goals
- Not replacing the transformer (it stays, untouched, as the baseline arm).
- Not changing the reward, Lyapunov verification (SHGO / dReal), GP-meld, or counterexample feedback.
- Not building a new token library or a new search space — the diffusion arm uses the **same per-system
  `Library`** as the transformer arm (fairness requirement).

## 3. Background: the existing pipeline (what stays fixed)

Training loop (`utils/train.py`), 11 steps per epoch, **only steps 1 and 8 touch the generator**:

1. `controller.sample(B)` → candidate token sequences
2. decode tokens → expression trees → sympy `V(x)`
3. optimize constants (scipy)
4. reward `R = 1/(1+L)`, `L` = Lyapunov-risk over training set `X`
5. GP-meld refinement (deap)
6. risk-seeking baseline `b = top-α% reward quantile`; advantage `A = R − b`
7. build `Batch` (`actions, obs, priors, lengths, rewards, …`; `memory.py:9`)
8. policy-gradient loss on the controller; backprop; `optimizer.step()`
9. verify best candidate (SHGO); counterexamples → `X`
10–11. update priority queue / logging

The **controller contract** the rest of the code depends on (must be preserved):
- `sample(n, input_) -> (actions, obs, priors)` numpy, shapes `(B,L)`, `(B,4,L)`, `(B,L,n_choices)`
- `train_loss(b, sampled_batch, pqt_batch, test) -> (loss, mle_loss)` (differentiable)
- attributes: `task`, `prior`, `max_length`, `tgt_padding_token`, `pqt`, `pqt_k`, `pqt_batch_size`,
  `rl_weight`, `learning_rate`, `parameters()`, `state_dict()`, `train()`.

## 4. Locked design decisions

| # | Decision |
|---|---|
| D1 | New `DiffusionController` selected via a config flag; transformer arm untouched. |
| D2 | Denoiser is a **single encoder-only** (bidirectional) transformer — no decoder, no cross-attention (faithful MDLM). |
| D3 | **Dynamics conditioning = clean prefix** in the same sequence (in-context); the existing dynamics-embedding stack may be reused to build the prefix. |
| D4 | **Tree structure kept** as **additive per-token structural embeddings** (parent-type / sibling / depth), computed from revealed tokens; plus a **timestep embedding**. |
| D5 | **Traversal = pre-order DFS (baseline)**; BFS available as an **ablation flag**. |
| D6 | **Validity = in-loop masking + repair fallback + reward-0** for un-repairable (Procedure "B"; in-loop mask = full pre-order `JointPrior` in Phase-1 in-order reconstruction, lightweight rules in random-order Phase-3). Early-stop on **structural completion (`dangling==0`), no EOS**, PAD after. |
| D7 | **RL objective = RL-D² Forward-KL (advantage-weighted ELBO) first**, with **Reverse-KL / token-wise GRPO as a drop-in Phase-3 objective** behind a config flag. Reuse the risk-seeking quantile baseline; extend the priority queue into the **Long-Short-Term buffer**. |
| D8 | Build the generation mechanism as a **standalone, test-first module**, then wrap it in `DiffusionController`. |

### Honest caveats (recorded for the writeup)
- **BFS ablation** changes serialization vs the transformer arm → a coupled variable; the **token library
  and expression space are identical**, so the comparison remains valid, but BFS results are reported as an
  ablation, not the controlled baseline.
- The diffusion likelihood is **intractable**; we use the **ELBO as the surrogate** "neg-log-prob" in the
  policy update (standard in RL-D²/DDPO-style fine-tuning).
- **Phase-1 reconstruction is in traversal order** (semi-autoregressive). The fully any-order ("pure
  diffusion") sampler is Phase 3; it requires the no-interior-MASK stop guard (see §7).

## 5. Architecture

A single bidirectional transformer denoiser over one sequence:

```
input to ONE encoder-only transformer:
  [ dynamics prefix (clean, never masked) | expression tokens (masked ⇄ unmasked) | PAD tail ]
  + timestep embedding t                         (added to all positions)
  + structural embeddings (parent/sibling/depth) (added to expression positions)
        │
        ▼  (self-attention, no causal mask)
  per-position logits over the token vocabulary  → read at masked expression positions
```

- **Dynamics prefix (D3):** reuse the existing dynamics embedding (`dym_encoder` + `dym_embedded_encoder`,
  `transformers2.py:128/158`) to produce prefix token embeddings, or feed raw dynamics-token embeddings
  directly (purest MDLM). Prefix positions are always clean and never masked.
- **Denoiser:** standard pre-norm transformer encoder blocks; predicts the clean token at each masked
  expression position simultaneously.
- **Timestep / structural conditioning (D4):** additive embeddings; timestep encodes the noise level
  (fraction masked); structural embeddings inject `(parent-type, sibling, depth)` derived from revealed tokens.

## 6. Components (isolated units, clear interfaces)

1. **`serialization`** — pre-order tokens ↔ tree, completion/`dangling`. Reuse repo `Library` / `Program`
   / `subroutines` for pre-order; add `bfs_tokens_to_tree` only for the BFS ablation.
   - `is_complete(tokens, library) -> bool` (= repo `dangling==0`).
2. **`validity_repair`** — `valid_token_mask(partial, library, phase)` (Phase-1 reuses pre-order
   `JointPrior`; Phase-3 random-order uses lightweight mask: no MASK, PAD-gating, ≤max-const, no nested
   trig); `repair_invalid_expression(tokens) -> tokens`; `fallback_expression()` (a single terminal).
3. **`denoiser`** (`nn.Module`) — the encoder-only transformer above. `forward(seq, t, dynamics_prefix) -> logits (B,L,V)`.
4. **`sampler`** — the reverse-diffusion loop: start all-MASK → choose next masked position (Phase-1: pre-order
   in-order; Phase-3: random) → apply `valid_token_mask` → sample → preserve revealed → **early-stop on
   `is_complete`** → PAD tail → repair fallback at `MAX_LEN`. Returns token sequences (+ per-step info).
5. **`DiffusionController`** (`nn.Module`) — wraps denoiser + sampler; implements the **contract** (§3).
   - `sample()` → decode to actions; compute `obs`/`priors` for the `Batch` via the existing pre-order
     `parents_siblings` / `JointPrior.at_once` (valid because baseline is pre-order).
   - `train_loss()` → the diffusion ELBO + FKL weighting (§8).
6. **`diffusion_loss`** — `elbo(seq, denoiser)` (weighted masked-CE); `fkl_loss(elbo, advantage, λ)`
   (`−softmax(A/λ)·ELBO`); Phase-3 `rkl_grpo_loss(...)` drop-in.
7. **`lst_buffer`** — extend the existing priority queue: persist best expressions across epochs, prune
   bottom α% (Long-Short-Term risk-seeking).
8. **`config wiring`** — `model: DiffusionController` selector + diffusion hyperparameters
   (`num_diffusion_steps`, `max_len`, `λ`, traversal, sampler-order, objective).

## 7. Validity, early-stop, repair (Procedure B)

- **Completion / early-stop:** maintain `dangling` (start 1; `dangling += arity[tok] − 1`); a sample is
  **done iff `dangling==0`** on the contiguous revealed prefix. No EOS token; PAD fills the tail. Proven
  exact for valid pre-order (and BFS) traversals — `dangling==0` only at full completion, never on a proper
  prefix.
- **In-loop masking:** Phase 1 (in-order, gap-free) applies the existing pre-order `JointPrior` at the
  revealed position (subsumes the lightweight rules and matches the transformer arm's constraints exactly).
  If all valid tokens have zero model prob → uniform over valid tokens.
- **Random-order guard (Phase 3):** declare complete only when there are **no interior MASKs** in the
  active region (the naive skip-MASK count can falsely complete across gaps).
- **Repair fallback:** if `MAX_LEN` reached without completion, repair (resample invalid tokens until
  valid, per 2505.24776); if still invalid, use `fallback_expression()` and let **reward = 0** (the
  pipeline already penalizes invalid/incomplete expressions).

## 8. RL training (step 8)

- **Advantage:** reuse the risk-seeking quantile `b` (`train.py:650`) → `A = R − b`.
- **ELBO (surrogate neg-log-prob):** weighted masked-cross-entropy of the denoiser reconstructing the
  candidate from masked versions (MDLM objective).
- **Phase-2 objective — FKL (RL-D²):** `loss = − E[ softmax(A/λ) · ELBO ] (+ entropy)`. Maps onto the
  existing `(R−b)·neglogp` shape with `neglogp → ELBO` and quantile weighting → softmax weighting.
- **Phase-3 objective — RKL / token-wise GRPO (drop-in):** PPO-style clipped per-denoising-step (or
  per-token) ratio vs a frozen reference net + KL reg, group-relative advantages.
- **LST buffer:** persistent best-expression pool feeds high-advantage targets to the FKL update.

## 9. Error handling
- Invalid / un-repairable expressions → `fallback_expression()` + reward 0 (no crash).
- Constant-optimization failure → skip optimization, keep raw expression.
- NaN/Inf in ELBO or logits → guarded (clamp / skip sample), mirroring existing `assert not isnan` checks.
- `MAX_LEN` cap bounds the sampler loop; denoiser step count `num_diffusion_steps` bounds compute.

## 10. Testing
- **Unit (from the user's plan):** `is_complete` (valid/invalid cases), early-stop (forced `[+,x1,1]` →
  stop + PAD), repair (`[+,x1,<PAD>]` → completed), constant-opt (`c*x1` on `y=3x1` → `3.0*x1`).
- **Serialization round-trip:** tree → tokens → tree identity (pre-order; and BFS for the ablation).
- **Dummy-denoiser integration:** sampler with a handcrafted distribution yields only valid expressions.
- **Pipeline smoke test:** real denoiser on **pendulum** with reduced `n_samples` (reuse the existing GPU
  smoke-test pattern) → produces a `[TEST RESULT]` end-to-end.
- **A/B harness:** reuse `repro/*_anvil` scripts; add a diffusion config; compare to the transformer sweep.

## 10b. Implementation status — Phase-3 random-order sampler (added 2026-06-29)

**Status: implemented (M5, partial).** The any-order ("pure diffusion") reverse sampler is built
and config-selectable; it complements the pre-order semi-AR baseline rather than replacing it.

- **Why it matters:** the denoiser is *trained* on random-subset masking (`DiffusionController._denoise_ce`
  masks each position i.i.d. with prob `t`), so an any-order reverse sampler matches the training
  distribution; the Phase-1 pre-order sampler is semi-autoregressive and only exercises one reveal order.
- **Completion guard (§7) — `Grammar.is_complete_grid` / `Grammar.contiguous_prefix`:** the old
  `is_complete` filters MASK/PAD via `_real()`, so a gappy grid (`[+, MASK, x1, 1]`) *falsely* completes.
  The guard scans the **contiguous, gap-free prefix from position 0** (any id `>= L` — PAD, MASK, or the
  denoiser mask sentinel — is a boundary) and only declares completion there. No-interior-MASK ⇒ no false stop.
- **Sampler — `sampler.diffusion_sample_random`:** start all-MASK → at each step pick the next position
  from a per-sample **random permutation** (`rng`, or an explicit `order` for tests) → lightweight
  `valid_token_mask` over the gap-skipping revealed context → masked-softmax sample → freeze (carry-over)
  → early-stop on `is_complete_grid` → repair fallback at `max_len`. PAD is never emitted by the denoiser
  (head size `L`); when the revealed context is already complete the mask falls back to terminals and the
  stray token is trimmed at `_decode` (minimal-complete-prefix, else `grammar.repair`).
- **Controller wiring:** `DiffusionController(sampler_order="preorder"|"random")` (config/`**kwargs`),
  routes `sample()` to the chosen sampler; transformer arm and training loss untouched.
- **Tests (TDD):** `tests/test_diffusion_grammar.py` (+2: contiguous-prefix, completion guard),
  `tests/test_diffusion_sampler.py` (3: valid-complete, expected-tree, out-of-order guard),
  `tests/test_diffusion_controller_order.py` (2: routing + validity). All green.
- **Known limitations:** (a) the lightweight mask over a gappy context is best-effort — structural
  constraints (nested-trig, dangling) are exact only once the relevant ancestors are revealed, so
  random-order relies more on `repair`; (b) no tight-completion terminal pressure in random order yet
  (pre-order has it), so completion rates lean on repair; (c) sampler still recomputes a full-batch
  denoiser forward per step reading one position/sample (vectorization is future work). Validate
  end-to-end speed/quality on a GPU node, not the CPU login node.

## 11. Milestones (build order)
- **M1 — generation module (standalone, test-first):** library/arity, pre-order ser/de, `is_complete`,
  valid-token mask, sampler with early-stop, repair, constant-opt, dummy denoiser. All unit tests green.
- **M2 — denoiser + controller:** encoder-only denoiser, dynamics prefix, structural/timestep embeddings,
  `DiffusionController` satisfying the contract; `sample()` returns valid `Batch` fields.
- **M3 — FKL training + integration:** ELBO + FKL loss; wire into `train.py`; pendulum smoke test passes.
- **M4 — A/B:** run the diffusion arm across the 13 systems; compare to the transformer reproduction.
- **M5 — extensions (optional):** ~~random-order sampler (+guard)~~ **DONE (see §10b)**;
  RKL/token-GRPO objective, BFS ablation remain.

## 12. Open items to confirm
- **D7 (RL objective):** FKL-first is proposed; confirm before M3 (RKL/GRPO remains a documented Phase-3
  add-on either way).
