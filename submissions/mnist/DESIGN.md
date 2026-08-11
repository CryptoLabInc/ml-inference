# How the solution works

Design, parameters, results and security posture. For building and running, see
[BUILDING.md](BUILDING.md).

> **Security notice.** The ≥128-bit justification is **not finalised** for either scheme — see
> [§5](#5-security). Do not cite these parameters as reviewed.

---

## 1. Circuit and parameters

```
fc1 (128×484, +b1)  →  x²  →  fc2 (10×128, +b2)
```

A BN-folded 2-layer MLP, identical for both schemes. `x²` is the activation the network was
**trained** with, not a polynomial approximation of ReLU, so the circuit evaluates the model
exactly — the only error is CKKS noise (measured: max |decrypted − plaintext logit| = 0.109 on
logits spanning [−32, +14]). Weights are row-major CSV in [`weights/`](weights/); shapes,
normalization and provenance in [`weights/manifest.txt`](weights/manifest.txt). Folded plaintext
accuracy 97.96%.

Two circuits evaluate it, chosen by instance size alone (`mlp::usePcmm` in
[`include/mlp_params.hpp`](include/mlp_params.hpp)) — same model, same weights, different packing.

### Scheme A — Halevi–Shoup (sizes 0–1)

| | |
| --- | --- |
| Ring | N = 2^17, CI subring (ePrint 2018/952), `GRAFTED`, 65536 slots |
| Modulus chain | 30 + 3×25 ≈ 105 bits, 4 levels, no bootstrapping |
| Secret key | uniform ternary (hw = 0), sampled at 2^15, **lifted** to 2^17 |
| Noise / SWK budget | σ = 3.2 / `maxBits128(14) = 430` bits, margin 5.0 |

```
L3  encrypt → fc1 matvec              → L2
L2  fold, +b1, x², rescale            → L1
L1  fc2 matvec                        → L0, +b2, decrypt
```

**Packing.** One image occupies 512 padded coordinates; coordinate `c` of image `i` sits in slot
`c*128 + i` — **128 images per ciphertext**, so a batch of `n` needs `ceil(n/128)` ciphertexts, not
`n`. The harness only measures the byte size of `ciphertexts_upload/`, so the layout is a free
choice.

`fc1` is rectangular 128×512 and folds 4:1. `fc2` is deliberately *squared* to 128×128 rather than
the natural 16×128: its fold stride would miss the key-less fold's invariance period, and a keyed
fold costs a mod-down per coset — squared, the cosets instead ride the matvec's giant steps, which
double-hoisted BSGS accumulates behind a single mod-down.

**The key-less fold.** A key lifted from 2^15 is invariant under exactly those rotations whose step
is a multiple of 2^14, and fc1's fold stride (128 × 128 = 16384) is one. Its fold therefore runs as
bare Galois automorphisms (`HomEval::frobMap`) — no rotation keys, no level, no scale change, no
noise growth — in log2(4) = 2 automorphisms. The divisibility is asserted at key generation:
getting it wrong throws nothing, it silently decrypts to noise. `frobMap` is supplied by the
vendored HEaaN2 in [`install/`](install/).

### Scheme B — PCMM (sizes 2–3)

Feature = ciphertext **row** (the GEMM contraction dim), image = **column/slot** — the opposite
convention from HS. Written against HEaaN2's public API (`HomEvalMatrix::pcmm`,
`ICtMatrix`/`IPtMatrix`, `Matrix<Real>`); HEaaN2's own `mlp/PCMM` benchmark was read as a design
reference for the algorithm, but [`src/mlp_pcmm.cpp`](src/mlp_pcmm.cpp) is our own code.

| | |
| --- | --- |
| Ring | N = 2^13, CI subring, `GRAFTED`, 4096 coefficients/message |
| Modulus chain | 28 + 3×22 ≈ 94 bits, 4 levels, no bootstrapping |
| Secret key | uniform ternary (hw = 0), sampled **directly** at 2^13 — no lifting |
| Noise / SWK budget | σ = 3.2 / `maxBits128(12) = 106` bits, margin 5.0 |

```
L3  encrypt X, encode U1 → fc1 pcmm+rescale   → L2
L2  x²: tensor → rescale → L1, relin @L1
L1  encode U2 → fc2 pcmm+rescale              → L0
L0  +b2 (plaintext add, no level cost), decrypt
```

One message holds 4096 images; larger batches split into further *blocks* inside a single
`ICtMatrix`, transparently. `x²` runs tensor → rescale → relin (not the usual tensor → relin →
rescale), so the relin key is built at the *post-rescale* level.

**Bias fold.** `b1` rides as an extra column of `U1` (`128×485`) against an appended ones-row in the
packed input, so fc1's pcmm computes `W1·X + b1` directly. `b2` is *not* folded that way — it would
carry the ones-row through `x²` and `fc2`, taking the contraction from 128 to 129 rows for one bias
vector — so it is added afterwards as a plaintext delta (its inverse DFT is one nonzero coefficient
per block, which a zero-filled matrix already provides).

**Coefficient/slot relabeling** (ePrint 2024/1284 §6.3). `pcmm` wants weights coefficient-encoded
(literal GEMM scalars) but data slot-encoded (so the library's iDFT applies on encrypt). Both are
bit-for-bit the same array — `M' = MF`, and pcmm commutes with the per-row DFT — so the input is
encrypted once through the slot encoder and its DFT flag flipped in place via
`HomEvalFlexible::setDFT`, bridged through a `BatchRLWE` ciphertext (`ICtMatrix` exposes no
`setDFT`).

**Serialization.** `heaan::serial` has no `ICtMatrix` overload, so it crosses `io/` bridged through
a `BatchRLWE` `ICiphertext`, which *is* serializable. Shape is never stored in the file — both ends
recompute it from the batch size and the constants above, so writer and reader cannot drift.

### Encryption: public key vs symmetric key

An API constraint, not a design choice. HEaaN2's `EnDecryptor` has a public-key overload for plain
ciphertexts but **only a secret-key overload for matrices** (no `IEncKey` overload for
`ICtMatrix`). So HS encrypts under a public encryption key; **PCMM encrypts under the client's own
secret key.** Both are purely client-side and the secret key never leaves `seckeydir()` (which the
harness does not measure) — but it is a real asymmetry, disclosed here rather than left to be
noticed.

### Stage split

Every stage binary parses the size and dispatches; the harness contract (seven fixed names,
`<size>` as the only argument) is unchanged.

| Stage | HS (0–1) | PCMM (2–3) |
| --- | --- | --- |
| `client_key_generation` | sk (lifted), public enc key, 2× rotation keys, relin key | sk (unlifted), relin key only |
| `server_preprocess_model` | caches CSV weights in padded p×q layout | same call also caches raw CSV shapes |
| `client_preprocess_input` | normalize + center-crop — cleartext, identical both schemes ([§2](#2-cleartext-pre--and-post-processing)) | ← |
| `client_encode_encrypt_input` | 128 img/ciphertext, **public** key | whole batch → 1 `ICtMatrix`, **secret** key |
| `server_encrypted_compute` | the entire inference, on ciphertext | ← |
| `client_decrypt_decode` | read slot `r*128+i` | read column `(i/ringDim)*degree + i%ringDim` of row `r` |
| `client_postprocess` | argmax — cleartext, same output format both schemes | ← |

`server_preprocess_model` is invoked **with no arguments**, so it cannot know the instance size or
reach `io/<size>/public_keys`. HS's `MatrixVectorEval` encodes diagonals in its constructor on the
rotation keys' device; PCMM's `buildModel` encodes at batch-size-dependent shapes. Neither fits in
stage 3, so both happen in stage 7 — which therefore reports *setup*, *warm-up* and *evaluation*
separately in `server_reported_steps.json`. The reference submission has the same shape (its
stage 3 is a no-op).

**Warm-up.** Between setup and the timed evaluation, stage 7 runs one full inference pass and
discards the result — the same discarded pass HEaaN2's own mlp benchmarks run. The first use of
each CUDA kernel pays module loading, the first allocation grows the memory pool, and the NTT
workspace is built lazily; without the warm-up those one-time costs land inside the reported
evaluation, which then understates a warm server (and is not comparable to HEaaN2's benchmark
figures). Stated plainly: the harness times stage 7 as one process, so the warm-up moves nothing
out of the harness's own `Encrypted computation` — it *adds* roughly one warm evaluation
(tens of ms) to it, and that price is accepted for an honest evaluation figure.

**Device synchronization.** The stage-7 timers call `cudaDeviceSynchronize()` at both ends, as
HEaaN2's own mlp benchmarks do. Kernel launches are asynchronous, so an unsynchronized timer
closes while the GPU is still working and reports launch time rather than completion time — for
PCMM at 1000 images that under-reported the evaluation by ~4.7× (0.25 ms against a true 1.19 ms).
The cost was never lost, only misattributed: it resurfaced in whatever forced completion next
(the result serialization), which the harness still counted. Only the submission's own
`server_reported_steps.json` breakdown was affected, never the harness's `Encrypted computation`.

With both in place the submission's evaluation matches the library benchmark on identical
hardware, weights and batch — see [§4](#4-results).

---

## 2. Cleartext pre- and post-processing

Everything the **model** computes runs on ciphertext. The cleartext steps:

| Step | Side | Note |
| --- | --- | --- |
| `(p − 0.1307) / 0.3081` normalization | client, pre-encryption | Harness writes pixels already in [0,1]. Same as the reference submission and the harness's own model (`harness/mnist/test.py:60`) |
| **center-crop 28×28 → 22×22** | client, pre-encryption | see below |
| argmax over 10 logits | client, post-decryption | same as the reference's `client_postprocess` |
| BatchNorm folded into fc1 | offline, model-only | standard eval-mode folding at weight export; does not touch the input |

### The crop

The client trims 3 pixels per side before encrypting, so the encrypted input is 484-dimensional,
not 784.

**Why it is part of the model.** The network was *trained* on 22×22: `fc1` is literally a 128×484
matrix. There is no 784-input model to run, and no operation the model performs has moved out of
the encrypted stage.

**Its effect, stated plainly.** 484 pads to **512** coordinates instead of the 1024 that 784 would
need. At 65536 slots that is **128 images per ciphertext instead of 64** — a **2× throughput and
bandwidth advantage that follows directly from a cleartext step.** Every per-image timing and every
upload figure here carries that factor, and anyone comparing against a submission that encrypts all
784 pixels should keep it in mind. Removing the objection entirely would mean training a 784-input
variant.

---

## 3. Deviation from the harness model

| | |
| --- | --- |
| `harness/mnist/model.py` | `fc1 Linear(784,128) → ReLU → fc2 Linear(128,64) → ReLU → fc3 Linear(64,10)` |
| This submission | `fc1 Linear(484,128) → x² → fc2 Linear(128,10)` |

Two layers instead of three, `x²` instead of ReLU, 484-dim cropped input; trained separately
(x²+BatchNorm, 30 epochs, seed 0, BN folded at export). The reference OpenFHE submission also
deviates — 512×784 first layer with a polynomial approx-ReLU. Accuracy is in line with the harness
model ([§4](#4-results)).

---

## 4. Results

Seed 3, through the **unmodified harness**, on 1× RTX 5090 (sm_120). **Not official measurements**:
`measurements/` still holds the reference OpenFHE numbers. All figures come from a HEaaN2 verified
as natively sm_120 by the [architecture check](BUILDING.md#-this-submission-requires-an-sm_120-gpu)
— a library that falls back to JIT-compiling PTX reports several seconds of first-run cost as if it
were evaluation time, so that check is a precondition for quoting any timing here.

### As shipped

| | size 0 (1) HS | size 1 (100) HS | size 2 (1000) PCMM | size 3 (10000) PCMM |
| --- | --- | --- | --- | --- |
| harness `Encrypted computation` | 4.63 s | 4.65 s | **0.44 s** | **0.57 s** |
| ├─ model setup | 4.44 s | 4.46 s | 0.19–0.22 s | 0.20 s |
| └─ evaluation | 0.016 s | 0.016 s | 0.040–0.046 s | 0.042 s |
| Public + evaluation keys | 174.8 M | 174.8 M | **54.0 K** | **54.0 K** |
| Encrypted input | 1.6 M | 1.6 M | 44.5 M | 133.6 M |
| Encrypted results | 480 K | 480 K | 280 K | 840 K |
| **Accuracy** | PASS | **0.980** | **0.989** | **0.979** |
| Harness plaintext model | — | 0.970 | 0.982 | 0.978 |

Accuracy varies slightly run to run from encryption noise (size 3 measured 0.9794–0.9796 across
runs); the table rounds.

### Why PCMM at medium/large — a controlled A/B

Both schemes, same instances, same machine, same session, same sm_120 library (HS forced at sizes
2–3 by overriding `usePcmm`):

| | HS setup | HS eval | HS scored | PCMM setup | PCMM eval | PCMM scored |
| --- | --- | --- | --- | --- | --- | --- |
| size 2 (1000) | 4.34–4.39 s | **0.018 s** | 4.55–4.60 s | 0.19–0.22 s | 0.040–0.046 s | **0.44 s** |
| size 3 (10000) | 4.35 s | 0.044 s | 4.66 s | 0.20 s | **0.042 s** | **0.57 s** |

**PCMM wins on setup, not arithmetic.** It needs no rotation keys at all (54 K of key material
against HS's 174.8 M), so it skips the ~4.4 s of key deserialization and diagonal encoding HS
re-pays *every run*. Since the harness times the whole stage-7 process, that is what gets scored,
and PCMM lands 8–10× ahead.

On pure evaluation the two cross over right about where the split is placed: HS is ~2.4× faster at
1000 images, and they are level at 10000. Two honest qualifications:

- **If HS's setup were amortized** (a long-running server loading keys once), HS would be the
  faster choice at size 2. The split is right *for this benchmark's cost model*, which re-pays
  setup every run — not a general claim about the algorithms.
- **PCMM's encrypted input is larger**, not smaller (44.5 M vs 13.1 M at size 2). It wins on compute
  and key material, loses on upload bandwidth.

### Cross-check against HEaaN2's own mlp benchmarks

The library ships its own benchmarks for both circuits (`mlp/MLInference_128`,
`mlp/MLInference_Large`). Run on **1× RTX 4090 (sm_89)** — a development box, not the bench
machine — against the same sm_89 library build, this submission's weights and seed 3, the
submission's warm, synchronized evaluation lands on the library's own figure:

| | HEaaN2 benchmark | This submission (stage 7) |
| --- | --- | --- |
| HS, one ciphertext | 1.434 ± 0.044 ms (20 runs) | 1.439 ms |
| PCMM, 1000 images | 1.378 ± 0.105 ms (20 runs) | 1.187 ms |

The submission is marginally *faster* on PCMM only because it brackets the whole inference in one
timer, where the benchmark sums six separately-synchronized sub-timers. Treat the two as equal:
the submission adds no evaluation overhead over calling the library directly.

**What does not transfer is setup.** The benchmark generates keys and encodes the model once, in
process, and reports that as untimed "offline" work; it then measures many evaluations against it.
The submission cannot: the harness runs stage 7 as a fresh process per run, so it re-pays key
*deserialization from disk* (174.8 MB of rotation keys for HS) and diagonal encoding every time,
and the harness scores that. The gap between this submission's scored figure and the library's
headline number is that structural difference, not an implementation difference.

### Against the reference OpenFHE submission

| | size 0 | size 1 | size 2 |
| --- | --- | --- | --- |
| reference `Encrypted computation` | 7.61 s | 647.07 s | 6505.61 s |
| reference keys / input | 1.0 G / 5.0 M | 1.0 G / 500.3 M | 1.0 G / 4.9 G |
| reference accuracy | — | 0.96 | 0.974 |

Before quoting any ratio:

1. **Setup dominates HS**, so sizes 0–1 are setup-bound: 4.63 s scored against 0.016 s of actual
   evaluation. For single-shot latency, 4.63 s is the honest number — not 16 ms.
2. **The table above predates stage 7's warm-up pass and its timer synchronization.** Its
   evaluation figures are both cold (absorbing first-use CUDA costs a long-running server would
   amortize) and unsynchronized (closing before the GPU finished). Both are fixed now, and the
   official measurements will be re-taken with them in place.
3. **Reference numbers are CPU; ours are GPU.** Not the same hardware.

---

## 5. Security

**The ≥128-bit claim is not signed off for either scheme.**

The shared uniform-ternary (hw = 0) budget table is `maxBits128` in
[`include/mlp_params.hpp`](include/mlp_params.hpp): entries for 2^13–2^15 from HEaven's
`maxBitsPolicy128()`, 2^16–2^17 citing ePrint 2024/463, and a 2^12 entry added for PCMM. **These
are provisional and the Hamming weight may change.**

- **HS.** The switching-key budget is sized from the degree the key was *sampled* at, not the ring
  it was lifted into: lifting adds no entropy, so a key lifted from 2^15 carries the LWE problem of
  2^15, and CI halves it again (only half the coefficients are sampled) — effective dimension 2^14,
  budget 430 bits, against a ~105-bit chain. **The lifted-key construction specifically has not
  been reviewed.**
- **PCMM.** No lifting: sampled directly at 2^13, so security rests on `maxBits128(12) = 106` bits
  (CI → RLWE dimension 2^12) against a ~94-bit chain. Simpler to justify — no lifting argument.

Everything a review would change is confined to one block each in `mlp::` and `mlp::pcmm`.

### Why the Zn-multiplication citation does not transfer

CryptoLab's [Zn-multiplication submission](https://github.com/CryptoLabInc/Zn-multiplication) closes
its security section by citing
[sparse-key-estimate](https://github.com/jdumezy/sparse-key-estimate/blob/master/Precomputed-Tables/128bits_security.md),
and its numbers line up exactly (N = 2^16, hw = 32, log(PQ) = 114 ≤ that table's 349).

**That citation cannot be reused here.** The table is indexed by *sparse* Hamming weights — columns
h ∈ {32, 64, 128, 192, 256, 512, 1024}. Both our schemes use `hw = 0`, i.e. a **uniform-ternary
(dense)** secret with h ≈ 2n/3, off the right edge of the table. There is no row to read.

Our values sit near its densest column without matching it (our 2^13 entry 214 equals its `h=1024`
exactly; 2^14 is 430 vs 426; 2^15 is 868 vs 854), consistent with their stated provenance in
`maxBitsPolicy128()` rather than in this table. A dense key *is* at least as hard as an h=1024 one
at the same (n, q), so reading off that column would be conservative — but that is an argument a
reviewer accepts, not a citation, and the numeric drift shows they are different analyses.

Two coherent ways to close it — a decision, not an oversight:

1. **Keep `hw = 0`** and cite something valid for a dense key: a lattice-estimator run at our exact
   (n, q, σ) points, or the HE standard. This is what the pending review needs to supply.
2. **Switch to a sparse key** (hw = 32) so the same public table applies directly. A real parameter
   change, not a docs edit: it alters noise growth, so the level schedule, accuracy and
   bottom-modulus headroom all need re-validating.

Recorded so the gap is not mistaken for "the sibling submission already solved this."
