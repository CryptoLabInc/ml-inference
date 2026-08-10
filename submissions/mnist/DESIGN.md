# How the solution works

Design, parameters, results and security posture of the HEaaN2 MNIST submission. For building and
running it, see [BUILDING.md](BUILDING.md); for what it is, [README.md](README.md).

> **Security notice.** The ≥128-bit parameter justification for **both** schemes' configurations is
> **not finalised** — see [§5](#5-security). Do not cite this submission's parameters as reviewed.

---

## 1. Circuit and parameters

### Circuit

```
fc1 (128x484, +b1)  ->  x^2  ->  fc2 (10x128, +b2)
```

A BN-folded 2-layer MLP, identical for both schemes. `x^2` is the activation the network was
**trained** with, not a polynomial approximation of ReLU, so the homomorphic circuit evaluates the
model exactly — the only error is CKKS noise. Measured on the harness's size-0 input (HS): max
|decrypted − plaintext logit| = 0.109 on logits spanning [−32, +14].

Weights are in [`weights/`](weights/) as row-major CSV, with shapes, normalization and provenance
in [`weights/manifest.txt`](weights/manifest.txt). Folded plaintext accuracy 97.96%.

Two homomorphic circuits evaluate it, chosen purely by instance size (`mlp::usePcmm` in
[`include/mlp_params.hpp`](include/mlp_params.hpp)) — same model, same weights, same accuracy
target, different packing.

### Scheme A — Halevi–Shoup (single, small)

| | |
| --- | --- |
| Scheme | CKKS on the **conjugate-invariant subring** (ePrint 2018/952) |
| Ring | N = 2^17, `PolyType::GRAFTED`, `NTTAlgorithm::CYC_FOR_CI` |
| Slots | 65536 |
| Modulus chain | base 30 bits + 3 × 25 bits ≈ 105 bits, 4 levels, **no bootstrapping** |
| Secret key | uniform ternary (hw = 0), sampled at 2^15 and lifted to 2^17 |
| Noise | discrete Gaussian, σ = 3.2 |
| Switching-key budget | `maxBits128(14) = 430` bits, margin 5.0 |

Level schedule, one level per operation:

```
L3  encrypt -> fc1 matvec ---------------> L2
L2  fold, +b1, x^2, rescale -------------> L1
L1  fc2 matvec --------------------------> L0, +b2, decrypt
```

**Packing.** One image occupies 512 padded coordinates; coordinate `c` of image `i` sits in slot
`c*128 + i`. That puts **128 images in a single ciphertext**, so a batch of `n` needs `ceil(n/128)`
ciphertexts rather than `n`. The harness measures only the byte size of `io/<size>/ciphertexts_upload/`,
so this layout is a free choice — and it is where the bandwidth difference against the reference
comes from.

`fc1` is the rectangular 128×512 shape and folds 4:1. `fc2` is deliberately *squared* to 128×128
rather than the natural 16×128: its fold stride would miss the key-less fold's invariance period,
and a keyed fold costs one mod-down per coset, so instead the cosets ride the matvec's giant steps
where the double-hoisted BSGS accumulates them behind a single mod-down.

#### The key-less fold

The secret key is sampled at 2^15 and lifted to 2^17 with `SKGenerator::genHighDegreeKey`. Such a
key is invariant under exactly those rotations whose step is a multiple of 2^14, and fc1's fold
stride (128 × 128 = 16384) is one. Its fold therefore runs as bare Galois automorphisms
(`HomEval::frobMap`) — no rotation keys, no level, no scale change, no noise growth — in
log2(4) = 2 automorphisms. The divisibility is asserted at key generation, because getting it wrong
throws nothing: it would silently decrypt to noise.

`HomEval::frobMap` is why this submission pins a specific HEaaN2 branch — see
[the branch requirement](BUILDING.md#the-heaan2-branch).

### Scheme B — PCMM (medium, large)

Feature = ciphertext **row** (pcmm's GEMM contraction dimension), image = **column/slot** — the
opposite packing convention from the HS scheme. HS's fixed 128-images-per-ciphertext packing makes
its per-image cost flat; PCMM's per-image cost keeps falling as the batch grows, which is why it
takes over at medium/large. Written independently against HEaaN2's public API
(`HomEvalMatrix::pcmm`, `ICtMatrix`/`IPtMatrix`, `Matrix<Real>`) — no HEaaN2 source vendored.
HEaaN2's own `mlp/PCMM` benchmark was read as a design reference for the algorithm (the pcmm
circuit for this model, the coefficient/slot relabeling trick below, the bias-folded-into-a-weight-
column layout); the code in [`src/mlp_pcmm.cpp`](src/mlp_pcmm.cpp) is our own.

| | |
| --- | --- |
| Scheme | CKKS on the **conjugate-invariant subring**, same as HS |
| Ring | N = 2^13, `PolyType::GRAFTED`, `NTTAlgorithm::CYC_FOR_CI` |
| Coefficients/message | 4096 (CI: N/2) — one message holds 4096 images; a bigger batch splits into further "blocks" inside one `ICtMatrix`, transparently |
| Modulus chain | base 28 bits + 3 × 22 bits ≈ 94 bits, 4 levels, **no bootstrapping** |
| Secret key | uniform ternary (hw = 0), sampled **directly** at 2^13 — no lifting; PCMM has no key-less fold to buy with one |
| Noise | discrete Gaussian, σ = 3.2 |
| Switching-key budget | `maxBits128(12) = 106` bits, margin 5.0 |

Level schedule (`SQUARE_OUT_DROP` etc. in [`include/mlp_params.hpp`](include/mlp_params.hpp)):

```
L3  encrypt X, encode U1 -> fc1 pcmm+rescale       -> L2
L2  x^2: tensor (still L2) -> rescale -----------> L1, relin @L1
L1  encode U2 -> fc2 pcmm+rescale                  -> L0
L0  +b2 (plaintext add, no level cost), decrypt
```
`x^2` runs tensor → rescale → relin rather than the more usual tensor → relin → rescale, so the
relin key is built at the *post-rescale* level (L1), not the input level (L2).

**Bias fold.** `b1` rides along as an extra column of `U1` (making it `[128 x 485]`) against an
appended "ones" row in the packed input, so `fc1`'s pcmm computes `W1·X + b1` directly — no
separate plaintext add. `b2` is *not* folded the same way: that would carry the ones-row through
`x^2` and `fc2`, taking the contraction from 128 to 129 rows for one bias vector. Instead `b2` is
added as a plaintext delta after fc2 (its inverse DFT is a single nonzero coefficient per block,
which a zero-filled matrix already provides).

**The section-6.3 relabeling trick** (ePrint 2024/1284 §6.3). `pcmm` needs its weights
coefficient-encoded (used as literal GEMM scalars) but its data slot-encoded (so the library's own
iDFT applies on encrypt). Both are, bit-for-bit, the same coefficient array — `M' = MF` for the
DFT matrix `F`, and `pcmm` (a linear combination of rows) commutes with the per-row DFT — so rather
than encoding twice, the input is encrypted once through the slot encoder and its DFT flag is
flipped in place afterward via `HomEvalFlexible::setDFT`, bridged through a `BatchRLWE`
`ICiphertext` (`ICtMatrix` itself exposes no `setDFT`). See `mlp::pcmm::setDFT` in
[`src/mlp_pcmm.cpp`](src/mlp_pcmm.cpp).

**Serialization.** HEaaN2's public `heaan::serial` has no `ICtMatrix`/`IPtMatrix` overload, so an
`ICtMatrix` crosses `io/` the same way: bridged through a `BatchRLWE` `ICiphertext` via
`moveTo`/`moveFrom`, which *is* serializable. Shape (`rows`, `cols`) is never stored in the file —
both ends recompute it from the instance's batch size and the constants above, so nothing can drift
between writer and reader.

### Encryption: public key vs symmetric key

The two schemes encrypt differently, and this is a real API constraint, not a design choice worth
hiding. HEaaN2's public `EnDecryptor` exposes a public-encryption-key overload for plain
ciphertexts (`encrypt(ptxt, IEncKey, ctxt)`) but **only a secret-key overload for matrices**
(`encrypt(IPtMatrix, ISecretKey, ICtMatrix)` — no `IEncKey` overload exists for `ICtMatrix`). So:

- **HS** generates a separate public encryption key (`EncKeyGenerator`) and encrypts under it.
- **PCMM** has no such option and encrypts under the client's own secret key directly.

Both are exclusively client-side operations — the secret key never leaves `seckeydir()` (which the
harness does not measure) either way — but it is a disclosed asymmetry: PCMM's client-side
encryption is symmetric-key CKKS, not public-key.

### Stage split

Every stage binary parses the instance size first and dispatches (`mlp::usePcmm`); the harness's
contract — seven fixed executable names, `<size>` as the only argv — does not change.

| Stage | Side | HS (single/small) | PCMM (medium/large) |
| --- | --- | --- | --- |
| `client_key_generation <size>` | client | sk (lifted), public enc key, rotation keys ×2, relin key | sk (unlifted), relin key only |
| `server_preprocess_model` | server | parse+cache CSV weights in padded p×q layout | *(same invocation)* also caches raw CSV shapes for PCMM |
| `client_preprocess_input <size>` | client | normalize, center-crop — **cleartext**, identical for both schemes, see [§2](#2-cleartext-pre--and-post-processing) | |
| `client_encode_encrypt_input <size>` | client | pack 128 img/ciphertext, encrypt under **public** key | pack whole batch into 1 `ICtMatrix`, encrypt under **secret** key |
| `server_encrypted_compute <size>` | server | **the entire inference, on ciphertext**, either way | |
| `client_decrypt_decode <size>` | client | decrypt, read slot `r*128+i` | decrypt, read matrix column `(i/ringDim)*degree + i%ringDim` of row `r` |
| `client_postprocess <size>` | client | argmax — **cleartext**, identical output format for both schemes, see [§2](#2-cleartext-pre--and-post-processing) | |

`server_preprocess_model` is invoked **with no arguments** in both schemes, so it cannot know the
instance size and cannot reach `io/<size>/public_keys`. HS's `MatrixVectorEval` encodes its weight
diagonals in its constructor, on the device of the rotation keys; PCMM's `buildModel` encodes `U1`/
`U2`/`b2` at levels/shapes that depend on the batch size. Neither can happen in stage 3, so both
happen in stage 7. Stage 7 therefore times *setup* (key load + diagonal/weight encoding) separately
from *evaluation* and reports both in `io/<size>/server_reported_steps.json`. The reference
submission has the same shape — its `server_preprocess_model` is a no-op.

---

## 2. Cleartext pre- and post-processing

Everything the **model** computes runs on ciphertext. The cleartext steps are:

| Step | Side | Note |
| --- | --- | --- |
| `(p − 0.1307) / 0.3081` normalization | client, before encryption | The harness writes pixels already scaled to [0,1]. Identical to the reference submission and to the harness's own model (`harness/mnist/test.py:60`). |
| **center-crop 28×28 → 22×22** | client, before encryption | See below. |
| argmax over 10 decrypted logits | client, after decryption | Identical to the reference's `client_postprocess`. |
| BatchNorm folded into fc1 | offline, model-only | Standard eval-mode BN folding, done once at weight export; does not touch the input. |

### The crop

**What it is.** Before encrypting, the client trims 3 pixels from each side of the 28×28 image,
giving 22×22 = 484 values. The encrypted input is therefore 484-dimensional, not 784.

**Why it is part of the model, not an optimization.** The network was *trained* on 22×22 input:
`fc1` is literally a 128×484 matrix. There is no 784-input model here to run, and no operation the
model performs has been moved out of the encrypted stage.

**Its effect, stated plainly.** The crop is not free of consequence, and we would rather point at
it than let a reader find it. 484 pads to **512** coordinates instead of the 1024 that 784 would
need. Since a ciphertext holds 65536 slots, that is **128 images per ciphertext instead of 64** —
a **2× throughput and 2× bandwidth advantage that follows directly from a cleartext step.** Every
per-image timing and every `ciphertexts_upload` figure in this submission carries that factor.

A reader comparing against a submission that encrypts all 784 pixels should keep that 2× in mind.
Removing the objection entirely would mean training and deploying a 784-input variant.

---

## 3. Deviation from the harness model

`harness/mnist/model.py` defines:

```
fc1 Linear(784,128) -> ReLU -> fc2 Linear(128,64) -> ReLU -> fc3 Linear(64,10)
```

This submission evaluates a different network:

```
fc1 Linear(484,128) -> x^2 -> fc2 Linear(128,10)
```

Two layers instead of three, `x²` instead of ReLU, and a 484-dimensional cropped input. It is
trained separately (x²+BatchNorm, 30 epochs, seed 0, BN folded at export). The reference OpenFHE
submission in this repository also deviates — it uses a 512×784 first layer with a polynomial
approx-ReLU.

Measured accuracy is in line with the harness plaintext model; see [§4](#4-results).

---

## 4. Results

Development runs, seed 3, through the **unmodified harness**, on the **shared** 5090 developer
server — not the benchmark server. **These are not official measurements**: `measurements/` still
holds the reference OpenFHE submission's numbers, and per the benchmark rules official figures are
produced only on confirmed hardware, which here means the exclusive bench server
([which machine](BUILDING.md#which-machine)). A shared node also means these timings carry whatever
else was running at the time.

All figures below are from a HEaaN2 built **natively for sm_120**, verified with `cuobjdump`
(20/20 cubins sm_120, no PTX). That verification is not incidental — see
[the retraction](#retracted-the-first-gpu-pcmm-numbers-were-measured-on-a-mis-built-library) at the
end of this section.

### All four sizes, as shipped

| | size 0 (1) HS | size 1 (100) HS | size 2 (1000) PCMM | size 3 (10000) PCMM |
| --- | --- | --- | --- | --- |
| harness `Encrypted computation` (wall) | 4.63 s | 4.65 s | **0.44 s** | **0.57 s** |
| ├─ server-reported model setup | 4.44 s | 4.46 s | 0.19–0.22 s | 0.20 s |
| └─ server-reported evaluation | 0.016 s | 0.016 s | 0.040–0.046 s | 0.042 s |
| Public + evaluation keys | 174.8 M | 174.8 M | **54.0 K** | **54.0 K** |
| Encrypted input | 1.6 M | 1.6 M | 44.5 M | 133.6 M |
| Encrypted results | 480 K | 480 K | 280 K | 840 K |
| **Encrypted-model accuracy** | PASS | **0.9800** | **0.9890** | **0.9796** |
| Harness plaintext model | — | 0.9700 | 0.9820 | 0.9776 |

### Why PCMM at medium/large: a controlled A/B on the same node

The scheme split is not taken on faith. Both schemes were built and run against the same instances,
on the same machine, in the same session, against the same sm_120 library (HS forced at sizes 2/3 by
temporarily overriding `usePcmm`). They differ in *opposite directions* on the two halves of the
server's work:

| size 2 (1000 images) | HS | PCMM | |
| --- | --- | --- | --- |
| model setup (key load + weight encoding) | 4.34–4.39 s | **0.19–0.22 s** | PCMM ~21× faster |
| evaluation only | **0.018 s** | 0.040–0.046 s | HS ~2.4× faster |
| **`Encrypted computation` (what the harness scores)** | 4.55–4.60 s | **0.44 s** | **PCMM ~10× faster** |
| public + evaluation keys | 174.8 M | **54.0 K** | PCMM ~3300× smaller |
| encrypted input | **13.1 M** | 44.5 M | HS ~3.4× smaller |

| size 3 (10000 images) | HS | PCMM | |
| --- | --- | --- | --- |
| model setup | 4.35 s | **0.20 s** | PCMM ~22× faster |
| evaluation only | 0.044 s | **0.042 s** | roughly equal |
| **`Encrypted computation`** | 4.66 s | **0.57 s** | **PCMM ~8× faster** |

**PCMM wins decisively on the metric the harness scores, and the reason is setup, not arithmetic.**
Deserializing 174.8 M of rotation keys and encoding weight diagonals costs HS ~4.4 s *every run*,
at every instance size. PCMM needs **no rotation keys at all** — 54 K of key material, essentially
just the relinearization key — so its setup is near-free. Since the harness times the whole stage-7
process, that fixed cost is what gets scored.

On pure evaluation the two cross over right about where the split is placed: HS is ~2.4× faster at
1000 images, and by 10000 they are level (0.044 vs 0.042 s). That is the expected shape — PCMM's
GEMM amortizes better as the batch grows — and it means the split at medium/large is the right
call on *both* halves at size 3, and on the scored metric at size 2.

Two honest qualifications:

- At size 2, if HS's setup were amortized (a long-running server loading keys once), **HS would be
  the faster choice on evaluation** by ~2.4×. The split as shipped is right *for this benchmark's
  cost model*, which re-pays setup on every run; it is not a claim that PCMM's arithmetic is faster
  at every batch size.
- PCMM's **encrypted input is larger**, not smaller (44.5 M vs 13.1 M at size 2). PCMM wins on
  compute and key material and loses on upload bandwidth; it does not dominate on every axis.

### Retracted: the first GPU PCMM numbers were measured on a mis-built library

An earlier revision of this document reported PCMM's GPU evaluation as **~9.3 s** at size 2 —
~35× slower than the same batch on CPU — and flagged it as an unexplained anomaly, guessing at
kernel-launch overhead on a new architecture. **The measurement was real; the explanation was
wrong, and so was the number's relevance.**

Root cause, found by running `cuobjdump` on the library rather than trusting how it was configured:
the HEaaN2 build in use contained **sm_52 cubins and `compute_52` PTX only** — CMake's default
architecture. HEaven's own architecture-selection fallback is unreachable once CMake seeds
`CMAKE_CUDA_ARCHITECTURES` in the cache (see
[why `HEAAN2_CUDA_ARCH` has to be passed](BUILDING.md#environment-variables)), so a build that looks
correctly configured silently targets sm_52. That library still *runs* on an RTX 5090 — by
JIT-compiling its PTX at load — which is exactly what the 9.3 s was: one-time JIT, not computation.
Subsequent runs hit `~/.nv/ComputeCache` and dropped to ~0.12 s, which is why the figure looked
like an unreproducible transient.

Rebuilding with `CMAKE_CUDA_ARCHITECTURES=120-real` fixed both halves of the problem:

| size 2 PCMM evaluation | sm_52 + PTX JIT | sm_120 native |
| --- | --- | --- |
| first run on a cold JIT cache | ~9.3 s | **0.046 s** |
| subsequent runs | 0.109–0.124 s | 0.040–0.046 s |

So the mis-built library was both ~2.6× slower once warm *and* carried a multi-second first-run
cliff that made timings depend on a cache outside the repo. Every number in this section is from the
sm_120 build. Two lessons worth keeping: **verify a CUDA library's architectures before quoting any
timing from it**, and treat "unexplained transient" as a hypothesis to falsify, not a caveat to
publish.

### Against the reference OpenFHE submission recorded in `measurements/`

| | size 0 | size 1 | size 2 |
| --- | --- | --- | --- |
| reference `Encrypted computation` | 7.61 s | 647.07 s | 6505.61 s |
| reference keys | 1.0 G | 1.0 G | 1.0 G |
| reference encrypted input | 5.0 M | 500.3 M | 4.9 G |
| reference accuracy | — | 0.96 | 0.974 |

Caveats worth stating before anyone quotes the ratios:

1. **Fixed setup dominates HS's every instance size.** The harness's `Encrypted computation` row is
   wall-clock around the whole process, so it also contains key deserialization and weight-diagonal
   encoding. At HS size 0 that is 4.56 s against 0.031 s of actual evaluation. For single-shot
   latency, 4.56 s is the number to quote — not 31 ms. PCMM's setup is much smaller (no rotation
   keys to deserialize at all), so this effect is far less pronounced there.
2. **The evaluation figures are cold.** Every stage is a fresh process, so the first homomorphic
   operation pays context and kernel initialization that a long-running server would amortize. The
   equivalent warm figure from HEaaN2's own standalone HS benchmark is ~1 ms per 128 images.
3. **Machines differ across rows above.** The reference numbers in `measurements/` are CPU; ours
   are GPU. Do not compare evaluation columns across those rows as if they were the same hardware.
4. **This is a shared development node.** One measurement taken on it was already wrong by two
   orders of magnitude (see the retraction above). Treat every timing here as indicative, and
   re-measure on the bench server before publishing anything.

---

## 5. Security

**The ≥128-bit claim for both schemes' configurations has not been signed off.**

The uniform-ternary (hw = 0) budget table (`maxBits128` in
[`include/mlp_params.hpp`](include/mlp_params.hpp), shared by both schemes) has 2^13–2^15 entries
from HEaven's `maxBitsPolicy128()` and 2^16–2^17 entries citing ePrint 2024/463; a 2^12 entry (106
bits) was added for PCMM. **Those numbers are provisional here and the hamming weight may change.**

**HS.** The switching-key budget is sized from the degree the secret key was actually *sampled*
at, not the ring it was lifted into: lifting adds no entropy, so a key lifted from 2^15 carries the
LWE problem of 2^15, and the conjugate-invariant ring halves it again because only half the
coefficients are sampled — giving an effective dimension of 2^14 and a budget of 430 bits, against
a ~105-bit modulus chain. **The lifted-key construction specifically has not been reviewed** — see
`NOTES_FOR_HUMAN.md` for what happens to the scheme choice if it is rejected.

**PCMM.** No lifting is involved: the secret key is sampled directly at N=2^13, so its security
rests on `maxBits128(12) = 106` bits alone (CI again halves the sampled degree to the RLWE
dimension 2^12), against a ~94-bit modulus chain. There is no separate lifting argument to review
for this scheme, which is simpler to reason about than HS's even before a review lands.

Everything a review would need to change is confined to one block of each of `mlp::` and
`mlp::pcmm` in [`include/mlp_params.hpp`](include/mlp_params.hpp). This section must be replaced
with a finalised justification before the submission is considered complete; per the benchmark
rules a parameter claim that cannot be justified is recorded as a gap rather than asserted.

### Why CryptoLab's Zn-multiplication citation does not transfer here

The [Zn-multiplication submission](https://github.com/CryptoLabInc/Zn-multiplication/tree/CryptoLabInc)
closes its own security section by citing
[sparse-key-estimate](https://github.com/jdumezy/sparse-key-estimate/blob/master/Precomputed-Tables/128bits_security.md),
whose table gives a 128-bit log(PQ) bound per (ring degree, secret Hamming weight). Its numbers line
up exactly: N = 2^16, hw = 32, log(PQ) = 114 ≤ the table's `logn=16, h=32` entry of **349**.

**That citation cannot be reused for this submission as parameterised**, and the reason is not
cosmetic. That table is indexed by *sparse* Hamming weights — its columns are h ∈ {32, 64, 128, 192,
256, 512, 1024}. Both of our schemes use `hw = 0`, which in HEaaN2 means a **uniform-ternary**
(dense) secret, i.e. h ≈ 2n/3 — far outside every column. There is no row to read off.

Our numbers are close to, but not identical to, that table's densest column (e.g. our
`maxBits128(13) = 214` equals its `logn=13, h=1024` entry exactly, while our 2^14 entry is 430
against its 426, and 2^15 is 868 against its 854), consistent with their stated provenance in
HEaven's own `maxBitsPolicy128()` rather than in this table. Reading a dense key's bound off the
h=1024 column would in principle be *conservative* — security increases with Hamming weight, so a
uniform-ternary key is at least as hard as an h=1024 one at the same (n, q) — but "in principle
conservative" is an argument a reviewer has to accept, not a citation, and the small numeric
disagreements show the two sources are not the same analysis.

So there are two coherent ways to close this, and it is a decision, not an oversight:

1. **Keep `hw = 0` and cite something appropriate for a dense key** — a lattice-estimator run at our
   exact (n, q, σ, dense-ternary) points, or the HE standard. This is what the pending crypto-side
   review needs to supply.
2. **Switch to a sparse key** (hw = 32, say) so the same public table the sibling submission cites
   applies directly. This is a real parameter change, not a documentation edit: it alters the noise
   growth the level schedule was tuned against, so accuracy and the bottom-modulus headroom would
   both need re-validating.

Recorded here so the gap is not mistaken for "the sibling submission already solved this."
