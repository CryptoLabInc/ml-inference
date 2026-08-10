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

### HS (single, small) — GPU, RTX 5090 via `srun`

| | size 0 (1) | size 1 (100) |
| --- | --- | --- |
| harness `Encrypted computation` (wall) | 4.56 s | 4.59 s |
| ├─ server-reported model setup | 4.34 s | 4.37 s |
| └─ server-reported evaluation | **0.031 s** | **0.032 s** |
| Public + evaluation keys | 174.8 M | 174.8 M |
| Encrypted input | 1.6 M | 1.6 M |
| Encrypted results | 480 K | 480 K |
| Total harness latency | 10.8 s | 73.7 s |
| **Encrypted-model accuracy** | PASS | **0.9800** |
| Harness plaintext model | — | 0.9700 |

### PCMM (medium, large) — correctness verified on both CPU and GPU

| | size 2 (1000), CPU | size 3 (10000), CPU | size 2 (1000), GPU |
| --- | --- | --- | --- |
| harness `Encrypted computation` (wall) | 0.45 s | 1.00 s | 9.82 s |
| ├─ server-reported model setup | 0.11 s | 0.12 s | 0.24 s |
| └─ server-reported evaluation | 0.267 s | 0.700 s | 9.34 s |
| Public + evaluation keys | 54.0 K | 54.0 K | 54.0 K |
| Encrypted input | 44.5 M | 133.6 M | 44.5 M |
| Encrypted results | 280 K | 840 K | 280 K |
| **Encrypted-model accuracy** | **0.9890** | **0.9796** | **0.9880** |
| Harness plaintext model | 0.9820 | 0.9776 | 0.9820 |

CPU figures are from a full harness run on an isolated CPU-only HEaaN2 install
(`HEAAN2_BUILD_CUDA=OFF`); GPU figures are from the same shared RTX 5090 node the HS numbers above
came from. **Read the GPU eval row with real skepticism before quoting it anywhere:**

- The GPU number (9.34 s) is **~35× slower than the same batch on CPU** (0.267 s) for the identical
  operation sequence — the opposite of the expected direction, and the opposite of what the
  standalone `HEaaN2/mlp/PCMM/MLInference_Large` benchmark reports on its own hardware
  (2.54 ms/1000 images). Accuracy is correct on both (0.988 GPU vs 0.989 CPU — the half-point gap
  is independent encryption noise on a different random draw of the same seed, not a bug), so this
  is a *performance* anomaly, not a correctness one.
- The likely explanation is fixed per-kernel-launch overhead not being amortized: PCMM's `pcmm`/
  `tensor`/`relin` calls at N=2^13 are individually tiny compared to HS's N=2^17 operations, and
  this GPU is a very recent architecture (RTX 5090, sm_120/Blackwell) that a research library's
  kernels may not yet be well-tuned for. This is a hypothesis, not a diagnosis — it was not chased
  further; see the note below.
- **We did not attempt to root-cause or optimize this.** The user separately measured PCMM as
  *faster* than HS at medium/large on GPU, on hardware other than this shared dev node, which is
  the basis for choosing PCMM at these sizes at all. This dev box's GPU PCMM timing should be
  treated as unreliable and re-measured on the actual benchmark server before any performance claim
  is published — see `NOTES_FOR_HUMAN.md`.

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
3. **Machines differ across rows above.** The reference numbers in `measurements/` are CPU; our HS
   figures are GPU; our PCMM figures span both CPU and GPU on this dev node specifically because the
   GPU number needed the CPU one alongside it to be legible at all. Do not compare evaluation
   columns across machines as if they were the same hardware.

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
