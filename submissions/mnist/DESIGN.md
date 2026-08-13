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

### Scheme A — Halevi–Shoup (size 0)

| | |
| --- | --- |
| Ring | N = 2^15, CI subring (ePrint 2018/952), `GRAFTED`, 16384 slots |
| Modulus chain | 30 + 3×25 ≈ 105 bits, 4 levels, no bootstrapping |
| Secret key | uniform ternary (hw = 0), sampled directly at 2^15 |
| Noise / SWK budget | σ = 3.2 / `maxBits128(14) = 430` bits, margin 5.0 |

```
L3  encrypt → fc1 matvec              → L2
L2  fold, +b1, x², rescale            → L1
L1  fc2 matvec                        → L0, +b2, decrypt
```

**Packing.** One image occupies 512 padded coordinates; coordinate `c` of image `i` sits in slot
`c*32 + i` — **32 images per ciphertext**, so a batch of `n` needs `ceil(n/32)` ciphertexts, not
`n`. Only size 0 runs this scheme, so one ciphertext is all it ever builds. The harness only
measures the byte size of `ciphertexts_upload/`, so the layout is a free choice.

`fc1` is rectangular 128×512 and folds 4:1, with keys. `fc2` is deliberately *squared* to 128×128
rather than the natural 16×128, making `q/p == 1` so it needs no fold at all: its cosets ride the
matvec's giant steps instead, which double-hoisted BSGS accumulates behind a single mod-down.

### Scheme B — PCMM (sizes 1–3)

Feature = ciphertext **row** (the GEMM contraction dim), image = **column/slot** — the opposite
convention from HS. Written against HEaaN2's public API.

Sizes 1–2 and size 3 are **tuned separately**. The batch size decides how many blocks a matrix row
splits into, and that in turn decides which ring and which modulus chain come out cheapest, so one
setting cannot be right for every size. Sizes 1 and 2 share one profile because both fit a single
block and do identical work.
`mlp::pcmm::profile(size)` in [`include/mlp_params.hpp`](include/mlp_params.hpp) is the selector.

| | sizes 1–2 (100, 1000) | size 3 (10000) |
| --- | --- | --- |
| Ring | N = 2^12, **NORMAL**, `SIMPLE32` | N = 2^15, **CI subring**, `SIMPLE32` |
| Coefficients / message | 4096 (the full degree) | 16384 (CI's free half) |
| Images / message | 2048 | 16384 |
| Blocks per row | 1 | 1 |
| Modulus chain | 34 + 3×24 ≈ 106 bits, 4 levels, no bootstrapping | 44 + 3×27 ≈ 125 bits, 4 levels, no bootstrapping |
| Secret key | uniform ternary (hw = 0), sampled **directly** at 2^12 — no lifting | uniform ternary (hw = 0), sampled **directly** at 2^15 — no lifting |
| Noise / SWK budget | σ = 3.2 / `maxBits128(12) = 106` bits, margin 5.0 | σ = 3.2 / `maxBits128(14) = 430` bits, margin 5.0 |

`SIMPLE32` (32-bit RNS primes) halves the word size pcmm's GEMM backend operates on, at the cost of
more primes for the same modulus budget. Both chains are sized against that trade, so neither is
valid read back against a `GRAFTED` budget.

Note the first two rows differ: **coefficients per message is not images per message.** CI stores
half the degree but packs one real image per stored coefficient; NORMAL stores the whole degree but
packs images into its N/2 complex slots. The two coincide under CI only, which is why
[`mlp_params.hpp`](include/mlp_params.hpp) keeps `ringDim()` and `slotsPerMsg()` as separate
functions.

```
L3  encrypt X, encode U1 → fc1 pcmm+rescale   → L2
L2  x²: tensor → rescale → L1, relin @L1
L1  encode U2 → fc2 pcmm+rescale              → L0
L0  +b2 (plaintext add, no level cost), decrypt
```

One message holds 2048 images at sizes 1–2 and 16384 at size 3, so every shipped size is a single
block; a batch beyond that would split into further *blocks* inside a single
`ICtMatrix`, transparently. `x²` runs tensor → rescale → relin (not the usual tensor → relin →
rescale), so the relin key is built at the *post-rescale* level.

**Bias fold.** `b1` rides as an extra column of `U1` (`128×485`) against an appended ones-row in the
packed input, so fc1's pcmm computes `W1·X + b1` directly. `b2` is *not* folded that way — it would
carry the ones-row through `x²` and `fc2`, taking the contraction from 128 to 129 rows for one bias
vector — so it is added afterwards as a plaintext delta (its inverse DFT is one nonzero coefficient
per block, which a zero-filled matrix already provides).

### Encryption: public key vs symmetric key

An API constraint, not a design choice. HEaaN2's `EnDecryptor` has a public-key overload for plain
ciphertexts but **only a secret-key overload for matrices** (no `IEncKey` overload for
`ICtMatrix`). So HS encrypts under a public encryption key; **PCMM encrypts under the client's own
secret key.** Both are purely client-side and the secret key never leaves `seckeydir()` (which the
harness does not measure) — but it is a real asymmetry, disclosed here rather than left to be
noticed.

Moving size 1 onto PCMM widens it: public-key encryption is now exercised at size 0 alone. Keeping
size 0 on HS is deliberate for that reason — it is what stops the submission from resting on the
symmetric-key path everywhere.

### Stage split

Every stage binary parses the size and dispatches; the harness contract (seven fixed names,
`<size>` as the only argument) is unchanged.

| Stage | HS (size 0) | PCMM (sizes 1–3) |
| --- | --- | --- |
| `client_key_generation` | sk, public enc key, 2× rotation keys, fc1 fold keys, relin key | sk, relin key only |
| `server_preprocess_model` | caches CSV weights in padded p×q layout | same call also caches raw CSV shapes |
| `client_preprocess_input` | normalize + center-crop — cleartext, identical both schemes ([§2](#2-cleartext-pre--and-post-processing)) | ← |
| `client_encode_encrypt_input` | 32 img/ciphertext, **public** key | whole batch → 1 `ICtMatrix`, **secret** key |
| `server_encrypted_compute` | the entire inference, on ciphertext | ← |
| `client_decrypt_decode` | read slot `r*32+i` | read column `(i/ringDim)*degree + i%ringDim` of row `r` |
| `client_postprocess` | argmax — cleartext, same output format both schemes | ← |

`server_preprocess_model` is invoked **with no arguments**, so it cannot know the instance size or
reach `io/<size>/public_keys`. That constrains what each scheme can hoist into stage 3, and the two
answers differ:

- **HS encodes its diagonals in stage 3.** The diagonal geometry is fixed (`fc1` 128×512, `fc2`
  128×128) and does not depend on the batch size — batch size only changes how many ciphertexts are
  fed through — so stage 3 has everything it needs without knowing the instance. `MatrixVectorEval`
  would normally encode inside its constructor, on the rotation keys' device, behind an interface
  that cannot return the encoded state; `MatrixVectorEvalEncoded` performs that encoding separately
  from any key material, taking only the gadget decomposition, and `serial::save`/`load` carry the
  result across the process boundary. Stage 7 then rebuilds the evaluator from the encoded
  diagonals, skipping what was the expensive half of its setup.
- **PCMM still encodes in stage 7.** `buildModel` encodes at batch-size-dependent shapes, which
  stage 3 cannot know. This costs it little: PCMM's setup is ~50 ms either way, because it has no
  rotation keys to deserialize.

Stage 7 reports *setup*, *warm-up* and *evaluation* separately in `server_reported_steps.json`. The
reference submission leaves stage 3 a no-op and pays everything in stage 7.

The harness times stage 3 as `Encrypted model preprocessing`, so this moves HS's ~7.3 s of diagonal
encoding out of the scored `Encrypted computation` rather than eliminating it — it is genuinely
model-only work, done once per model instead of once per input. [§4](#4-results) reports both
figures side by side.

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
need. At the HS scheme's 16384 slots that is **32 images per ciphertext instead of 16** — a **2×
throughput and bandwidth advantage that follows directly from a cleartext step.** Every per-image
timing and every upload figure here carries that factor, and anyone comparing against a submission
that encrypts all 784 pixels should keep it in mind. Removing the objection entirely would mean
training a 784-input variant.

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

Seed 3, through the **unmodified harness**, on 1× RTX 5090 (sm_120). These are the official
measurements: every figure below is the mean of the three runs committed under
[`measurements/`](../../measurements/), taken with the stage-7 warm-up and timer synchronization in
place. All come from a HEaaN2 verified as natively sm_120 by the
[architecture check](BUILDING.md#-this-submission-requires-an-sm_120-gpu) — a library that falls
back to JIT-compiling PTX reports several seconds of first-run cost as if it were evaluation time,
so that check is a precondition for quoting any timing here.

### As shipped

| | size 0 (1) HS | size 1 (100) PCMM | size 2 (1000) PCMM | size 3 (10000) PCMM |
| --- | --- | --- | --- | --- |
| harness `Encrypted model preprocessing` | 2.15 s | — | **0.068 s** | **0.071 s** |
| harness `Encrypted computation` | 0.41 s | — | 0.44 s | 0.56 s |
| ├─ model setup | 0.116 s | — | 0.051 s | 0.050 s |
| ├─ warm-up (discarded) | 0.017 s | — | 0.032 s | 0.033 s |
| └─ evaluation | **0.76 ms** | — | **0.91 ms** | **2.63 ms** |
| Public + evaluation keys | 44.9 M | — | **54.0 K** | **54.0 K** |
| Encrypted input | 420 K | — | 44.5 M | 133.6 M |
| Encrypted results | 120 K | — | 280 K | 840 K |
| **Accuracy** | PASS | — | **0.989** | **0.979** |
| Harness plaintext model | — | — | 0.981 | 0.978 |

> **Size 1 awaits re-measurement.** It moved from HS to PCMM after this table was taken, so its
> former column described a circuit it no longer runs and has been cleared rather than carried
> over. Because `numBlocks` is 1 at both 100 and 1000 images, it is expected to land on size 2's
> figures; that is a prediction, not a measurement, and the column stays empty until the bench
> machine fills it.

**Read the two harness rows together.** HS's diagonal encoding runs in stage 3
([§1](#stage-split)), so at size 0 the scored `Encrypted computation` is 0.41 s while the ~2.1 s
of encoding it depends on is reported as `Encrypted model preprocessing`. The work moved out of the
scored stage; it did not get cheaper. What makes the move legitimate rather than accounting is that
the encoding depends only on the weights — it is paid once per model, whereas stage 7 is paid once
per input batch. Quoting the 0.41 s alone, without the 2.15 s beside it, would misrepresent
single-shot latency.

The stage-7 sub-rows do not sum to the scored figure: the harness times the whole stage-7 *process*,
so it also carries ~0.25 s of interpreter and CUDA-context startup and ciphertext I/O that sits
outside the submission's own timers.

Accuracy varies slightly run to run from encryption noise (size 3 measured 0.9793–0.9798 across
runs); the table rounds. The harness plaintext row is the harness's own model on the same subset,
reported for reference — the encrypted model scores at or above it at every size.

### Why PCMM at sizes 1–3 — a controlled A/B

Both schemes, same instances, same machine, same session, same sm_120 library (HS forced at sizes
2–3 by overriding `usePcmm`):

> **The HS columns predate the size-0 parameter change** and were taken with HS at 2^17 under a
> lifted key. HS now runs at 2^15, which packs 32 images per ciphertext instead of 128 and so needs
> 4× as many ciphertexts for the same batch — strictly worse at these sizes, where its cost already
> tracks ciphertext count almost exactly. The conclusion below therefore holds a fortiori, but the
> HS numbers themselves are not the ones the current tree would produce and are kept only as the
> comparison that motivated the split.

| | HS stage 3 | HS eval | HS scored | PCMM stage 3 | PCMM eval | PCMM scored |
| --- | --- | --- | --- | --- | --- | --- |
| size 2 (1000) | 7.36 s | 7.72 ms | 0.64 s | **0.068 s** | **0.91 ms** | **0.44 s** |
| size 3 (10000) | 7.36 s | 75.3 ms | 0.85 s | **0.071 s** | **2.63 ms** | **0.56 s** |

**PCMM wins on preprocessing and arithmetic; the two are close on the scored stage.** HS needs
174.8 M of rotation keys and ~7.3 s of diagonal encoding, against PCMM's 54 K and ~70 ms — roughly
105× less stage-3 preprocessing. On evaluation PCMM is 8.5× faster at 1000 images and 29× faster at
10000, the gap widening with batch size for the reason below.

On the scored `Encrypted computation` the margin is now only ~1.5×, because moving HS's encoding
into stage 3 took most of its cost out of the timed stage. **The split is therefore justified by
stage 3 and by evaluation, not by the scored figure alone** — and by key material, which is a
property of the scheme rather than of where the work is timed.

The evaluation gap widens with batch size because the two scale differently: HS packs a fixed 128
images per ciphertext, so 1000 images need 8 ciphertexts and 10000 need 79, and its cost tracks
that count almost exactly (7.72 ms → 75.3 ms, ~10× for 10× the images). PCMM's GEMM amortizes over
the batch instead (0.91 ms → 2.63 ms, 2.9× for the same 10×). This is why the split is on instance
size rather than a tuning constant.

Two honest qualifications:

- **The crossover turned out to be below 100 images, not above it.** This A/B forces HS at the
  sizes PCMM ships at, not the reverse, so it could not locate the crossover; an earlier draft
  predicted from that gap that HS's single-ciphertext packing would win at 1 and 100 images. The
  mirrored experiment has since been run at 100 images (sm_89 development box, not this table's
  machine) and contradicted the prediction: PCMM reproduced size 2's cost outright, because
  `numBlocks` is 1 for any batch up to 4096, at equal accuracy and ~58 K of key material against
  HS's key material at the time. Size 1 therefore ships on PCMM. Size 0 stays on HS both for
  public-key encryption and because HS is now the faster of the two there — 0.76 ms against
  0.91 ms, since a PCMM block costs the same whether it carries 1 image or 4096.
- **Earlier drafts of this table reported the opposite on arithmetic.** Their stage-7 timers closed
  before the GPU had finished, under-reporting PCMM by roughly 4.7× and making HS look faster at
  size 2. The timers now synchronize and the evaluation is warm, which reverses that conclusion.
  Only the submission's self-reported breakdown was ever affected — the harness's own
  `Encrypted computation` was honest throughout.
- **PCMM's encrypted input is larger**, not smaller (44.5 M vs 13.1 M at size 2; 133.6 M vs 129.6 M
  at size 3, where the gap nearly closes). It wins on compute and key material, and loses on upload
  bandwidth — most visibly at size 2. **The per-size tuning widened that gap deliberately**: the
  new chains are 106 and 125 bits against the 94 these figures were taken at, which buys evaluation
  time and costs upload. On the development box the same trade showed size 2's input at 51.2 M
  against 44.5 M, and size 3's at 242.5 M against 133.6 M. Bandwidth is reported by the harness
  alongside timing, so this is a real cost, not an accounting artefact.

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
It is not cheaper — `MLInference_128` spends ~14.6 s of its 14.67 s wall time there, doing the same
256-diagonal encoding — it simply amortizes it over many evaluations, which is the right shape for
a throughput benchmark. The harness models a cold client→server round trip instead, so stage 7 is a
fresh process every run and re-pays that setup each time.

Measured split of HS setup at size 0 (same 4090, and at the 2^17 lifted parameters HS used then —
the shape is what matters here, not the magnitudes):

| | |
| --- | ---: |
| Rotation-key deserialization (174.8 M) | 0.27 s |
| Weight read | 0.002 s |
| **Diagonal encoding (`MatrixVectorEval` construction ×2)** | **8.44 s** |

So it is encoding, not I/O. Those encoded diagonals depend only on the weights, the layer geometry
and the level/scale — every one a compile-time constant — so for HS they are genuinely
instance-independent, model-only artifacts, exactly what stage 3 exists for.

**They now run there.** `MatrixVectorEvalEncoded` encodes the diagonals given only the gadget
decomposition, with no rotation keys involved; `serial::save`/`load` carry the encoded plaintexts
across the process boundary; and `MatrixVectorEval` gained a constructor that takes them and shares
rather than re-encodes. Stage 3 therefore does the encoding once and stage 7 rebuilds the evaluator
from it, which is what drops HS's stage-7 setup to ~0.12 s on the bench machine. Encoding is done
on the device the evaluation will run on: the library does not guarantee that encoding on the CPU
and moving to a GPU afterwards yields bit-identical plaintexts. PCMM was never affected — its setup
is ~50 ms, because it has no rotation keys and no diagonals.

### Against the reference OpenFHE submission

| | size 0 | size 1 | size 2 |
| --- | --- | --- | --- |
| reference `Encrypted computation` | 7.61 s | 647.07 s | 6505.61 s |
| reference keys / input | 1.0 G / 5.0 M | 1.0 G / 500.3 M | 1.0 G / 4.9 G |
| reference accuracy | — | 0.96 | 0.974 |

Before quoting any ratio:

1. **HS is still preprocessing-bound at size 0**, the cost has only moved stages: 0.58 s scored
   and 7.23 s of stage-3 encoding, against 0.99 ms of actual evaluation. For a single-shot,
   cold-start latency comparison the honest number is the ~7.8 s of both stages together — not the
   0.58 s scored figure, and certainly not 1 ms. The reference submission pays its equivalent work
   inside `Encrypted computation`, so compare it against the two stages summed.
2. **Reference numbers are CPU; ours are GPU.** Not the same hardware.

---

## 5. Security

**The ≥128-bit claim is not signed off for either scheme.**

The shared uniform-ternary (hw = 0) budget table is `maxBits128` in
[`include/mlp_params.hpp`](include/mlp_params.hpp): entries for 2^13–2^15 from HEaven's
`maxBitsPolicy128()`, 2^16–2^17 citing ePrint 2024/463, and a 2^12 entry added for PCMM. **These
are provisional and the Hamming weight may change.**

- **HS.** No lifting: sampled directly at 2^15, the ring it is used in, and CI halves the effective
  dimension again (only half the coefficients are sampled) — effective dimension 2^14, budget
  `maxBits128(14) = 430` bits, against a ~105-bit chain. An earlier configuration sampled at 2^15
  and lifted to 2^17; that carried the same LWE problem and the same 430-bit budget, since lifting
  adds no entropy, but required a review to accept the lifted-key construction itself. Dropping the
  lifting removes that argument from the analysis without weakening any parameter — see
  [§1](#scheme-a--halevishoup-size-0) for why it was worth doing on performance grounds too.
- **PCMM.** Also no lifting: sampled directly at 2^13, so security rests on `maxBits128(12) = 106`
  bits (CI → RLWE dimension 2^12) against a ~94-bit chain.

Neither scheme now relies on a lifted key, so the two rest on the same kind of argument and differ
only in dimension. What remains unreviewed is the `maxBits128` table itself at hw = 0.

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
