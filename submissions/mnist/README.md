# MNIST MLP inference on HEaaN2 (CKKS, GPU)

FHE submission for the HomomorphicEncryption.org `ml-inference` benchmark
(`--dataset mnist`), built on [HEaaN2](https://github.com/CryptoLabInc/HEaaN2), Crypto Lab's
CKKS library. Replaces the reference OpenFHE/HEIR submission in this directory; the harness is
unmodified.

> **Security notice.** The ≥128-bit parameter justification for this configuration is **not
> finalised** — see [Security](#security). Do not cite this submission's parameters as reviewed.

---

## 1. Build and run

### Requirements

| | |
| --- | --- |
| HEaaN2 | v0.2.0, built **with CUDA** and installed to a prefix |
| GPU | CUDA device, compute capability ≥ 7.5. Developed on an RTX 5090 (sm_120) |
| Toolchain | CMake ≥ 3.23, a C++17 compiler, OpenMP, OpenBLAS **including headers** (`cblas.h`) |
| Python | the harness's own `requirements.txt` (torch 2.9.1, torchvision 0.24.1, numpy, absl-py) |

HEaaN2 is not redistributable, so `scripts/build_task.sh` does **not** fetch it. Point it at a
checkout or an existing install:

```bash
export HEAAN2_ROOT=/path/to/HEaaN2      # a checkout; built and installed on first use
# or, if HEaaN2 is already installed somewhere:
export HEAAN2_DIR=/path/to/heaan2/install
```

If only `HEAAN2_ROOT` is set and no install exists yet, the build script configures and installs
HEaaN2 into `$HEAAN2_ROOT/install` itself. Set `HEAAN2_BUILD_CUDA=OFF` for a CPU build (works, but
is not what this submission is measured on).

### Running

```bash
python -m venv bmenv && source ./bmenv/bin/activate
pip install -r requirements.txt

srun python3 harness/run_submission.py 0 --seed 3     # single
srun python3 harness/run_submission.py 1 --seed 3     # small (100)
srun python3 harness/run_submission.py 2 --seed 3     # medium (1000)
srun python3 harness/run_submission.py 3 --seed 3     # large (10000)
```

**`srun` (or an equivalent allocation) is required on a shared Slurm node.** The harness invokes
each stage binary through `subprocess.run`, and those children inherit the allocation, so one
`srun` around the whole harness is enough — no harness change is needed. Run bare on a node with
no CUDA device visible, stage 7 aborts with *"CUDA device is not available in the current
environment"*.

The harness builds the submission itself. To build standalone:

```bash
HEAAN2_ROOT=/path/to/HEaaN2 ./scripts/build_task.sh ./submissions/mnist
```

`scripts/get_openfhe.sh` is left untouched and still runs: `submissions/cifar10` depends on
OpenFHE, and the harness calls the script unconditionally. It is not used by this submission.

---

## 2. How the solution works

### Circuit

```
fc1 (128x484, +b1)  ->  x^2  ->  fc2 (10x128, +b2)
```

A BN-folded 2-layer MLP. `x^2` is the activation the network was **trained** with, not a
polynomial approximation of ReLU, so the homomorphic circuit evaluates the model exactly — the
only error is CKKS noise. Measured on the harness's size-0 input: max |decrypted − plaintext
logit| = 0.109 on logits spanning [−32, +14].

Weights are in [`weights/`](weights/) as row-major CSV, with shapes, normalization and provenance
in [`weights/manifest.txt`](weights/manifest.txt). Folded plaintext accuracy 97.96%.

### Scheme and parameters

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

All parameters live in one place, [`include/mlp_params.hpp`](include/mlp_params.hpp). HEaaN2 has no
serializable `CryptoContext`: every stage rebuilds its parameters from those constants, so they
are the de-facto context and must not be changed for one stage alone.

### Packing

One image occupies 512 padded coordinates; coordinate `c` of image `i` sits in slot
`c*128 + i`. That puts **128 images in a single ciphertext**, so a batch of `n` needs
`ceil(n/128)` ciphertexts rather than `n`. The harness measures only the byte size of
`io/<size>/ciphertexts_upload/`, so this layout is a free choice — and it is where the bandwidth
difference against the reference comes from.

`fc1` is the rectangular 128×512 shape and folds 4:1. `fc2` is deliberately *squared* to 128×128
rather than the natural 16×128: its fold stride would miss the key-less fold's invariance period,
and a keyed fold costs one mod-down per coset, so instead the cosets ride the matvec's giant steps
where the double-hoisted BSGS accumulates them behind a single mod-down.

### The key-less fold

The secret key is sampled at 2^15 and lifted to 2^17 with `SKGenerator::genHighDegreeKey`. Such a
key is invariant under exactly those rotations whose step is a multiple of 2^14, and fc1's fold
stride (128 × 128 = 16384) is one. Its fold therefore runs as bare Galois automorphisms
(`HomEval::frobMap`) — no rotation keys, no level, no scale change, no noise growth — in
log2(4) = 2 automorphisms. The divisibility is asserted at key generation, because getting it
wrong throws nothing: it would silently decrypt to noise.

### Stage split

| Stage | Side | What it does |
| --- | --- | --- |
| `client_key_generation <size>` | client | sk (lifted), public encryption key, rotation keys for both matvecs, relin key |
| `server_preprocess_model` | server | parse CSV weights, validate shapes, cache in the padded p×q layout |
| `client_preprocess_input <size>` | client | normalize, center-crop — **cleartext**, see §3 |
| `client_encode_encrypt_input <size>` | client | pack 128 images/ciphertext, encode, encrypt under the **public** key |
| `server_encrypted_compute <size>` | server | **the entire inference, on ciphertext** |
| `client_decrypt_decode <size>` | client | decrypt, read logit `r` of image `i` from slot `r*128 + i` |
| `client_postprocess <size>` | client | argmax — **cleartext**, see §3 |

The harness invokes `server_preprocess_model` **with no arguments**, so it cannot know the
instance size and cannot reach `io/<size>/public_keys`. Since `MatrixVectorEval` encodes its weight
diagonals in its constructor, on the device of the rotation keys, that encoding necessarily happens
in stage 7. Stage 7 therefore times *setup* (key load + diagonal encoding) separately from
*evaluation* and reports both in `io/<size>/server_reported_steps.json`. The reference submission
has the same shape — its `server_preprocess_model` is a no-op.

---

## 3. Cleartext pre- and post-processing

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

## 4. Deviation from the harness model

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

Measured accuracy is in line with the harness plaintext model; see §5.

---

## 5. Results

Development runs on one RTX 5090 (shared node, via `srun`), seed 3, through the **unmodified
harness**. **These are not official measurements** — `measurements/` still holds the reference
OpenFHE submission's numbers, and per the benchmark rules official figures are produced only on
confirmed hardware.

| | size 0 (1) | size 1 (100) | size 2 (1000) | size 3 (10000) |
| --- | --- | --- | --- | --- |
| harness `Encrypted computation` (wall) | 4.56 s | 4.59 s | 4.88 s | 4.75 s |
| ├─ server-reported model setup | 4.34 s | 4.37 s | 4.66 s | 4.42 s |
| └─ server-reported evaluation | **0.031 s** | **0.032 s** | **0.021 s** | **0.059 s** |
| Public + evaluation keys | 174.8 M | 174.8 M | 174.8 M | 174.8 M |
| Encrypted input | 1.6 M | 1.6 M | 13.1 M | 129.6 M |
| Encrypted results | 480 K | 480 K | 3.8 M | 37.0 M |
| Total harness latency | 10.8 s | 73.7 s | 12.9 s | 28.2 s |
| **Encrypted-model accuracy** | PASS | **0.9800** | **0.9890** | **0.9794** |
| Harness plaintext model | — | 0.9700 | 0.9820 | 0.9776 |

Against the reference OpenFHE submission recorded in `measurements/`:

| | size 0 | size 1 | size 2 |
| --- | --- | --- | --- |
| reference `Encrypted computation` | 7.61 s | 647.07 s | 6505.61 s |
| reference keys | 1.0 G | 1.0 G | 1.0 G |
| reference encrypted input | 5.0 M | 500.3 M | 4.9 G |
| reference accuracy | — | 0.96 | 0.974 |

Three caveats worth stating before anyone quotes the ratios:

1. **Fixed setup dominates every instance size.** The harness's `Encrypted computation` row is
   wall-clock around the whole process, so it also contains key deserialization and weight-diagonal
   encoding. At size 0 that is 4.56 s against 0.031 s of actual evaluation. For single-shot latency,
   4.56 s is the number to quote — not 31 ms.
2. **The evaluation figures are cold.** Every stage is a fresh process, so the first homomorphic
   operation pays CUDA context and kernel initialization that a long-running server would amortize.
   The equivalent warm figure from HEaaN2's own standalone benchmark is ~1 ms per 128 images.
3. **This is a GPU submission**; the reference numbers in `measurements/` are CPU. Comparing the
   evaluation columns directly compares two different machines as much as two implementations.

## Security

**The ≥128-bit claim for this configuration has not been signed off.**

The parameters are as tabulated in §2. The switching-key budget is sized from the degree the secret
key was actually *sampled* at, not the ring it was lifted into: lifting adds no entropy, so a key
lifted from 2^15 carries the LWE problem of 2^15, and the conjugate-invariant ring halves it again
because only half the coefficients are sampled — giving an effective dimension of 2^14 and a budget
of 430 bits, against a ~105-bit modulus chain.

The 430-bit figure comes from a uniform-ternary (hw = 0) table whose 2^13–2^15 entries originate in
HEaven's `maxBitsPolicy128()` and whose 2^16–2^17 entries cite ePrint 2024/463. **Those numbers are
provisional here, the hamming weight may change, and the lifted-key construction in particular has
not yet been reviewed.** Everything a review would need to change is confined to one block of
[`include/mlp_params.hpp`](include/mlp_params.hpp).

This section must be replaced with a finalised justification before the submission is considered
complete; per the benchmark rules a parameter claim that cannot be justified is recorded as a gap
rather than asserted.

---

## Licence

Submission code: Apache-2.0, as the repository. It uses only HEaaN2's **public API**; no HEaaN2
source is vendored here. HEaaN2 itself is proprietary to Crypto Lab Inc. and must be obtained
separately.
