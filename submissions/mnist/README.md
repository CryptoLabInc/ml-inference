# Workload implementation — MNIST ML inference

This is a submission for the `ml-inference` workload (`--dataset mnist`) by CryptoLab Inc., written
in C++ and using the [HEaaN2](https://heaan.io) CKKS library. It replaces the reference
OpenFHE/HEIR implementation in this directory; the harness is unmodified. A prebuilt HEaaN2 is
vendored in [`install/`](install/), so the build needs no HEaaN2 source and no private-repo access.

Two circuits evaluate the same model, chosen by instance size alone: a **Halevi–Shoup**
rotation-folded matrix-vector product at size 0, and a **PCMM** (GEMM-based) circuit at sizes 1–3.
No bootstrapping in either.

> **Hardware.** An **NVIDIA sm_120 (Blackwell) GPU is required** — the vendored library targets
> sm_120 only, with no PTX fallback. 

## Model and circuit

```
fc1 (128×484, +b1)  →  x²  →  fc2 (10×128, +b2)
```

A BatchNorm-folded 2-layer MLP, identical for both schemes. `x²` is the activation the network was
**trained** with, not a polynomial approximation of ReLU, so the circuit evaluates the model
exactly — the only error is CKKS noise (measured: max |decrypted − plaintext logit| = 0.109 on
logits spanning [−32, +14]). Weights are row-major CSV in [`weights/`](weights/); shapes,
normalization and provenance in [`weights/manifest.txt`](weights/manifest.txt). Folded plaintext
accuracy is 97.96%.

### Cleartext pre- and post-processing

Everything the **model** computes runs on ciphertext. The cleartext steps:

| Step | Side | Note |
| --- | --- | --- |
| `(p − 0.1307) / 0.3081` normalization | client, pre-encryption | Harness writes pixels already in [0,1]. Same as the reference submission and the harness's own model (`harness/mnist/test.py:60`) |
| **center-crop 28×28 → 22×22** | client, pre-encryption | trims 3 pixels per side before encrypting |
| argmax over 10 logits | client, post-decryption | same as the reference's `client_postprocess` |
| BatchNorm folded into fc1 | offline, model-only | standard eval-mode folding at weight export; does not touch the input |

## Schemes

Selection is by instance size alone (`mlp::usePcmm` in
[`include/mlp_params.hpp`](include/mlp_params.hpp)) — same model, same weights, different packing.

| Size | Scheme | 
| --- | --- | 
| 0 (1 image) | Halevi–Shoup | 
| 1–3 (100, 1000, 10000) | PCMM | 

### Halevi–Shoup (size 0)

| | |
| --- | --- |
| Ring | N = 2^15, conjugate-invariant subring (ePrint 2018/952), 16384 slots |
| Modulus chain | 30 + 3×25 ≈ 105 bits, 4 levels |
| Secret key | uniform ternary (hw = 0), sampled directly at 2^15 |
| Noise / SWK budget | σ = 3.2 / `maxBits128(14) = 430` bits, margin 5.0 |

```
L3  encrypt → fc1 matvec              → L2
L2  fold, +b1, x², rescale            → L1
L1  fc2 matvec                        → L0, +b2, decrypt
```

### PCMM (sizes 1–3)

Feature = ciphertext **row** (the GEMM contraction dimension), image = **column/slot** — the
opposite convention from HS. Sizes 1–2 and size 3 are **tuned separately**: the batch size decides
how many blocks a matrix row splits into, and that decides which ring and which modulus chain come
out cheapest. Sizes 1 and 2 share one profile because both fit a single block and do identical
work. `mlp::pcmm::profile(size)` is the selector.

| | sizes 1–2 (100, 1000) | size 3 (10000) |
| --- | --- | --- |
| Ring | N = 2^12, **plain ring** | N = 2^15, **CI subring** |
| Coefficients / message | 4096 (the full degree) | 16384 (CI's free half) |
| Images / message | 2048 | 16384 |
| Modulus chain | 34 + 3×24 ≈ 106 bits, 4 levels | 44 + 3×27 ≈ 125 bits, 4 levels |
| Secret key | uniform ternary (hw = 0), sampled **directly** at 2^12 — no lifting | uniform ternary (hw = 0), sampled **directly** at 2^15 — no lifting |
| Noise / SWK budget | σ = 3.2 / `maxBits128(12) = 106` bits, margin 5.0 | σ = 3.2 / `maxBits128(14) = 430` bits, margin 5.0 |

```
L3  encrypt X, encode U1 → fc1 GEMM + rescale → L2
L2  x²: square → rescale → L1, relinearize @L1
L1  encode U2 → fc2 GEMM + rescale            → L0
L0  +b2 (plaintext add, no level cost), decrypt
```

## Security and parameters

Every instance targets **128-bit classical security** against a semi-honest server. The claim
rests on **Table 5.2** of [BCC+24], *Security guidelines for implementing homomorphic
encryption* ([IACR CiC 1(4):26](https://cic.iacr.org/p/1/4/26/pdf)), which gives the largest
ciphertext modulus that can be used at each RLWE dimension for a uniform ternary secret and
Gaussian error with σ = 3.19. Its λ = 128, ternary column is transcribed into `mlp::maxBits128`
([`include/mlp_params.hpp`](include/mlp_params.hpp)) for the dimensions this submission can
reach; any other dimension throws rather than guessing:

| RLWE dimension | 2^12 | 2^13 | 2^14 | 2^15 | 2^16 | 2^17 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| max log₂(*q*) | 106 | 214 | 430 | 868 | 1747 | 3523 |

Each scheme stays within its entry:

| | size 0 (HS) | sizes 1–2 (PCMM) | size 3 (PCMM) |
| --- | --- | --- | --- |
| Ring | 2^15, CI subring | 2^12, plain | 2^15, CI subring |
| **RLWE dimension** | **2^14** | **2^12** | **2^14** |
| Ciphertext modulus | 30 + 3×25 = 105 bits | 34 + 3×24 = 106 bits | 44 + 3×27 = 125 bits |
| Table 5.2 budget | 430 bits | 106 bits | 430 bits |

The error distribution is a discrete Gaussian with σ = 3.2 at every size, marginally wider than
the σ = 3.19 the table is estimated for, so the entries apply unchanged — the paper notes that
distributions with standard deviation close to 3.19 yield essentially the same log₂(*q*).

The same budget bounds key switching. `mlp::maxBits128` feeds
`SwKeyGenParamsBuilder::setModUpPrimes(max_bits, margin)`, whose `max_bits` is the total
bit-size budget of the *QP* modulus, so the hybrid key-switching modulus is capped at the
table entry as well, not just the ciphertext chain.

### Secret key: Hamming weight

`HW = 0` selects a **uniform ternary** secret — every coefficient drawn independently and
uniformly from {−1, 0, 1}, with no sparsity constraint. This is exactly the distribution
Table 5.2's `Ternary` column is estimated for, so the table applies with no further argument.

A sparse secret (fixed low Hamming weight, e.g. hw = 32) would admit a larger modulus at the
same dimension and is common in CKKS deployments, but it falls outside this table: sparse keys
need their own estimates, and their concrete security is sensitive to hybrid attacks that
exploit the sparsity. Keeping hw = 0 costs some modulus budget and buys a claim that rests
entirely on the guidelines document.

Both circuits also sample the secret **directly at the working degree** — no key lifting from a
smaller ring — so the dimension in the estimate is the dimension actually used.

### Conjugate-invariant ring

Sizes 0 and 3 work in the **conjugate-invariant subring** ([ePrint 2018/952](https://eprint.iacr.org/2018/952))
rather than the full cyclotomic. Its elements are those fixed by conjugation, so a message of
degree 2^15 is carried by **2^14 RLWE coefficients**, and the slots are real instead of complex.
The submission uses it for the packing win: 16384 real slots per message instead of 16384
complex ones it has no use for.

The security consequence is that **the RLWE dimension is half the message degree**, and the
lookup must use the former. `mlp::pcmm::logRlweDim` performs that halving, and its result — never
the raw `log_degree` — is what reaches `maxBits128`:

```cpp
constexpr u32 logRlweDim(const Profile &p) {
    return p.ntt_alg == heaan::NTTAlgorithm::CYC_FOR_CI ? p.log_degree - 1
                                                        : p.log_degree;
}
```

So sizes 0 and 3 are budgeted at 2^14 → **430 bits**, not at 2^15 → 868 bits. Reading the table
at the message degree would overstate the allowance by a factor of two and is the mistake this
helper exists to prevent. Sizes 1–2 use the plain ring, where degree and RLWE dimension coincide
and the helper is the identity.

## Build and run

| | |
| --- | --- |
| GPU | **NVIDIA sm_120 (Blackwell) only** |
| CUDA | ≥ 12.8 runtime (`libcudart.so.12`, `libcublas.so.12`) on the library path |
| Toolchain | CMake ≥ 3.23, GCC 14 (C++17), OpenMP, gperftools (`libtcmalloc`) |
| Python | the repo's `requirements.txt` |

GCC and CUDA must be a matching pair; a mismatch fails obscurely during CMake's
CUDA compiler detection rather than at your code.

> ### ⚠ This submission requires an sm_120 GPU
>
> The vendored library contains **sm_120 cubins only, and no PTX**. 

**Build.** From the repository root:

```console
python -m venv bmenv && source ./bmenv/bin/activate
pip install -r requirements.txt

./scripts/build_task.sh ./submissions/mnist      # optional; the harness runs it anyway
```

which configures and builds all stage executables against the vendored library:

```console
cmake -S submissions/mnist -B submissions/mnist/build \
      -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_CUDA=ON \
      -DCMAKE_PREFIX_PATH="$PWD/submissions/mnist/install"
cmake --build submissions/mnist/build -j $(nproc)
```

`CMAKE_PREFIX_PATH` must be absolute: CMake resolves a relative one against the *build* directory,
not the invocation directory. Binaries are placed in `submissions/mnist/build/`.

Using conda for the toolchain? **Activate conda first, the venv second.** The harness spawns every
stage via `subprocess.run(["python3", ...])`, resolved through `PATH`; a venv that is created but
not active leaves those children on an interpreter with no `torch`.

**Run.** The submission is driven by the harness. From the repository root:

```console
python3 harness/run_submission.py 0 --seed 3     # single (1 image)
python3 harness/run_submission.py 1 --seed 3     # small  (100)
python3 harness/run_submission.py 2 --seed 3     # medium (1000)
python3 harness/run_submission.py 3 --seed 3     # large  (10000)
```

Every run overwrites `measurements/<size>/results-<n>.json`. 

## Executables

Every stage binary parses `<size>` and dispatches on it, so the harness contract (seven fixed
names, `<size>` as the only argument) is unchanged.

| Executable | HS (size 0) | PCMM (sizes 1–3) |
| --- | --- | --- |
| `client_key_generation` | Generates sk, public encryption key, two sets of rotation keys, fc1 fold keys, relinearization key. | Generates sk and a relinearization key only. |
| `server_preprocess_model` | Caches CSV weights in padded p×q layout and encodes both layers' diagonals. | Same call caches the raw CSV shapes. |
| `client_preprocess_input` | Normalizes and center-crops, in the clear — identical for both schemes. | ← |
| `client_encode_encrypt_input` | Packs 32 images per ciphertext, encrypts under the **public** key. | Packs the whole batch into one matrix, encrypts under the **secret** key. |
| `server_encrypted_compute` | Runs the entire inference on ciphertext, and reports its own timing breakdown. | ← |
| `client_decrypt_decode` | Decrypts and reads logit `r` of image `i` from slot `r*32+i`. | Decrypts and reads the column for image `i` of row `r`. |
| `client_postprocess` | argmax over the 10 logits, in the clear — same output format for both schemes. | ← |

## Results

Seed 3, through the **unmodified harness**, on 1× NVIDIA RTX 5090 (sm_120). Every figure is the
mean of the three runs committed under [`measurements/`](../../measurements/), taken with the
stage-7 warm-up and timer synchronization in place, on hardware that passes the
[architecture check](#-this-submission-requires-an-sm_120-gpu). That check is a precondition for
quoting any timing here: on a mismatched GPU the first run absorbs several seconds of just-in-time
compilation and reports it as evaluation time.

| | size 0 (1) HS | size 1 (100) PCMM | size 2 (1000) PCMM | size 3 (10000) PCMM |
| --- | --- | --- | --- | --- |
| harness `Encrypted model preprocessing` | 2.15 s | — | 0.068 s | 0.071 s |
| harness `Encrypted computation` | 0.41 s | — | 0.44 s | 0.56 s |
| ├─ model setup | 0.116 s | — | 0.051 s | 0.050 s |
| ├─ warm-up (discarded) | 0.017 s | — | 0.032 s | 0.033 s |
| └─ evaluation | **0.76 ms** | — | **0.91 ms** | **2.63 ms** |
| Public + evaluation keys | 44.9 M | — | 54.0 K | 54.0 K |
| Encrypted input | 420 K | — | 44.5 M | 133.6 M |
| Encrypted results | 120 K | — | 280 K | 840 K |
| **Accuracy** | PASS | — | 0.989 | 0.979 |
| Harness plaintext model | — | — | 0.981 | 0.978 |

The stage-7 sub-rows do not sum to the scored figure: the harness times the whole stage-7 *process*,
so it also carries ~0.25 s of interpreter and CUDA-context startup and ciphertext I/O that sits
outside the submission's own timers. Accuracy varies in the third decimal run to run from
encryption noise (size 3 measured 0.9793–0.9798); the table rounds. The harness plaintext row is
the harness's own model on the same subset — the encrypted model scores at or above it at every
size.

## Licence

Submission code (`src/`, `include/`, `CMakeLists.txt`, `weights/`) is Apache-2.0, as the
repository. It uses only HEaaN2's public API — no HEaaN2 implementation source is in this repo.

[`install/`](install/) carries a **prebuilt** HEaaN2 (public headers + `libheaan2.so.0.2.0`),
proprietary to CryptoLab Inc. and **not** Apache-2.0. It is redistributed under [LICENSE](LICENSE),
which permits use solely for reproducing and verifying benchmark results. Its dependencies are
listed in [LICENSE-THIRD-PARTY](LICENSE-THIRD-PARTY).
