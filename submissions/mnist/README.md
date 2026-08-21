# Workload implementation — MNIST ML inference

A submission for the `ml-inference` workload (`--dataset mnist`) by CryptoLab, Inc., written in
C++, using the pre-release [HEaaN2](https://heaan.io) CKKS library.

Every instance size evaluates the model with a single **PCMM** (GEMM-based) circuit. PCMM needs no
Every instance size evaluates the model with a single, **PCMM** based circuit. Note that the circuit is a short sequence of two PCMMs and a multiplication and does not require any bootstrap.

> **Hardware.** An **NVIDIA sm_120 (Blackwell) GPU is required** — the vendored library targets
> sm_120 only, with no PTX fallback. 

## Model and circuit

```
fc1 (128×484, +b1)  →  x²  →  fc2 (10×128, +b2)
```

`x²` is the activation with which the
network was trained, following the idea of AESPA (Park et al.,
[arXiv:2201.06699](https://arxiv.org/abs/2201.06699)). Weights are row-major CSV in
[`weights/`](weights/). Folded plaintext accuracy is 97.96%.

### Cleartext pre- and post-processing

Everything the **model** computes runs on ciphertext. The cleartext steps:

| Step | Side | Note |
| --- | --- | --- |
| normalization | client, pre-encryption | Harness writes pixels already in [0,1]; matches the harness's own model (`harness/mnist/test.py:60`) |
| center-crop 28×28 → 22×22 | client, pre-encryption | trims 3 pixels per side before encrypting |
| argmax over 10 logits | client, post-decryption | |
| BatchNorm folded into fc1 | offline, model-only | standard eval-mode folding at weight export; does not touch the input |

## Scheme

One circuit at every size. Sizes 0–2 and size 3 are **tuned separately**
(`mlp::pcmm::profile(size)`): the batch size decides how many blocks a matrix row splits into, and
that decides which ring and modulus chain come out cheapest.

### PCMM

Feature = ciphertext **row** (the GEMM contraction dimension), image = **column/slot**. Sizes 0–2
share one profile because each fits a single block and does identical work.

Written on the plaintext matrices, PCMM is the model at the top of this page. `buildModel` and
`inference` in [`src/mlp_pcmm.cpp`](src/mlp_pcmm.cpp) follow it line by line:

```
U1 = [ W1 | b1 ]        128 × 485   fc1 weights, b1 folded in as an extra column
X  = [ x_1 … x_n ; 1 ]  485 × n     one image per column, a constant-1 row appended

H1 = U1 · X              128 × n    W1·x_i + b1 for every image i              (fc1, +b1)
H2 = H1 ⊙ H1             128 × n    elementwise square               (the x² activation)
Y  = W2 · H2 + b2         10 × n    fc2, then b2 added to every column         (fc2, +b2)
```

`U1` and `W2` are the plaintext matrices `model.fc1_weights`/`model.fc2_weights` encode;
`X` and `Y` are the ciphertext matrices `cx`/`cy`. Each line is one step in the code — a `pcmm`
GEMM, a square, or an bias addition, with `U1`, `W2` and `b2` staying plaintext.

| | sizes 0–2 (1, 100, 1000) | size 3 (10000) |
| --- | --- | --- |
| Ring | N = 2^12, **plain ring** | N = 2^15, **CI subring** |
| Coefficients / message | 4096 (the full degree) | 16384 (CI's free half) |
| Images / message | 2048 | 16384 |
| Modulus chain | 34 + 3×24 ≈ 106 bits, 4 levels | 44 + 3×27 ≈ 125 bits, 4 levels |
| Secret key | uniform ternary (hw = 0), sampled directly at 2^12 — no lifting | uniform ternary (hw = 0), sampled directly at 2^15 — no lifting |
| Noise / SWK budget | σ = 3.2 / `maxBits128(12) = 106` bits, margin 5.0 | σ = 3.2 / `maxBits128(14) = 430` bits, margin 5.0 |

```
L3  encrypt X, encode U1 → fc1 GEMM + rescale → L2
L2  x²: square → rescale → L1, relinearize @L1
L1  encode U2 → fc2 GEMM + rescale            → L0
L0  +b2 (plaintext add, no level cost), decrypt
```

## Security and parameters

| Parameter | PCMM (sizes 0–2) | PCMM (size 3) |
| --- | --- | --- |
| Ring degree | 2^12 | 2^15, CI subring |
| LWE dimension | 2^12 | 2^14 |
| Secret key | uniform ternary (hw = 0) | uniform ternary (hw = 0) |
| Error distribution | Discrete Gaussian (σ = 3.2) | Discrete Gaussian (σ = 3.2) |
| `log(QP)` | 106 | 158 |

According to Table 5.2 of [[BCC+24]](https://doi.org/10.62056/anxra69p1), which bounds `log(q)`
to 430 for dimension 2^14 and 106 for 2^12 under a uniform-ternary secret, these configurations
provide 128 bits of security in the IND-CPA model. The bound is on `q = PQ`, the ciphertext
modulus together with the additional modulus `P` used for relinearization, not on the ciphertext
chain alone — `log(QP)` above is what the bound is checked against.

`hw = 0` means the secret is uniform ternary rather than sparse, so Table 5.2's ternary column
applies directly. Under the conjugate-invariant
subring the secret has half the free coefficients of the full ring, so the CI configurations are
read at dimension 2^14 rather than 2^15; `mlp::pcmm::logRlweDim` performs that halving, and its
result is what reaches `mlp::maxBits128`.

[BCC+24] Bossuat et al., *Security guidelines for implementing homomorphic encryption.* IACR
Communications in Cryptology, 1(4):26, 2024.

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

**Run.** From the repository root:

```console
python3 harness/run_submission.py 0 --seed 3     # single (1 image)
python3 harness/run_submission.py 1 --seed 3     # small  (100)
python3 harness/run_submission.py 2 --seed 3     # medium (1000)
python3 harness/run_submission.py 3 --seed 3     # large  (10000)
```

Every run overwrites `measurements/<size>/results-<n>.json`. 

## Executables

Every stage binary takes `<size>` as its only argument; the size selects the PCMM profile.

| Executable | What it does |
| --- | --- |
| `client_key_generation` | Generates sk and a relinearization key. No rotation keys. |
| `server_preprocess_model` | Caches the raw CSV weight shapes. |
| `client_preprocess_input` | Normalizes and center-crops, in the clear. |
| `client_encode_encrypt_input` | Packs the whole batch into one matrix, encrypts under the **secret** key. |
| `server_encrypted_compute` | Runs the entire inference on ciphertext, and reports its own timing breakdown. |
| `client_decrypt_decode` | Decrypts and reads the column for image `i` of row `r`. |
| `client_postprocess` | argmax over the 10 logits, in the clear. |

## Results

Seed 3, on one NVIDIA RTX 5090 (sm_120). Every figure is the mean of the three runs committed
under [`measurements/`](../../measurements/), taken with the `Encrypted computation` warm-up and
timer synchronization in place. The [architecture check](#-this-submission-requires-an-sm_120-gpu)
is a precondition for quoting any timing here: on a mismatched GPU the first run absorbs several
seconds of just-in-time compilation and reports it as evaluation time.

| | size 0 (1) | size 1 (100) | size 2 (1000) | size 3 (10000) |
| --- | --- | --- | --- | --- |
| Harness `Encrypted model preprocessing` | 0.079 s | 0.065 s | 0.072 s | 0.071 s |
| Harness `Encrypted computation` | 0.417 s | 0.441 s | 0.424 s | 0.485 s |
| ├─ model setup | 0.058 s | 0.057 s | 0.053 s | 0.073 s |
| ├─ warm-up (discarded) | 0.031 s | 0.035 s | 0.033 s | 0.015 s |
| └─ evaluation | **0.57 ms** | **0.57 ms** | **0.57 ms** | **2.32 ms** |
| Public + evaluation keys | 108.5K | 108.5K | 108.5K | 320.5K |
| Encrypted input | 51.2M | 51.2M | 51.2M | 242.5M |
| Encrypted results | 350.1K | 350.1K | 350.1K | 1.8M |
| **Accuracy** | **PASS** | **0.98** | **0.989** | **0.9796** |
| Harness plaintext model | n/a | 0.96 | 0.981 | 0.9779 |

Sizes 0–2 report identical bandwidth because PCMM encrypts the batch as a fixed 485 × 4096
ciphertext matrix: one image occupies one column and the remaining 4095 are padding. The upload is
therefore sized by the profile, not by the batch — efficient at 1000 images, heavily
over-provisioned at one.

The indented sub-rows do not sum to `Encrypted computation` above them: the harness times that
stage as a whole process, so the harness figure also carries ~0.36 s of interpreter and
CUDA-context startup and ciphertext I/O that sits outside the submission's own timers. The harness
plaintext row is the harness's own model on the same subset — the encrypted model scores at or
above it at every size.

## License

Submission code (`src/`, `include/`, `CMakeLists.txt`, `weights/`) is Apache-2.0, as the
repository. It uses only HEaaN2's public API — no HEaaN2 implementation source is in this repo.

[`install/`](install/) carries a prebuilt HEaaN2 (public headers + `libheaan2.so.0.2.0`),
proprietary to CryptoLab, Inc. and not Apache-2.0. It is redistributed under [LICENSE](LICENSE),
which permits use solely for reproducing and verifying benchmark results. Its dependencies are
listed in [LICENSE-THIRD-PARTY](LICENSE-THIRD-PARTY).
