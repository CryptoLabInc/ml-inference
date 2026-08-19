# Workload implementation — MNIST ML inference

This is a submission for the `ml-inference` workload (`--dataset mnist`) by CryptoLab, Inc., written
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

A BatchNorm-folded 2-layer MLP, identical for both schemes. `x²` is the activation with which the
network was **trained**, not a polynomial approximation of ReLU, so the circuit evaluates the model
exactly — the only error is CKKS noise (measured: max |decrypted − plaintext logit| = 0.109 on
logits spanning [−32, +14]). Replacing ReLU with a low-degree polynomial activation follows AESPA (Park et al.,
[arXiv:2201.06699](https://arxiv.org/abs/2201.06699)). Weights are row-major CSV in
[`weights/`](weights/). Folded plaintext accuracy is 97.96%.

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

Written on the plaintext matrices, PCMM computes exactly the model at the top of this page —
`buildModel` and `inference` in [`src/mlp_pcmm.cpp`](src/mlp_pcmm.cpp) are this, line for line:

```
U1 = [ W1 | b1 ]        128 × 485   fc1 weights, b1 folded in as an extra column
X  = [ x_1 … x_n ; 1 ]  485 × n     one image per column, a constant-1 row appended

H1 = U1 · X              128 × n    W1·x_i + b1 for every image i              (fc1, +b1)
H2 = H1 ⊙ H1             128 × n    elementwise square               (the x² activation)
Y  = W2 · H2 + b2         10 × n    fc2, then b2 added to every column         (fc2, +b2)
```

`U1` and `W2` are the plaintext matrices `model.fc1_weights`/`model.fc2_weights` encode; `X` and
`Y` are the ciphertext matrices `cx`/`cy`. Each line above is exactly one step in the code: a
`pcmm` GEMM, a square, or an add, on those same shapes. PCMM does not approximate or restructure
the model — it runs this same computation, just with `X` and `Y` encrypted and `U1`/`W2`/`b2`
staying plaintext, exactly as the server already stores them.

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

The submission uses CKKS with the following configuration:

| Parameter | HS (size 0) | PCMM (sizes 1–2) | PCMM (size 3) |
| --- | --- | --- | --- |
| Ring degree | 2^15, CI subring | 2^12 | 2^15, CI subring |
| LWE dimension | 2^14 | 2^12 | 2^14 |
| Secret key | uniform ternary (hw = 0) | uniform ternary (hw = 0) | uniform ternary (hw = 0) |
| Error distribution | Discrete Gaussian (σ = 3.2) | Discrete Gaussian (σ = 3.2) | Discrete Gaussian (σ = 3.2) |
| `log(QP)` | 227 | 106 | 158 |

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
`Encrypted computation` warm-up and timer synchronization in place, on hardware that passes the
[architecture check](#-this-submission-requires-an-sm_120-gpu). That check is a precondition for
quoting any timing here: on a mismatched GPU the first run absorbs several seconds of just-in-time
compilation and reports it as evaluation time.

| | size 0 (1) HS | sizes 1–2 (100 / 1000) PCMM | size 3 (10000) PCMM |
| --- | --- | --- | --- |
| Harness `Encrypted model preprocessing` | 2.094 s | 0.058 s / 0.070 s | 0.068 s |
| Harness `Encrypted computation` | 0.371 s | 0.416 s / 0.417 s | 0.521 s |
| ├─ model setup | 0.090 s | 0.046 s | 0.058 s |
| ├─ warm-up (discarded) | 0.013 s | 0.028 s | 0.012 s |
| └─ evaluation | **0.70 ms** | **0.80 ms** | **3.40 ms** |
| Public + evaluation keys | 44.9 M | 108.5 K | 320.5 K |
| Encrypted input | 420 K | 51.2 M | 242.5 M |
| Encrypted results | 120 K | 350.1 K | 1.8 M |
| **Accuracy** | PASS | 0.980 / 0.989 | 0.9796 |
| Harness plaintext model | n/a | 0.960 / 0.981 | 0.9779 |

The indented sub-rows do not sum to `Encrypted computation` above them: the harness times that
stage as a whole process, so the harness figure also carries ~0.25 s of interpreter and
CUDA-context startup and ciphertext I/O that sits outside the submission's own timers. The harness
plaintext row is the harness's own model on the same subset — the encrypted model scores at or
above it at every size.

## License

Submission code (`src/`, `include/`, `CMakeLists.txt`, `weights/`) is Apache-2.0, as the
repository. It uses only HEaaN2's public API — no HEaaN2 implementation source is in this repo.

[`install/`](install/) carries a **prebuilt** HEaaN2 (public headers + `libheaan2.so.0.2.0`),
proprietary to CryptoLab, Inc. and **not** Apache-2.0. It is redistributed under [LICENSE](LICENSE),
which permits use solely for reproducing and verifying benchmark results. Its dependencies are
listed in [LICENSE-THIRD-PARTY](LICENSE-THIRD-PARTY).
