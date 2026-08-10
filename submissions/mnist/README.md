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
| GPU | CUDA device, compute capability ≥ 7.5. Developed on an RTX 5090 (sm_120) |
| CUDA | **≥ 12.8** — sm_120 (Blackwell) is not supported by earlier toolkits |
| HEaaN2 | branch **`hem-heaan-mlinf`** (v0.2.0 line), built **with CUDA**. Private; see [access](#access). Not `dev` — see [the branch requirement](#the-heaan2-branch) |
| Toolchain | CMake ≥ 3.23, GCC 14 (C++17), OpenMP, gperftools (`libtcmalloc`), OpenBLAS **including headers** (`cblas.h`) |
| Python | the repo's `requirements.txt` (torch 2.9.1, torchvision 0.24.1, numpy, absl-py) |

The toolchain row is satisfied in one step by HEaaN2's own conda environment — see step 3. Do not
assemble it by hand; the GCC and CUDA versions have to match each other, and the failure when they
do not is [obscure](#troubleshooting).

### Access

Three private Crypto Lab repositories are needed, and **all three are fetched over SSH**:

| Repo | How it is obtained |
| --- | --- |
| `CryptoLabInc/HEaaN2` | you clone it (step 2) |
| `CryptoLabInc/HEaven` | CPM fetches it during the HEaaN2 build |
| `CryptoLabInc/hem` | CPM fetches it during the HEaaN2 build |

You need a GitHub account with access to the `CryptoLabInc` org and a working SSH key
(`ssh -T git@github.com` should greet you by name). CPM requests the two transitive deps over
**https**, which cannot prompt for a password in a non-interactive build, so `scripts/build_task.sh`
detects a usable SSH key and rewrites those URLs for the duration of the build. The rewrite is
scoped to that one invocation via `GIT_CONFIG_*` — **your global git config is not modified.**

### Replicating from a fresh clone

```bash
# 1. the benchmark repo
git clone git@github.com:yongwonchoi-Cryptolab/ml-inference.git
cd ml-inference
git checkout heaan2-mnist-submission

# 2. HEaaN2, anywhere you like. The branch is not optional — the default (dev)
#    does not compile this submission. If you already have a checkout, point
#    HEAAN2_ROOT at it instead of cloning, but see "The HEaaN2 branch" below.
git clone -b hem-heaan-mlinf git@github.com:CryptoLabInc/HEaaN2.git ~/HEaaN2
export HEAAN2_ROOT=~/HEaaN2

# 3. the toolchain: CUDA 12.8.1 + GCC 14.3 + CMake + ninja + gperftools + BLAS headers,
#    pinned together. This is the step that makes the CUDA build work.
conda env create -f $HEAAN2_ROOT/conda/heaven-dev-cuda.yml
conda activate heaven-dev-cuda

# 4. python deps, in a venv layered on top (activate it *after* conda)
python -m venv bmenv && source ./bmenv/bin/activate
pip install -r requirements.txt

# 5. build. HEaaN2 is configured in $HEAAN2_ROOT/build/ml-inference and installed
#    into $HEAAN2_ROOT/install on first use, then the submission is built against
#    it. Takes a while.
./scripts/build_task.sh ./submissions/mnist

# 6. run
python3 harness/run_submission.py 0 --seed 3
```

**Activation order in steps 3–4 matters.** The venv must come second so that `python3` resolves to
it: the harness launches every stage through `subprocess.run(["python3", ...])`, which goes through
`PATH`, so a venv that is merely *created* but not active leaves those children on the system
interpreter — which has no `torch`. With both active, `python3` is the venv's and `gcc` is conda's,
which is what you want. (`nvcc` is the one exception: a system CUDA already on `PATH` can outrank
the env's — see [the note under Environment variables](#environment-variables).)

Step 5 is optional in practice — the harness runs `build_task.sh` itself on every invocation — but
running it once on its own keeps build errors separate from run errors.

### The HEaaN2 branch

`$HEAAN2_ROOT` must be on **`hem-heaan-mlinf`**, at or after `bc4b0e1` (PR #218, 2026-08-07). That
merge is what adds

```cpp
void frobMap(const ICiphertext &op, i32 pow, ICiphertext &res) const;   // include/HEaaN2/HomEval.hpp
```

— the bare Galois automorphism the [key-less fold](#the-key-less-fold) is built on, together with
the `SKGenerator` documentation of the lifted-key invariance period it requires. It is **not** on
`dev` and not on the `0.2.0` tag, and no other public entry point substitutes for it: `HomEval`'s
other automorphisms (`rot`, `conj`) all take a switching key. On any other branch the submission
fails to compile at [`src/mlp_pipeline.cpp:289`](src/mlp_pipeline.cpp#L289).

**Reusing a checkout you already work in.** Supported, with two caveats beyond the branch. First,
the build tree: `$HEAAN2_ROOT/build` is what HEaaN2's own `CMakePresets.json` uses as its
`binaryDir`, and the presets configure with **Ninja** while this script configures with the default
generator. To keep the two from fighting over one CMake cache, `build_task.sh` builds in
`$HEAAN2_ROOT/build/ml-inference` (inside HEaaN2's `.gitignore`, but a separate cache) and never
touches yours — override with `HEAAN2_BUILD_DIR` if you want it elsewhere. Second, the **checkout is
built as it stands**: whatever branch, commit and uncommitted edits are in `$HEAAN2_ROOT` are what
the submission links against, so confirm it before quoting any number from a run.

```bash
git -C $HEAAN2_ROOT fetch origin                     # local branches go stale
git -C $HEAAN2_ROOT log -1 --oneline origin/hem-heaan-mlinf
git -C $HEAAN2_ROOT status --short                   # should be empty
```

### Running the other instance sizes

```bash
python3 harness/run_submission.py 0 --seed 3     # single
python3 harness/run_submission.py 1 --seed 3     # small (100)
python3 harness/run_submission.py 2 --seed 3     # medium (1000)
python3 harness/run_submission.py 3 --seed 3     # large (10000)
```

**On a shared Slurm node, prefix each with `srun`** (or an equivalent allocation). The harness
invokes each stage binary through `subprocess.run`, and those children inherit the allocation, so
one `srun` around the whole harness is enough — no harness change is needed. Run bare on a node
with no CUDA device visible, stage 7 aborts with *"CUDA device is not available in the current
environment"*.

### Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `HEAAN2_ROOT` | `$HOME/HEaaN2` | HEaaN2 checkout; built and installed on first use |
| `HEAAN2_BUILD_DIR` | `$HEAAN2_ROOT/build/ml-inference` | Where HEaaN2 is *configured*. Deliberately not `$HEAAN2_ROOT/build`, which HEaaN2's own presets own |
| `HEAAN2_DIR` | `$HEAAN2_ROOT/install` | Use an **existing** HEaaN2 install and skip building it |
| `HEAAN2_BUILD_CUDA` | `ON` | `OFF` gives a CPU build — works, but is not what this submission is measured on |
| `HEAAN2_GIT_SSH` | `1` | `0` disables the https→SSH rewrite (use if you have a credential helper) |
| `HEAAN2_NVCC` | auto | Path to `nvcc`, if detection picks the wrong one |
| `HEAAN2_CUDA_HOST_COMPILER` | the `g++` beside `nvcc` | Host compiler nvcc drives |

`scripts/build_task.sh` locates `nvcc` by checking `CUDACXX`, then `$CONDA_PREFIX/bin`, then `PATH`,
then its own `CMakeCache.txt`, so it works whether or not the conda env is active. The conda prefix
is checked ahead of `PATH` on purpose: activating an env prepends its `bin` to `PATH`, but a system
`/usr/local/cuda-*/bin` exported from `/etc/profile.d` or a login profile can already sit ahead of
it, so `command -v nvcc` may report the toolkit `heaven-dev-cuda` was activated to override. The
line to check in the configure output is `Found CUDAToolkit` — it should name a path under
`$CONDA_PREFIX` and version 12.8.x.

`scripts/get_openfhe.sh` is left untouched and still runs: `submissions/cifar10` depends on
OpenFHE, and the harness calls the script unconditionally. It is not used by this submission.

### Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `Compiling the CUDA compiler identification source file "CMakeCUDACompilerId.cu" failed`, with `error: identifier "__is_array" is undefined` in `type_traits` | nvcc probed one `gcc` for a version but preprocessed with another, so one GCC's `libstdc++` headers got parsed in the other's language mode and its newer builtins came out undefined. nvcc prepends its own `bin/` to `PATH` before invoking the host compiler, so this appears whenever the conda env is *not* active: it probes the system `gcc` and then runs conda's. The build script pins the host compiler automatically; if it still happens, set `HEAAN2_CUDA_HOST_COMPILER` to the `g++` from `heaven-dev-cuda`. |
| `'const class heaan::HomEval' has no member named 'frobMap'` at `src/mlp_pipeline.cpp:289` | `$HEAAN2_ROOT` is not on `hem-heaan-mlinf`, or is on a stale local copy of it. `frobMap` landed in `bc4b0e1` (PR #218) and exists on no other branch — see [The HEaaN2 branch](#the-heaan2-branch). `git -C $HEAAN2_ROOT fetch origin` first: a local branch of that name can predate the merge and shows no `frobMap` even when the remote has it. Then delete `$HEAAN2_ROOT/install` so the next build re-installs the right headers. |
| `Error: generator : Ninja` / `Does not match the generator used previously: Unix Makefiles`, raised from `FetchContent.cmake` via `CPM_*.cmake` and `external/CMakeLists.txt` | A CMake cache in `$HEAAN2_ROOT` left by a *different* configuration — almost always HEaaN2's own `CMakePresets.json`, whose `binaryDir` is `$HEAAN2_ROOT/build` and whose generator is Ninja. CPM's sub-builds inherit the parent generator but keep their own caches, so the clash surfaces one level down, inside the `HEaven` fetch, not at the top. Fixed in `build_task.sh`, which now builds in `$HEAAN2_ROOT/build/ml-inference`; if you pinned `HEAAN2_BUILD_DIR` at a tree someone else configured, point it somewhere private or delete it. Note the same stale cache also pins `CMAKE_CXX_COMPILER` and `CUDAToolkit_ROOT`, so `Found CUDAToolkit … 12.4` and a `/usr/bin/g++` in the "variables have changed" list are symptoms of it, not separate faults. |
| `fatal: could not read Username for 'https://github.com'` while cloning `HEaven` or `hem` | No usable SSH key, or no access to the `CryptoLabInc` org. Check `ssh -T git@github.com`. |
| `Could not find nvcc executable in any searched paths, please set CUDAToolkit_ROOT` | `nvcc` is not on `PATH` and was not auto-detected. Activate `heaven-dev-cuda`, or set `HEAAN2_NVCC`. |
| A wall of `undefined reference to 'cuda…@libcudart.so.12'` / `'cublas…@libcublas.so.12'` at link | A stale `submissions/mnist/build` from before the shared-CUDA fix. `rm -rf submissions/mnist/build` and rebuild. |
| `error while loading shared libraries: libcudart.so.12` when a stage runs | Stage binaries built without `DT_RPATH`. `rm -rf submissions/mnist/build` and rebuild; as a stopgap, export `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`. |
| `ModuleNotFoundError: No module named 'torch'` from `generate_dataset.py` | The venv is not active, so the harness's `python3` children fall back to the system interpreter. `source ./bmenv/bin/activate`. |
| `CUDA device is not available in the current environment` in stage 7 | No GPU visible. Use `srun` or an equivalent allocation. |

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
