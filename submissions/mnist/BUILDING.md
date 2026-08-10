# Building and running

Everything needed to get the HEaaN2 MNIST submission compiled and through the harness. For what it
computes and why, see [DESIGN.md](DESIGN.md); for what it is, [README.md](README.md).

---

## Requirements

| | |
| --- | --- |
| GPU | CUDA device, compute capability ≥ 7.5. Measured on an RTX 5090 (sm_120); also built and run on RTX 4090 (sm_89) — see [which machine](#which-machine) |
| CUDA | **≥ 12.8** — sm_120 (Blackwell) is not supported by earlier toolkits. The *driver* may be older (a 550.x/CUDA-12.4 driver is fine) **provided** `HEAAN2_CUDA_ARCH` names your GPU, so nothing has to JIT PTX |
| HEaaN2 | branch **`hem-heaan-mlinf`** (v0.2.0 line), built **with CUDA**. Private; see [access](#access). Not `dev` — see [the branch requirement](#the-heaan2-branch) |
| Toolchain | CMake ≥ 3.23, GCC 14 (C++17), OpenMP, gperftools (`libtcmalloc`), OpenBLAS **including headers** (`cblas.h`) |
| Python | the repo's `requirements.txt` (torch 2.9.1, torchvision 0.24.1, numpy, absl-py) |

The toolchain row is satisfied in one step by HEaaN2's own conda environment — see step 3. Do not
assemble it by hand; the GCC and CUDA versions have to match each other, and the failure when they
do not is [obscure](#troubleshooting).

## Access

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

## Replicating from a fresh clone

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
#    it. Takes a while. Set HEAAN2_CUDA_ARCH to your GPU to make it much shorter.
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

## The HEaaN2 branch

`$HEAAN2_ROOT` must be on **`hem-heaan-mlinf`**, at or after `bc4b0e1` (PR #218, 2026-08-07). That
merge is what adds

```cpp
void frobMap(const ICiphertext &op, i32 pow, ICiphertext &res) const;   // include/HEaaN2/HomEval.hpp
```

— the bare Galois automorphism [the key-less fold](DESIGN.md#the-key-less-fold) is built on,
together with the `SKGenerator` documentation of the lifted-key invariance period it requires. It is
**not** on `dev` and not on the `0.2.0` tag, and no other public entry point substitutes for it:
`HomEval`'s other automorphisms (`rot`, `conj`) all take a switching key. On any other branch the
submission fails to compile at [`src/mlp_pipeline.cpp:289`](src/mlp_pipeline.cpp#L289).

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

## Running the instance sizes

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
environment"*. On a machine you have to yourself, run bare.

Every harness run overwrites `measurements/<size>/results-<n>.json`, which still holds the
**reference OpenFHE** submission's numbers. Restore them with `git checkout -- measurements/` after
a development run, and delete any `measurements/large/` the size-3 run creates.

## Which machine

Three machines are in use, and only one of them produces numbers worth quoting:

| Machine | GPU | How to run | `HEAAN2_CUDA_ARCH` |
| --- | --- | --- | --- |
| **Benchmark server** | 1× RTX 5090 (sm_120), exclusive to one developer at a time | bare | `120-real` |
| Developer server | 1× RTX 5090 (sm_120), shared | under `srun` | `120-real` |
| Developer server | 8× RTX 4090 (sm_89) | bare, no Slurm | `89-real` |

The two developer servers exist to confirm the **build** is correct. Timings taken on them are not
comparable to the bench server or to each other — at size 0 the 4090 box measures ~3× the 5090's
`Encrypted computation`, entirely from hardware. Every figure in
[DESIGN.md §4](DESIGN.md#4-results) comes from a 5090; **all official measurements are taken on the
bench server.**

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `HEAAN2_ROOT` | `$HOME/HEaaN2` | HEaaN2 checkout; built and installed on first use |
| `HEAAN2_BUILD_DIR` | `$HEAAN2_ROOT/build/ml-inference` | Where HEaaN2 is *configured*. Deliberately not `$HEAAN2_ROOT/build`, which HEaaN2's own presets own |
| `HEAAN2_DIR` | `$HEAAN2_ROOT/install` | Use an **existing** HEaaN2 install and skip building it |
| `HEAAN2_BUILD_CUDA` | `ON` | `OFF` gives a CPU build — works, but is not what this submission is measured on |
| `HEAAN2_CUDA_ARCH` | `75-real;80-real;89-real;120-real` | `CMAKE_CUDA_ARCHITECTURES` for HEaaN2 and its CUDA deps. Narrow it to your GPU for a much faster build — see [which machine](#which-machine). Must be passed explicitly — see below |
| `HEAAN2_GIT_SSH` | `1` | `0` disables the https→SSH rewrite (use if you have a credential helper) |
| `HEAAN2_NVCC` | auto | Path to `nvcc`, if detection picks the wrong one |
| `HEAAN2_CUDA_HOST_COMPILER` | the `g++` beside `nvcc` | Host compiler nvcc drives |

**`$HEAAN2_ROOT/install` short-circuits the whole HEaaN2 build.** `build_task.sh` skips it whenever
`$HEAAN2_INSTALL/lib/cmake/HEaaN2/HEaaN2Config.cmake` exists, so changing the branch,
`HEAAN2_CUDA_ARCH` or `HEAAN2_BUILD_CUDA` has **no effect** until that tree is deleted. This is the
single most common way to spend an hour re-running a build that never changed.

`scripts/build_task.sh` locates `nvcc` by checking `CUDACXX`, then `$CONDA_PREFIX/bin`, then `PATH`,
then its own `CMakeCache.txt`, so it works whether or not the conda env is active. The conda prefix
is checked ahead of `PATH` on purpose: activating an env prepends its `bin` to `PATH`, but a system
`/usr/local/cuda-*/bin` exported from `/etc/profile.d` or a login profile can already sit ahead of
it, so `command -v nvcc` may report the toolkit `heaven-dev-cuda` was activated to override. The
line to check in the configure output is `Found CUDAToolkit` — it should name a path under
`$CONDA_PREFIX` and version 12.8.x.

**Why `HEAAN2_CUDA_ARCH` has to be passed at all.** HEaven picks a sensible architecture list only
when the cache variable is unset:

```cmake
if(DEFINED CACHE{CMAKE_CUDA_ARCHITECTURES})
  set(TARGET_CUDA_ARCHS "${CMAKE_CUDA_ARCHITECTURES}")
else()
  set(TARGET_CUDA_ARCHS "75-real;80-real;89-real;120-real")   # unreachable
endif()
```

CMake seeds that cache entry with the compiler default (`52`) the moment CUDA is enabled, so the
`else` never runs and the stack silently builds for **sm_52 only**. Every real GPU then JIT-compiles
the embedded `compute_52` PTX at load, which is both slow and fragile — see the
`cudaErrorUnsupportedPtxVersion` row in [Troubleshooting](#troubleshooting). HEaaN2's own presets
pass the list explicitly, which is why a preset build does not show this; `build_task.sh` now does
the same. Confirm what you got with:

```bash
cuobjdump --list-elf $HEAAN2_ROOT/install/lib/libheaan2.so.0.2.0 | grep -o 'sm_[0-9]*' | sort -u
```

`scripts/get_openfhe.sh` is left untouched and still runs: `submissions/cifar10` depends on
OpenFHE, and the harness calls the script unconditionally. It is not used by this submission.

## Troubleshooting

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
| `cudaErrorUnsupportedPtxVersion, "the provided PTX was compiled with an unsupported toolchain"` at `CudaMemoryResource.cu`, aborting a stage with SIGABRT | No cubin for your GPU, so the driver fell back to JIT-compiling `compute_52` PTX — and the driver is older than the toolkit that emitted it (`nvidia-smi`'s `CUDA Version` is the driver's ceiling; conda's nvcc is 12.8, a 550.x driver caps at 12.4). Rebuild with `HEAAN2_CUDA_ARCH` covering your GPU, which emits a native cubin and removes the JIT entirely — a cubin runs on any driver of the same CUDA major version. **Delete `$HEAAN2_ROOT/install` first**: `build_task.sh` skips the HEaaN2 build whenever that tree exists, so an architecture change otherwise has no effect. |
| `no kernel image is available for execution on the device` | Same cause, different symptom: cubins exist but none match your GPU, and there is no PTX to fall back to (`-real` suppresses it). Add your architecture to `HEAAN2_CUDA_ARCH` and rebuild, again deleting `$HEAAN2_ROOT/install` first. |
