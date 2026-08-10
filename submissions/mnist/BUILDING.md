# Building and running

Everything needed to get the HEaaN2 MNIST submission compiled and through the harness. For what it
computes and why, see [DESIGN.md](DESIGN.md); for what it is, [README.md](README.md).

---

## Requirements

**The default build needs no HEaaN2 source, no private-repo access, and no SSH key.** A prebuilt
HEaaN2 install (public headers + `libheaan2.so.0.2.0`) is vendored in this directory under
[`install/`](install/), the same pattern CryptoLab's own
[Zn-multiplication submission](https://github.com/CryptoLabInc/Zn-multiplication/tree/CryptoLabInc)
uses. See [LICENSE](LICENSE) and [LICENSE-THIRD-PARTY](LICENSE-THIRD-PARTY) for the terms it is
redistributed under.

| | |
| --- | --- |
| GPU | CUDA device. The vendored binary is built for **sm_120 only** — see the warning below |
| CUDA | **≥ 12.8** runtime libraries (`libcudart.so.12`, `libcublas.so.12`) on the library path |
| Toolchain | CMake ≥ 3.23, GCC 14 (C++17), OpenMP, gperftools (`libtcmalloc`) |
| Python | the repo's `requirements.txt` (torch 2.9.1, torchvision 0.24.1, numpy, absl-py) |
| HEaaN2 source | **not needed** for the default build; only to [rebuild for another GPU](#rebuilding-heaan2-from-source) |

> ### ⚠ Check the vendored binary's GPU architectures before measuring
>
> `install/lib/libheaan2.so.0.2.0` is compiled for **sm_120 (Blackwell / RTX 5090)**, the
> development machine's GPU. Verify what any copy actually contains before trusting a timing from
> it:
>
> ```bash
> cuobjdump --list-elf submissions/mnist/install/lib/libheaan2.so.0.2.0 \
>   | grep -oE 'sm_[0-9]+' | sort -u        # expect: sm_120
> ```
>
> **If that prints `sm_52`, the library is mis-built** — that is CMake's default architecture, which
> HEaven's fallback logic silently produces when `CMAKE_CUDA_ARCHITECTURES` is left to the cache (see
> [why `HEAAN2_CUDA_ARCH` has to be passed](#environment-variables)). Such a build still *runs*
> anywhere, by JIT-compiling its embedded `compute_52` PTX at load — but the first run on a given
> machine pays seconds of JIT (a measurement error large enough to be mistaken for a performance
> bug), later runs silently depend on `~/.nv/ComputeCache`, and the generated code never uses the
> target architecture's instructions. **Timings from an sm_52 build are not valid benchmark
> numbers.**
>
> **If the bench server's GPU is not sm_120**, rebuild for it and replace `install/` — see
> [Rebuilding HEaaN2 from source](#rebuilding-heaan2-from-source). An sm_120-only library will not
> run natively on an RTX 4090 (sm_89), A100 (sm_80) or T4 (sm_75); it would fall back to PTX JIT
> with the same caveats, if PTX is present at all.

The toolchain row is satisfied in one step by HEaaN2's conda environment if you have a checkout
(`conda env create -f $HEAAN2_ROOT/conda/heaven-dev-cuda.yml`); otherwise any GCC 14 + CUDA 12.8
environment with OpenMP and gperftools works. Do not mix GCC and CUDA versions by hand — the
failure when they disagree is [obscure](#troubleshooting).

## Replicating from a fresh clone

```bash
# 1. the benchmark repo -- this is the only clone needed
git clone git@github.com:yongwonchoi-Cryptolab/ml-inference.git
cd ml-inference
git checkout heaan2-mnist-submission

# 2. python deps
python -m venv bmenv && source ./bmenv/bin/activate
pip install -r requirements.txt

# 3. build against the vendored HEaaN2 in submissions/mnist/install/
./scripts/build_task.sh ./submissions/mnist

# 4. run
python3 harness/run_submission.py 0 --seed 3
```

If you are using conda for the toolchain, **activate conda first and the venv second**: the harness
launches every stage through `subprocess.run(["python3", ...])`, which resolves through `PATH`, so
a venv that is created but not active leaves those children on an interpreter with no `torch`. With
both active in that order, `python3` is the venv's and `gcc` is conda's, which is what you want.

Step 3 is optional in practice — the harness runs `build_task.sh` itself on every invocation — but
running it once on its own keeps build errors separate from run errors.

## Rebuilding HEaaN2 from source

Needed only to target a **different GPU architecture** than the vendored sm_120 binary, or to build
a CPU-only variant. Set `HEAAN2_ROOT` and `build_task.sh` switches to the from-source path
automatically, building and installing HEaaN2 before the submission:

```bash
git clone -b hem-heaan-mlinf git@github.com:CryptoLabInc/HEaaN2.git ~/HEaaN2
export HEAAN2_ROOT=~/HEaaN2
conda env create -f $HEAAN2_ROOT/conda/heaven-dev-cuda.yml && conda activate heaven-dev-cuda

export HEAAN2_CUDA_ARCH=89-real      # e.g. RTX 4090; see the table in Environment variables
./scripts/build_task.sh ./submissions/mnist
```

To make that build the new vendored default, copy its install tree over `install/`:

```bash
rm -rf submissions/mnist/install
cp -a $HEAAN2_ROOT/install submissions/mnist/install
```

### Access, for the from-source path only

Three private Crypto Lab repositories are involved, all fetched over SSH:

| Repo | How it is obtained |
| --- | --- |
| `CryptoLabInc/HEaaN2` | you clone it |
| `CryptoLabInc/HEaven` | CPM fetches it during the HEaaN2 build |
| `CryptoLabInc/hem` | CPM fetches it during the HEaaN2 build |

You need a GitHub account with access to the `CryptoLabInc` org and a working SSH key
(`ssh -T git@github.com` should greet you by name). CPM requests the two transitive deps over
**https**, which cannot prompt for a password in a non-interactive build, so `build_task.sh`
detects a usable SSH key and rewrites those URLs for the duration of the build. The rewrite is
scoped to that one invocation via `GIT_CONFIG_*` — **your global git config is not modified.**

### The HEaaN2 branch

A source checkout must be on **`hem-heaan-mlinf`**, at or after `bc4b0e1` (PR #218, 2026-08-07).
That merge is what adds

```cpp
void frobMap(const ICiphertext &op, i32 pow, ICiphertext &res) const;   // include/HEaaN2/HomEval.hpp
```

— the bare Galois automorphism [the key-less fold](DESIGN.md#the-key-less-fold) is built on,
together with the `SKGenerator` documentation of the lifted-key invariance period it requires. It is
**not** on `dev` and not on the `0.2.0` tag, and no other public entry point substitutes for it:
`HomEval`'s other automorphisms (`rot`, `conj`) all take a switching key. On any other branch the
submission fails to compile at [`src/mlp_pipeline.cpp:289`](src/mlp_pipeline.cpp#L289).

(The vendored `install/` already satisfies this — it was built from `hem-heaan-mlinf-frobmap` at
`5502b04`, which carries `frobMap`. This section matters only if you rebuild.)

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

By default **none of these need to be set** — the build uses the vendored
`submissions/mnist/install/`. They matter only when rebuilding HEaaN2 from source.

| Variable | Default | Purpose |
| --- | --- | --- |
| `HEAAN2_ROOT` | *(unset)* | Setting it switches to the [from-source path](#rebuilding-heaan2-from-source): HEaaN2 is built and installed from this checkout instead of using the vendored install |
| `HEAAN2_BUILD_DIR` | `$HEAAN2_ROOT/build/ml-inference` | Where HEaaN2 is *configured*. Deliberately not `$HEAAN2_ROOT/build`, which HEaaN2's own presets own |
| `HEAAN2_DIR` | `submissions/mnist/install`, or `$HEAAN2_ROOT/install` if `HEAAN2_ROOT` is set | Use an **existing** HEaaN2 install and skip building it |
| `HEAAN2_BUILD_CUDA` | `ON` | `OFF` gives a CPU build — works, but is not what this submission is measured on |
| `HEAAN2_CUDA_ARCH` | `75-real;80-real;89-real;120-real` | `CMAKE_CUDA_ARCHITECTURES` for HEaaN2 and its CUDA deps. Narrow it to your GPU for a much faster build — see [which machine](#which-machine). Must be passed explicitly — see below |
| `HEAAN2_GIT_SSH` | `1` | `0` disables the https→SSH rewrite (use if you have a credential helper) |
| `HEAAN2_NVCC` | auto | Path to `nvcc`, if detection picks the wrong one |
| `HEAAN2_CUDA_HOST_COMPILER` | the `g++` beside `nvcc` | Host compiler nvcc drives |

**An existing install short-circuits the whole HEaaN2 build.** `build_task.sh` skips building
whenever `$HEAAN2_INSTALL/lib/cmake/HEaaN2/HEaaN2Config.cmake` exists, so on the from-source path,
changing the branch, `HEAAN2_CUDA_ARCH` or `HEAAN2_BUILD_CUDA` has **no effect** until
`$HEAAN2_ROOT/install` is deleted. This is the single most common way to spend an hour re-running a
build that never changed.

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
| `no kernel image is available for execution on the device`, or a CUDA launch failure on the first homomorphic op, on a machine that is **not** an RTX 5090 | The vendored `install/lib/libheaan2.so.0.2.0` holds an sm_120 cubin only. Rebuild HEaaN2 with `HEAAN2_CUDA_ARCH` set to your GPU and replace `install/` — see [Rebuilding HEaaN2 from source](#rebuilding-heaan2-from-source). Confirm what a library contains with `cuobjdump --list-elf <lib> \| grep -o 'sm_[0-9]*' \| sort -u`. |
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
