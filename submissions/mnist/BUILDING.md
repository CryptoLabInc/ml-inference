# Building and running

What it computes and why is in [DESIGN.md](DESIGN.md).

## Requirements

A prebuilt HEaaN2 (public headers + `libheaan2.so.0.2.0`) is vendored in [`install/`](install/), so
**the default build needs no HEaaN2 source, no private-repo access and no SSH key.**

| | |
| --- | --- |
| GPU | CUDA device. The vendored binary is **sm_120 only** — see the check below |
| CUDA | ≥ 12.8 runtime (`libcudart.so.12`, `libcublas.so.12`) on the library path |
| Toolchain | CMake ≥ 3.23, GCC 14 (C++17), OpenMP, gperftools (`libtcmalloc`) |
| Python | the repo's `requirements.txt` |

If you have a HEaaN2 checkout, `conda env create -f $HEAAN2_ROOT/conda/heaven-dev-cuda.yml` gives
the whole toolchain at matching versions. Mismatched GCC/CUDA fails
[obscurely](#troubleshooting) — don't assemble it by hand.

> ### ⚠ Verify the GPU architecture before trusting any timing
>
> ```bash
> cuobjdump --list-elf submissions/mnist/install/lib/libheaan2.so.0.2.0 \
>   | grep -oE 'sm_[0-9]+' | sort -u          # expect: sm_120
> ```
>
> **`sm_52` means the library is mis-built** — that is CMake's default, which HEaven's fallback
> silently produces when `CMAKE_CUDA_ARCHITECTURES` is left to the cache (see
> [below](#why-heaan2_cuda_arch-must-be-passed)). Such a build still *runs* anywhere by JIT-compiling
> its `compute_52` PTX at load, but the first run pays seconds of JIT, later runs silently depend on
> `~/.nv/ComputeCache`, and the code never uses the target architecture. **Timings from an sm_52
> build are not valid benchmark numbers** — this already cost us one retracted measurement
> ([DESIGN.md](DESIGN.md#retraction)).
>
> **If the bench server is not sm_120**, rebuild and replace `install/`
> ([below](#rebuilding-heaan2-from-source)). An sm_120-only library will not run on sm_89/80/75.

## Quick start

```bash
git clone git@github.com:yongwonchoi-Cryptolab/ml-inference.git && cd ml-inference
git checkout heaan2-mnist-submission

python -m venv bmenv && source ./bmenv/bin/activate
pip install -r requirements.txt

./scripts/build_task.sh ./submissions/mnist      # optional; the harness runs it anyway
python3 harness/run_submission.py 0 --seed 3
```

Using conda for the toolchain? **Activate conda first, the venv second.** The harness spawns every
stage via `subprocess.run(["python3", ...])`, resolved through `PATH`; a venv that is created but
not active leaves those children on an interpreter with no `torch`.

## Running the instance sizes

```bash
python3 harness/run_submission.py 0 --seed 3     # single (1 image)
python3 harness/run_submission.py 1 --seed 3     # small  (100)
python3 harness/run_submission.py 2 --seed 3     # medium (1000)
python3 harness/run_submission.py 3 --seed 3     # large  (10000)
```

**On a shared Slurm node, prefix with `srun`.** One `srun` around the whole harness is enough —
stage binaries are children and inherit the allocation, so no harness change is needed. Without a
visible GPU, stage 7 exits non-zero with *"CUDA device is not available"* and the harness aborts.

Every run overwrites `measurements/<size>/results-<n>.json`, which currently holds the **reference
OpenFHE** numbers. After a development run: `git checkout -- measurements/`, and delete any
`measurements/large/` that a size-3 run created.

## Which machine

| Machine | GPU | Run how | `HEAAN2_CUDA_ARCH` |
| --- | --- | --- | --- |
| **Benchmark server** | 1× RTX 5090 (sm_120), exclusive | bare | `120-real` |
| Developer server | 1× RTX 5090 (sm_120), shared | under `srun` | `120-real` |
| Developer server | 8× RTX 4090 (sm_89) | bare | `89-real` |

Developer servers are for confirming the build. Their timings are not comparable to the bench
server or to each other (the 4090 box measures ~3× the 5090's `Encrypted computation` at size 0,
purely from hardware). **All official measurements are taken on the bench server.**

## Rebuilding HEaaN2 from source

Needed only to target a different GPU architecture, or to build CPU-only. Setting `HEAAN2_ROOT`
switches `build_task.sh` to this path automatically.

```bash
git clone -b hem-heaan-mlinf git@github.com:CryptoLabInc/HEaaN2.git ~/HEaaN2
export HEAAN2_ROOT=~/HEaaN2
conda env create -f $HEAAN2_ROOT/conda/heaven-dev-cuda.yml && conda activate heaven-dev-cuda

export HEAAN2_CUDA_ARCH=89-real          # your GPU
./scripts/build_task.sh ./submissions/mnist

# to make it the new vendored default:
rm -rf submissions/mnist/install && cp -a $HEAAN2_ROOT/install submissions/mnist/install
```

**Access.** Three private CryptoLab repos are involved: `HEaaN2` (you clone it), plus `HEaven` and
`hem`, which CPM fetches during the build. You need `CryptoLabInc` org access and a working SSH key
(`ssh -T git@github.com`). CPM requests the two transitive deps over **https**, which cannot prompt
in a non-interactive build, so `build_task.sh` detects a usable key and rewrites those URLs for the
duration of the build only, via `GIT_CONFIG_*` — **your global git config is untouched.**

**Branch.** A checkout must be on **`hem-heaan-mlinf`** at or after `bc4b0e1` (PR #218). That merge
adds `HomEval::frobMap`, the bare Galois automorphism
[the key-less fold](DESIGN.md#1-circuit-and-parameters) needs; it is on no other branch, and `rot`/`conj`
all require a switching key. Elsewhere the build fails at
[`src/mlp_pipeline.cpp:289`](src/mlp_pipeline.cpp#L289). The vendored `install/` already satisfies
this — built from `hem-heaan-mlinf-frobmap` at `5502b04`.

**Reusing a checkout you work in.** Two caveats. `build_task.sh` configures into
`$HEAAN2_ROOT/build/ml-inference`, *not* `$HEAAN2_ROOT/build`, because the latter belongs to
HEaaN2's own presets (Ninja) and the two caches would fight — override with `HEAAN2_BUILD_DIR`. And
the checkout is **built as it stands**, uncommitted edits included:

```bash
git -C $HEAAN2_ROOT fetch origin && git -C $HEAAN2_ROOT status --short   # should be empty
```

## Environment variables

None are needed by default. They apply only when rebuilding from source.

| Variable | Default | Purpose |
| --- | --- | --- |
| `HEAAN2_ROOT` | *(unset)* | Set it to build HEaaN2 from that checkout instead of using `install/` |
| `HEAAN2_DIR` | `submissions/mnist/install` | Use an existing install; skips building |
| `HEAAN2_BUILD_DIR` | `$HEAAN2_ROOT/build/ml-inference` | Where HEaaN2 is configured |
| `HEAAN2_BUILD_CUDA` | `ON` | `OFF` builds CPU-only — works, but not what this is measured on |
| `HEAAN2_CUDA_ARCH` | `75-real;80-real;89-real;120-real` | `CMAKE_CUDA_ARCHITECTURES`. Narrow to your GPU for a far faster build |
| `HEAAN2_GIT_SSH` | `1` | `0` disables the https→SSH rewrite |
| `HEAAN2_NVCC` | auto | Path to `nvcc` if detection picks the wrong one |
| `HEAAN2_CUDA_HOST_COMPILER` | the `g++` beside `nvcc` | Host compiler nvcc drives |

**An existing install short-circuits the HEaaN2 build.** `build_task.sh` skips building whenever
`$HEAAN2_DIR/lib/cmake/HEaaN2/HEaaN2Config.cmake` exists, so changing branch or
`HEAAN2_CUDA_ARCH` has **no effect** until you delete that tree. Most common way to waste an hour.

`nvcc` is located via `CUDACXX`, then `$CONDA_PREFIX/bin`, then `PATH`, then the existing
`CMakeCache.txt`. The conda prefix comes before `PATH` deliberately: a system `/usr/local/cuda-*/bin`
exported from a login profile can outrank an activated env. Check the `Found CUDAToolkit` line names
a path under `$CONDA_PREFIX` at 12.8.x.

### Why `HEAAN2_CUDA_ARCH` must be passed

HEaven picks a sensible architecture list only when the cache variable is unset:

```cmake
if(DEFINED CACHE{CMAKE_CUDA_ARCHITECTURES})
  set(TARGET_CUDA_ARCHS "${CMAKE_CUDA_ARCHITECTURES}")
else()
  set(TARGET_CUDA_ARCHS "75-real;80-real;89-real;120-real")   # unreachable
endif()
```

CMake seeds that entry with the compiler default (`52`) the moment CUDA is enabled, so the `else`
never runs and the stack builds for **sm_52 only**. HEaaN2's own presets pass the list explicitly,
which is why a preset build never shows this; `build_task.sh` now does the same.

`scripts/get_openfhe.sh` is untouched and still runs on every harness invocation:
`submissions/cifar10` needs OpenFHE. This submission does not.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `no kernel image is available for execution on the device`, or a launch failure on the first homomorphic op | No cubin matches your GPU and `-real` suppressed the PTX fallback. Rebuild with `HEAAN2_CUDA_ARCH` covering it and replace `install/`; **delete `$HEAAN2_DIR` first** or the build is skipped. |
| `cudaErrorUnsupportedPtxVersion` at `CudaMemoryResource.cu`, SIGABRT | An sm_52 build fell back to JIT, and the driver is older than the toolkit that emitted the PTX (`nvidia-smi`'s CUDA version is the driver ceiling; conda's nvcc is 12.8, a 550.x driver caps at 12.4). Rebuild with a real cubin for your GPU — that removes the JIT entirely. |
| `identifier "__is_array" is undefined` while CMake detects the CUDA compiler | nvcc probed one `gcc` but preprocessed with another — it prepends its own `bin/` to `PATH`, so this appears when the conda env is *not* active. `build_task.sh` pins the host compiler; if it persists, set `HEAAN2_CUDA_HOST_COMPILER`. |
| `HomEval has no member named 'frobMap'` at `mlp_pipeline.cpp:289` | `$HEAAN2_ROOT` is not on `hem-heaan-mlinf`, or is a stale local copy. `git fetch origin` first, then delete `$HEAAN2_ROOT/install` so headers are reinstalled. |
| `generator : Ninja / Does not match ... Unix Makefiles` from `FetchContent`/CPM | A CMake cache left by HEaaN2's own presets in `$HEAAN2_ROOT/build`. Fixed by building in `$HEAAN2_ROOT/build/ml-inference`; if you pinned `HEAAN2_BUILD_DIR` at someone else's tree, repoint or delete it. The same stale cache also pins `CMAKE_CXX_COMPILER` and `CUDAToolkit_ROOT`, so a surprising `Found CUDAToolkit … 12.4` is a symptom, not a separate fault. |
| `could not read Username for 'https://github.com'` cloning `HEaven`/`hem` | No usable SSH key or no `CryptoLabInc` access. Check `ssh -T git@github.com`. |
| `Could not find nvcc ... set CUDAToolkit_ROOT` | `nvcc` not on `PATH` and not auto-detected. Activate `heaven-dev-cuda` or set `HEAAN2_NVCC`. |
| `undefined reference to 'cuda…@libcudart.so.12'` at link, or `error while loading shared libraries: libcudart.so.12` at run | Stale `submissions/mnist/build`. `rm -rf` it and rebuild. |
| `ModuleNotFoundError: No module named 'torch'` from `generate_dataset.py` | venv not active, so harness children use the system interpreter. `source ./bmenv/bin/activate`. |
| `CUDA device is not available` in stage 7 | No GPU visible. Use `srun` or an equivalent allocation. |
