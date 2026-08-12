# Building and running

What it computes and why is in [DESIGN.md](DESIGN.md).

## Requirements

A prebuilt HEaaN2 (public headers + `libheaan2.so.0.2.0`) is vendored in [`install/`](install/), so
**the build needs no HEaaN2 source and no private-repo access.**

| | |
| --- | --- |
| GPU | **NVIDIA sm_120 (Blackwell) only** — see the check below before anything else |
| CUDA | ≥ 12.8 runtime (`libcudart.so.12`, `libcublas.so.12`) on the library path |
| Toolchain | CMake ≥ 3.23, GCC 14 (C++17), OpenMP, gperftools (`libtcmalloc`) |
| Python | the repo's `requirements.txt` |

GCC and CUDA must be a matching pair; a mismatch fails [obscurely](#troubleshooting) during CMake's
CUDA compiler detection rather than at your code.

> ### ⚠ This submission requires an sm_120 GPU
>
> The vendored library contains **sm_120 cubins only, and no PTX**. There is no JIT fallback, so it
> will not run on sm_90, sm_89, sm_80 or sm_75 — the first homomorphic operation fails with
> *"no kernel image is available for execution on the device"*.
>
> Check that your GPU and the library agree before running anything:
>
> ```bash
> nvidia-smi --query-gpu=name,compute_cap --format=csv          # expect compute_cap 12.0
>
> cuobjdump --list-elf submissions/mnist/install/lib/libheaan2.so.0.2.0 \
>   | grep -oE 'sm_[0-9]+' | sort -u                            # expect: sm_120
> ```
>
> Both were verified on the hardware the reported measurements were taken on (RTX 5090, compute
> capability 12.0), against the vendored library in this repository.
>
> Retargeting another architecture means rebuilding HEaaN2, which cannot be done outside Crypto Lab
> — see [Provenance](#provenance-of-the-vendored-library). If your benchmark hardware is not
> sm_120, please contact us for a library built for it rather than attempting a workaround: a build
> that JIT-compiles PTX produces multi-second first-run costs and is not a valid source of timings.

## Quick start

```bash
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

**Under a scheduler (Slurm or equivalent), prefix with `srun`.** One `srun` around the whole harness
is enough — stage binaries are children and inherit the allocation, so no harness change is needed.
Without a visible GPU, stage 7 exits non-zero with *"CUDA device is not available"* and the harness
aborts.

Every run overwrites `measurements/<size>/results-<n>.json`. To discard a development run:
`git checkout -- measurements/`, and delete any `measurements/large/` it created.

## Provenance of the vendored library

[`install/`](install/) holds HEaaN2 v0.2.0 built with CUDA for sm_120, with its public headers and
CMake package config. No HEaaN2 implementation source is in this repository.

HEaaN2 is proprietary to Crypto Lab Inc., and it depends on two further non-public libraries, so it
**cannot be rebuilt outside Crypto Lab.** The binary is redistributed here under
[LICENSE](LICENSE), which permits use solely for reproducing and verifying benchmark results, the
same arrangement as CryptoLab's
[Zn-multiplication submission](https://github.com/CryptoLabInc/Zn-multiplication). Third-party
dependencies are listed in [LICENSE-THIRD-PARTY](LICENSE-THIRD-PARTY).

`scripts/build_task.sh` builds against `install/` by default, and that is the only path this
submission is verified through. It retains a from-source path (`HEAAN2_ROOT` and friends) usable
only with access to the HEaaN2 source tree; **verifying this submission never needs it.**

## Environment variables

None are needed. The build works with the environment untouched.

| Variable | Default | Purpose |
| --- | --- | --- |
| `HEAAN2_DIR` | `submissions/mnist/install` | Point at a different HEaaN2 install tree |
| `HEAAN2_NVCC` | auto | Path to `nvcc` if detection picks the wrong one |
| `HEAAN2_CUDA_HOST_COMPILER` | the `g++` beside `nvcc` | Host compiler nvcc drives |

The remaining `HEAAN2_*` variables in `build_task.sh` apply only to the from-source path, which
this submission does not use.

`nvcc` is located via `CUDACXX`, then `$CONDA_PREFIX/bin`, then `PATH`, then an existing
`CMakeCache.txt` — it is needed because the CUDA toolkit is resolved at configure time. The conda
prefix is checked before `PATH` deliberately: a system `/usr/local/cuda-*/bin` exported from a
login profile can otherwise outrank an activated environment. Confirm the `Found CUDAToolkit` line
names a 12.8.x toolkit.

**A stale build tree is sticky.** `build_task.sh` reuses `submissions/mnist/build` and its CMake
cache, so if you change `HEAAN2_DIR` or the toolchain you must `rm -rf submissions/mnist/build`
first — otherwise the old settings persist *silently* and the build keeps using the previous
configuration no matter what you pass.

`scripts/get_openfhe.sh` is untouched and still runs on every harness invocation:
`submissions/cifar10` needs OpenFHE. This submission does not.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `no kernel image is available for execution on the device`, or a launch failure on the first homomorphic operation | Your GPU is not sm_120, and the vendored library has no PTX to fall back on. Confirm with the [architecture check](#-this-submission-requires-an-sm_120-gpu). This needs a library built for your architecture; it is not fixable from this repository. |
| `identifier "__is_array" is undefined` while CMake detects the CUDA compiler | nvcc probed one `gcc` but preprocessed with another — it prepends its own `bin/` to `PATH`, so this appears when a conda toolchain is *not* active. `build_task.sh` pins the host compiler; if it persists, set `HEAAN2_CUDA_HOST_COMPILER`. |
| `Could not find nvcc ... set CUDAToolkit_ROOT` | `nvcc` not on `PATH` and not auto-detected. Put the CUDA toolkit on `PATH` or set `HEAAN2_NVCC`. |
| `undefined reference to 'cuda…@libcudart.so.12'` at link, or `error while loading shared libraries: libcudart.so.12` at run | Stale `submissions/mnist/build`. `rm -rf` it and rebuild. |
| `ModuleNotFoundError: No module named 'torch'` from `generate_dataset.py` | venv not active, so harness children use the system interpreter. `source ./bmenv/bin/activate`. |
| `CUDA device is not available` in stage 7 | No GPU visible. Use `srun` or an equivalent allocation. |
