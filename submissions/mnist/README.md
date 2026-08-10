# MNIST MLP inference on HEaaN2 (CKKS, GPU)

FHE submission for the HomomorphicEncryption.org `ml-inference` benchmark
(`--dataset mnist`), built on [HEaaN2](https://github.com/CryptoLabInc/HEaaN2), Crypto Lab's
CKKS library. Replaces the reference OpenFHE/HEIR submission in this directory; the harness is
unmodified.

Two homomorphic circuits are used, chosen purely by instance size (`mlp::usePcmm` in
[`include/mlp_params.hpp`](include/mlp_params.hpp)) — same model, same weights, same accuracy
target, different packing:

| Instance size | Scheme | Why |
| --- | --- | --- |
| 0 single, 1 small (1, 100 images) | **Halevi–Shoup** rotation-folded matvec | fixed 128-images/ciphertext packing; cheap at small batches |
| 2 medium, 3 large (1000, 10000) | **PCMM** (GEMM-based) | per-image cost keeps falling as the batch grows, where HS's is flat |

Every stage binary dispatches on the instance size internally, so the harness's contract (seven
fixed executable names, `<size>` as the only argv) is unchanged.

> **Security notice.** The ≥128-bit parameter justification for **both** schemes' configurations is
> **not finalised** — see [DESIGN.md §5](DESIGN.md#5-security). Do not cite this submission's
> parameters as reviewed.

- **[BUILDING.md](BUILDING.md)** — requirements, build from a fresh clone (no HEaaN2 source or
  private-repo access needed: a prebuilt HEaaN2 is vendored in [`install/`](install/)), running the
  instance sizes, which of the three machines to use, environment variables, troubleshooting.
- **[DESIGN.md](DESIGN.md)** — the circuit, both schemes' parameters and level schedules, cleartext
  pre/post-processing, deviation from the harness's reference model, results, and the security
  section.

## Licence

Submission code (`src/`, `include/`, `CMakeLists.txt`, `weights/`): Apache-2.0, as the repository.
It uses only HEaaN2's **public API** — none of HEaaN2's own implementation source is in this repo.

[`install/`](install/) carries a **prebuilt** HEaaN2 (public headers + `libheaan2.so.0.2.0`), which
is proprietary to Crypto Lab Inc. and is *not* under Apache-2.0. It is redistributed here under
[LICENSE](LICENSE), which permits use solely for reproducing and verifying benchmark results —
the same arrangement CryptoLab's
[Zn-multiplication submission](https://github.com/CryptoLabInc/Zn-multiplication/tree/CryptoLabInc)
uses. Third-party components and runtime dependencies of that binary are listed in
[LICENSE-THIRD-PARTY](LICENSE-THIRD-PARTY).
