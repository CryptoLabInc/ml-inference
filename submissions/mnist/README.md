# MNIST MLP inference on HEaaN2 (CKKS, GPU)

FHE submission for the HomomorphicEncryption.org `ml-inference` benchmark (`--dataset mnist`),
built on [HEaaN2](https://heaan.io), Crypto Lab's CKKS library. Replaces the reference
OpenFHE/HEIR submission in this directory. The harness is unmodified.

> **Security notice.** The ≥128-bit parameter justification is **not finalised** for either scheme
> — see [DESIGN.md §5](DESIGN.md#5-security). Do not cite these parameters as reviewed.

## At a glance

| | |
| --- | --- |
| Model | 2-layer MLP, `fc1(128×484) → x² → fc2(10×128)`, BN folded. Plaintext accuracy 97.96% |
| Accuracy | 0.980 / 0.989 / 0.979 at sizes 1 / 2 / 3, against the harness model's 0.960 / 0.981 / 0.978 |
| Scheme | CKKS, no bootstrapping. Conjugate-invariant subring at sizes 0 and 3; the plain ring at sizes 1–2, where CI's doubled slot count removes no block — see [DESIGN.md §1](DESIGN.md#scheme-b--pcmm-sizes-13) |
| Hardware | **NVIDIA sm_120 (Blackwell) GPU required** — the vendored HEaaN2 binary targets sm_120 only, with no PTX fallback. [Check first](BUILDING.md#-this-submission-requires-an-sm_120-gpu) |

Two circuits, chosen by instance size alone (`mlp::usePcmm`) — same model, same weights, different
packing:

| Size | Scheme | Why |
| --- | --- | --- |
| 0 (1 image) | **Halevi–Shoup** rotation-folded matvec | faster on a single image (0.76 ms against PCMM's 0.91 ms — a PCMM block costs the same at 1 image as at 4096), and the one size exercising public-key encryption, which PCMM cannot offer |
| 1–3 (100, 1000, 10000) | **PCMM** (GEMM-based) | needs no rotation keys and no diagonal encoding, so ~105× less model preprocessing (~70 ms against HS's ~7.3 s), and 8.5–29× faster evaluation at the larger batch sizes |

Every stage binary dispatches internally, so the harness contract (seven fixed executable names,
`<size>` as the only argument) is unchanged.

## Quick start

```bash
pip install -r requirements.txt
./scripts/build_task.sh ./submissions/mnist     # uses the vendored HEaaN2 in install/
python3 harness/run_submission.py 0 --seed 3    # prefix with srun under a scheduler
```

No HEaaN2 checkout, private-repo access or SSH key needed. Details in
**[BUILDING.md](BUILDING.md)**; how it works and why, in **[DESIGN.md](DESIGN.md)**.

## Licence

Submission code (`src/`, `include/`, `CMakeLists.txt`, `weights/`) is Apache-2.0, as the
repository. It uses only HEaaN2's public API — no HEaaN2 implementation source is in this repo.

[`install/`](install/) carries a **prebuilt** HEaaN2 (public headers + `libheaan2.so.0.2.0`),
proprietary to Crypto Lab Inc. and **not** Apache-2.0. It is redistributed under
[LICENSE](LICENSE), which permits use solely for reproducing and verifying benchmark results —
the same arrangement as CryptoLab's
[Zn-multiplication submission](https://github.com/CryptoLabInc/Zn-multiplication). Its
dependencies are listed in [LICENSE-THIRD-PARTY](LICENSE-THIRD-PARTY).
