# FHE Benchmarking Suite — ML Inference: CryptoLab HEaaN2 submission

This fork is a submission for the [HomomorphicEncryption.org](https://www.HomomorphicEncryption.org)
`ml-inference` benchmark (`--dataset mnist`) by **CryptoLab Inc.**, built on
[HEaaN2](https://heaan.io), CryptoLab's CKKS library. The harness is unmodified; the submission
replaces the reference OpenFHE implementation under [`submissions/mnist/`](submissions/mnist/).

- **What it computes and how**: [submissions/mnist/DESIGN.md](submissions/mnist/DESIGN.md)
- **Building and running** (requires an sm_120 GPU — see the pre-flight check):
  [submissions/mnist/BUILDING.md](submissions/mnist/BUILDING.md)
- **Overview and licensing**: [submissions/mnist/README.md](submissions/mnist/README.md)
- Upstream harness documentation: [fhe-benchmarking/ml-inference](https://github.com/fhe-benchmarking/ml-inference)

Two CKKS circuits evaluate the same 2-layer MLP (`fc1(128×484) → x² → fc2(10×128)`), chosen by
instance size alone: a Halevi–Shoup rotation-folded matvec at size 0, a PCMM (GEMM-based)
circuit at sizes 1–3. Deviations from the harness model are documented in
[DESIGN.md §3](submissions/mnist/DESIGN.md#3-deviation-from-the-harness-model).

> **Security.** Both schemes use uniform-ternary secrets and size their switching-key modulus
> against a ≥128-bit budget table indexed by RLWE dimension; the policy, the table and how each
> scheme is instantiated under it are in
> [DESIGN.md §5](submissions/mnist/DESIGN.md#5-security). **That analysis is not finalised** — do
> not cite these parameters as reviewed.

## Benchmark results

Measured on **1× NVIDIA RTX 5090 (sm_120)**, seed 3, through the unmodified harness
(`python3 harness/run_submission.py <size> --num_runs 3`); the committed files under
[`measurements/`](measurements/) are the three-run results the leaderboard averages.

| | size 0 (single, 1) | size 1 (small, 100) | size 2 (medium, 1000) | size 3 (large, 10000) |
| --- | ---: | ---: | ---: | ---: |
| Circuit | Halevi–Shoup | PCMM | PCMM | PCMM |
| Encrypted model preprocessing | 2.15 s | — | 0.07 s | 0.07 s |
| Encrypted computation | **0.41 s** | — | **0.44 s** | **0.56 s** |
| └─ warm evaluation, server-reported | **0.76 ms** | — | **0.91 ms** | **2.63 ms** |
| Total latency | 9.40 s | — | 9.62 s | 16.30 s |
| Public + evaluation keys | 44.9 M | — | 54.0 K | 54.0 K |
| Encrypted input | 420 K | — | 44.5 M | 133.6 M |
| Encrypted results | 120 K | — | 280 K | 840 K |
| Accuracy | PASS | — | 0.989 | 0.979 |

*(`Encrypted computation` and `Total latency` are means of the three committed runs; model
preprocessing and key generation are measured once per size. Accuracy varies in the third decimal
across runs from encryption noise. Size 1 moved from Halevi–Shoup to PCMM after these
measurements were taken; its column is cleared pending re-measurement rather than carrying over
figures for a circuit it no longer runs.)*

*At size 0 the Halevi–Shoup circuit's diagonal encoding runs in stage 3 (`Encrypted model
preprocessing`), not inside the timed stage 7: it depends only on the weights, never on the input.
That is why `Encrypted computation` is 0.41 s rather than the ~2.5 s it would otherwise be — the
work moved out of the scored stage rather than disappearing. Quote the two rows together for a
cold single-shot latency.*

*The **warm evaluation** row is the submission's own timer around the homomorphic inference alone,
reported per run in the `Server Reported` block of each
[`measurements/`](measurements/) file. It is a sub-figure of `Encrypted computation`, not an
alternative to it: the harness times stage 7 as a whole process, so the scored figure also carries
key loading, process and CUDA start-up and ciphertext I/O, and the arithmetic itself is well under
1% of it. **`Encrypted computation` is the number the benchmark scores**; the evaluation row is
what the circuit costs once a server is warm. See
[DESIGN.md §4](submissions/mnist/DESIGN.md#4-results) for the full breakdown.*

## License

The repository is Apache-2.0 (see [LICENSE.md](LICENSE.md)). The prebuilt HEaaN2 library vendored
at [`submissions/mnist/install/`](submissions/mnist/install/) is proprietary to CryptoLab Inc. and
redistributed for benchmark verification only, under
[submissions/mnist/LICENSE](submissions/mnist/LICENSE).
