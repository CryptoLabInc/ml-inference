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
instance size alone: a Halevi–Shoup rotation-folded matvec at sizes 0–1, a PCMM (GEMM-based)
circuit at sizes 2–3. Deviations from the harness model are documented in
[DESIGN.md §3](submissions/mnist/DESIGN.md#3-deviation-from-the-harness-model).

## Benchmark results

Measured on **1× NVIDIA RTX 5090 (sm_120)**, seed 3, through the unmodified harness
(`python3 harness/run_submission.py <size> --num_runs 3`); the committed files under
[`measurements/`](measurements/) are the three-run results the leaderboard averages.

| | size 0 (single, 1) | size 1 (small, 100) | size 2 (medium, 1000) | size 3 (large, 10000) |
| --- | ---: | ---: | ---: | ---: |
| Circuit | Halevi–Shoup | Halevi–Shoup | PCMM | PCMM |
| Encrypted computation | | | | |
| Total latency | | | | |
| Public + evaluation keys | 174.8 M | 174.8 M | 54.0 K | 54.0 K |
| Encrypted input | 1.6 M | 1.6 M | 44.5 M | 133.6 M |
| Encrypted results | 480 K | 480 K | 280 K | 840 K |
| Accuracy | PASS | 0.980 | 0.989 | 0.979 |

*(Timing cells are filled from the official bench-machine runs; bandwidth and accuracy are
machine-independent and shown as measured. Accuracy varies in the third decimal across runs from
encryption noise.)*

## License

The repository is Apache-2.0 (see [LICENSE.md](LICENSE.md)). The prebuilt HEaaN2 library vendored
at [`submissions/mnist/install/`](submissions/mnist/install/) is proprietary to CryptoLab Inc. and
redistributed for benchmark verification only, under
[submissions/mnist/LICENSE](submissions/mnist/LICENSE).
