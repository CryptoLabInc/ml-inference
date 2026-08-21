# FHE Benchmarking Suite — ML Inference: CryptoLab HEaaN2 submission

A submission for the [HomomorphicEncryption.org](https://www.HomomorphicEncryption.org)
`ml-inference` benchmark (`--dataset mnist`) by CryptoLab, Inc., built on
[HEaaN2](https://heaan.io), CryptoLab's CKKS library. The submission lives under
[`submissions/mnist/`](submissions/mnist/).

The model is a two-layer MLP, `fc1(128×484) → x² → fc2(10×128)`, over a 484-dimensional
center-cropped input. A single CKKS circuit evaluates it at every instance size: PCMM, a GEMM-based
product that needs no rotations, so no rotation keys are generated at any size.

- Design, parameters, security, build, run and results:
  [submissions/mnist/README.md](submissions/mnist/README.md)
- Building requires an sm_120 GPU. See the
  [pre-flight check](submissions/mnist/README.md#-this-submission-requires-an-sm_120-gpu).
- Upstream harness documentation:
  [fhe-benchmarking/ml-inference](https://github.com/fhe-benchmarking/ml-inference)

## Benchmark results

Measured on one NVIDIA RTX 5090 (sm_120) with seed 3, running
`python3 harness/run_submission.py <size> --num_runs 3`. The files under
[`measurements/`](measurements/) hold the three runs that the leaderboard averages.

| | size 0 (1) | size 1 (100) | size 2 (1000) | size 3 (10000) |
| --- | --- | --- | --- | --- |
| Harness `Encrypted model preprocessing` | 0.079 s | 0.065 s | 0.072 s | 0.071 s |
| Harness `Encrypted computation` | 0.417 s | 0.441 s | 0.424 s | 0.485 s |
| ├─ model setup | 0.058 s | 0.057 s | 0.053 s | 0.073 s |
| ├─ warm-up (discarded) | 0.031 s | 0.035 s | 0.033 s | 0.015 s |
| └─ evaluation | **0.57 ms** | **0.57 ms** | **0.57 ms** | **2.32 ms** |
| Public + evaluation keys | 108.5K | 108.5K | 108.5K | 320.5K |
| Encrypted input | 51.2M | 51.2M | 51.2M | 242.5M |
| Encrypted results | 350.1K | 350.1K | 350.1K | 1.8M |
| **Accuracy** | **PASS** | **0.98** | **0.989** | **0.9796** |
| Harness plaintext model | n/a | 0.96 | 0.981 | 0.9779 |

`Encrypted computation` is the number the benchmark scores. The indented rows under it are the
submission's own timers and do not sum to it;
[Results](submissions/mnist/README.md#results) covers what each one measures.

Sizes 0–2 report identical bandwidth because PCMM encrypts the batch as a fixed 485 × 4096
ciphertext matrix: one image occupies one column and the remaining 4095 are padding. The upload is
sized by the profile rather than the batch — efficient at 1000 images, over-provisioned at one.

## License

The repository is Apache-2.0; see [LICENSE.md](LICENSE.md). The prebuilt HEaaN2 library under
[`submissions/mnist/install/`](submissions/mnist/install/) is proprietary to CryptoLab, Inc. and is
redistributed for benchmark verification only, under
[submissions/mnist/LICENSE](submissions/mnist/LICENSE).
