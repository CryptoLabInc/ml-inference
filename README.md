# FHE Benchmarking Suite — ML Inference: CryptoLab HEaaN2 submission

A submission for the [HomomorphicEncryption.org](https://www.HomomorphicEncryption.org)
`ml-inference` benchmark (`--dataset mnist`) by CryptoLab, Inc., built on
[HEaaN2](https://heaan.io), CryptoLab's CKKS library. The submission lives under
[`submissions/mnist/`](submissions/mnist/).

The model is a two-layer MLP, `fc1(128×484) → x² → fc2(10×128)`, over a 484-dimensional
center-cropped input. Two CKKS circuits evaluate it, picked depending on the instance size: a Halevi–Shoup
rotation-folded matvec at size 0, and a PCMM (GEMM-based) circuit at sizes 1–3.

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

| | size 0 (1) HS | sizes 1–2 (100 / 1000) PCMM | size 3 (10000) PCMM |
| --- | --- | --- | --- |
| Harness `Encrypted model preprocessing` | 2.094 s | 0.058 s / 0.070 s | 0.068 s |
| Harness `Encrypted computation` | 0.371 s | 0.416 s / 0.417 s | 0.521 s |
| ├─ model setup | 0.090 s | 0.046 s | 0.058 s |
| ├─ warm-up (discarded) | 0.013 s | 0.028 s | 0.012 s |
| └─ evaluation | **0.70 ms** | **0.80 ms** | **3.40 ms** |
| Public + evaluation keys | 44.9 M | 108.5 K | 320.5 K |
| Encrypted input | 420 K | 51.2 M | 242.5 M |
| Encrypted results | 120 K | 350.1 K | 1.8 M |
| **Accuracy** | PASS | 0.980 / 0.989 | 0.9796 |
| Harness plaintext model | n/a | 0.960 / 0.981 | 0.9779 |

`Encrypted computation` is the number the benchmark scores. The indented rows under it are the
submission's own timers and do not sum to it;
[Results](submissions/mnist/README.md#results) covers what each one measures.

At size 0 the Halevi–Shoup diagonal encoding runs in `Encrypted model preprocessing`
[`server_preprocess_model.cpp`](submissions/mnist/src/server_preprocess_model.cpp). That is why `Encrypted computation` reads 0.371 s here rather than
about 2.5 s. For a cold single-shot latency, add the two rows together.

## License

The repository is Apache-2.0; see [LICENSE.md](LICENSE.md). The prebuilt HEaaN2 library under
[`submissions/mnist/install/`](submissions/mnist/install/) is proprietary to CryptoLab, Inc. and is
redistributed for benchmark verification only, under
[submissions/mnist/LICENSE](submissions/mnist/LICENSE).
