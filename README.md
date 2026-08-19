# FHE Benchmarking Suite — ML Inference: CryptoLab HEaaN2 submission

This fork is a submission for the [HomomorphicEncryption.org](https://www.HomomorphicEncryption.org)
`ml-inference` benchmark (`--dataset mnist`) by **CryptoLab Inc.**, built on
[HEaaN2](https://heaan.io), CryptoLab's CKKS library. The harness is unmodified; the submission
replaces the reference OpenFHE implementation under [`submissions/mnist/`](submissions/mnist/).

- **Everything about the submission** — design, parameters, security, build, run, results:
  [submissions/mnist/README.md](submissions/mnist/README.md)
- **Building requires an sm_120 GPU** — see the
  [pre-flight check](submissions/mnist/README.md#-this-submission-requires-an-sm_120-gpu)
- Upstream harness documentation: [fhe-benchmarking/ml-inference](https://github.com/fhe-benchmarking/ml-inference)

Two CKKS circuits evaluate the same 2-layer MLP (`fc1(128×484) → x² → fc2(10×128)`), chosen by
instance size alone: a Halevi–Shoup rotation-folded matvec at size 0, a PCMM (GEMM-based)
circuit at sizes 1–3. The model deviates from the harness's own
(`784→128→64→10` with ReLU): two layers instead of three, `x²` instead of ReLU, and a 484-dim
center-cropped input, trained separately.

## Benchmark results

Measured on **1× NVIDIA RTX 5090 (sm_120)**, seed 3, through the unmodified harness
(`python3 harness/run_submission.py <size> --num_runs 3`); the committed files under
[`measurements/`](measurements/) are the three-run results the leaderboard averages.

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

*(`Encrypted computation` and `Total latency` are means of the three committed runs; model
preprocessing and key generation are measured once per size. Accuracy varies in the third decimal
across runs from encryption noise. Size 1 moved from Halevi–Shoup to PCMM after these
measurements were taken; its column is cleared pending re-measurement rather than carrying over
figures for a circuit it no longer runs.)*

*At size 0 the Halevi–Shoup circuit's diagonal encoding runs in [`client_preprocess_input.cpp`](submissions/mnist/src/client_preprocess_input.cpp), not inside the timed [`server_encrypted_compute.cpp`](submissions/mnist/src/server_encrypted_compute.cpp): it depends only on the weights, never on the input.
That is why `Encrypted computation` is 0.41 s rather than the ~2.5 s it would otherwise be — the
work moved out of the scored time rather than disappearing. Quote the two rows together for a
cold single-shot latency.*

*The **warm evaluation** row is the submission's own timer around the homomorphic inference alone,
reported per run in the `Server Reported` block of each
[`measurements/`](measurements/) file. It is a sub-figure of `Encrypted computation`, not an
alternative to it: the harness times [`server_encrypted_compute.cpp`](submissions/mnist/src/server_encrypted_compute.cpp) as a whole process, so the scored figure also carries
key loading, process and CUDA start-up and ciphertext I/O, and the arithmetic itself is well under
1% of it. **`Encrypted computation` is the number the benchmark scores**; the evaluation row is
what the circuit costs once a server is warm. See
[Results](submissions/mnist/README.md#results) for the full breakdown.*

## License

The repository is Apache-2.0 (see [LICENSE.md](LICENSE.md)). The prebuilt HEaaN2 library vendored
at [`submissions/mnist/install/`](submissions/mnist/install/) is proprietary to CryptoLab Inc. and
redistributed for benchmark verification only, under
[submissions/mnist/LICENSE](submissions/mnist/LICENSE).
