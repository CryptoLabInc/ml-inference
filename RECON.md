# RECON — HEaaN2 submission to the `ml-inference` benchmark

Recon deliverable per `PASSDOWN_ml_inference_submission.md` §3, kept as the design record now
that the implementation exists (see **Status** at the end). Everything below was read out of the
source in this repo and in `/home/yongwonchoi/HEaaN2`; where the upstream prose and the source
disagree, the source is recorded and the prose is flagged as stale.

---

## 0. The passdown's §0 block, filled in from evidence

| Field | Value | How determined |
| --- | --- | --- |
| Our library name | **HEaaN2** v0.2.0 | given by human |
| Library repo / install path | `/home/yongwonchoi/HEaaN2` (branch `hem-heaan-mlinf-frobmap`, HEAD `5502b04`) | local clone, builds |
| Language / binding | **C++17**, direct link against `libheaan2.so` | library is a C++ shared lib |
| FHE scheme | **CKKS**, conjugate-invariant (CI) subring, `PolyType::GRAFTED` | what the existing `mlp/` benchmarks use |
| Local path to fork | `/home/yongwonchoi/ml-inference` | — |
| Execution mode | **`local`** (`submissions/`), **on GPU via `srun`** | confirmed by human |
| Target model dir | **`submissions/mnist/`** | `run_submission.py:44`; confirmed by human |
| Open/closed source | **hardware-backed / GPU**; HEaaN2 **public API only**, no HEaaN2 source vendored into this fork | confirmed by human |

All §0 fields are now settled. Run command is `srun python3 harness/run_submission.py <size> …`
— see "GPU execution" below.

---

## 3a. The harness contract (extracted from source)

`harness/run_submission.py` is the spec. Resolution of stage → executable is
`harness/utils.py:242 run_exe_or_python(base, file_name, *args)`:

```python
py  = base / f"{file_name}.py"        # tried first
exe = base / "build" / file_name      # tried second
...
else: cmd = None                      # <-- NEITHER EXISTS: silently does nothing
if cmd is not None: subprocess.run(cmd, check=check)
```

> **Trap.** A stage whose executable is missing is **not** an error — the harness skips it
> silently and marches on. A typo in a target name produces a green run with garbage output, not
> a build failure. Every stage name below must match exactly.

`base` is `model_exec_dir = submissions/<dataset_name>` (`run_submission.py:41,44`), so our
binaries must land at **`submissions/mnist/build/<stage>`**.

### Stage table

`prms` below = `datasets/<inst>/…` and `io/<inst>/…` where `<inst>` ∈ `single|small|medium|large`
(`params.py:26-31,57-77`). Batch sizes: `[1, 100, 1000, 10000]` (`params.py:44`).

| # | stage | invoked as | reads | writes | success criterion |
| --- | --- | --- | --- | --- | --- |
| — | build | `scripts/get_openfhe.sh` then `scripts/build_task.sh ./submissions/mnist` | — | `submissions/mnist/build/*` | **exit 0** (`check=True`) |
| 1 | `generate_dataset` | harness, `<datasets/<inst>/dataset.txt> mnist` | torchvision MNIST | `datasets/<inst>/dataset_{pixels,labels}.txt` (10000 rows) | exit 0 |
| 2.1 | `server_get_params` | remote only, `<size>` | — | — | remote only |
| 2.2 | `client_key_generation` | **`<size>`** | — | key material under `io/<inst>/` | exit 0 |
| — | *size report* | `du -sb io/<inst>/public_keys` | — | → `"Public and evaluation keys"` | dir must exist or reports `0B` |
| 2.3 | `server_upload_ek` | remote only, `<size>` | — | — | remote only |
| 3 | `server_preprocess_model` | **no argv at all** (`run_submission.py:97`) | model weights | server-side cached model | exit 0 |
| 4 | `generate_input` | harness, `<size> --dataset mnist [--seed S]` | `datasets/<inst>/` | `datasets/<inst>/intermediate/test_{pixels,labels}.txt` | exit 0 |
| 5 | `client_preprocess_input` | **`<size>`** | `…/intermediate/test_pixels.txt` | our choice | exit 0 |
| 6 | `client_encode_encrypt_input` | **`<size>`** | step 5 output + public key | ciphertexts | exit 0 |
| — | *size report* | `du -sb io/<inst>/ciphertexts_upload` | — | → `"Encrypted input"` | dir must exist |
| 7 | `server_encrypted_compute` | **`<size>`** | ciphertexts + eval keys + model | ciphertexts | exit 0 |
| — | *size report* | `du -sb io/<inst>/ciphertexts_download` | — | → `"Encrypted results"` | dir must exist |
| 8 | `client_decrypt_decode` | **`<size>`** | result ciphertexts + secret key | scores | exit 0 |
| 9 | `client_postprocess` | **`<size>`** | scores | **`io/<inst>/encrypted_model_predictions.txt`** | exit 0 |
| 10 | verify / quality | harness | see below | `measurements/<inst>/results-<n>.json` | see below |

Every stage is run with `check=True` (non-zero exit aborts the whole run) **except** step 10's
`verify_result`, which is `check=False` so a wrong label is reported but does not stop the run.

### The three paths that are truly load-bearing

Only these are read by harness code we may not touch:

1. **`io/<inst>/encrypted_model_predictions.txt`** — our final output.
   - size 0: `verify_result.py:35` does `int(file.read_text().strip())` — **exactly one integer**,
     leading/trailing whitespace fine. Compared against `test_labels.txt` read the same way.
   - size ≥1: `utils.py:215-229` splits on `\n`, strips, drops empty lines, and compares
     **string-equal** against the labels file line by line. Accuracy denominator is
     `len(preds)` — so writing **fewer** lines than the batch size silently inflates accuracy.
     Write exactly `batch_size` lines of bare decimal integers.
2. **`datasets/<inst>/intermediate/test_pixels.txt`** — our input. Written by
   `mnist.export_test_pixels_labels`: one sample per line, **784 space-separated floats
   `%.6f`, already scaled to [0,1] by `ToTensor` but NOT mean/std-normalized**
   (`harness/mnist/mnist.py:129-131,150`).
3. **`io/<inst>/{public_keys,ciphertexts_upload,ciphertexts_download}/`** — measured with
   `du -sb` (`utils.py:151`). **Directory names are fixed**; contents/format are entirely ours.
   A missing directory logs a warning and records `0B` rather than failing.

Optional: `io/<inst>/server_reported_steps.json`, a flat `{"step name": <seconds as number>}`
object; the harness appends `"s"` and copies it into `results-<n>.json` under `"Server Reported"`
(`utils.py:176-182`).

### Timing and results

`log_step` (`utils.py:116`) takes wall-clock deltas between consecutive harness calls — so each
stage's reported time includes our process startup, key/ciphertext deserialization and file I/O,
not just the crypto. `save_run` writes `measurements/<inst>/results-<n>.json` per run with
`Timing`, `Bandwidth`, `Quality` (size ≥1 only), `Server Reported`, plus `model_name`/`dataset_name`.

### Correctness / accuracy oracle

- **size 0** → `verify_result.py`, prints `PASS (expected=N, got=N)`, exit 1 on mismatch (not fatal).
- **size ≥1** → `cleartext_impl.py` runs the *harness's own* `SimpleFFNN` over the same
  `test_pixels.txt` into `harness_model_predictions.txt`, then `calculate_quality` scores both
  ours and the harness model against ground truth.

### The harness's reference model (`harness/mnist/`)

```
SimpleFFNN:  fc1 Linear(784,128) -> ReLU -> fc2 Linear(128,64) -> ReLU -> fc3 Linear(64,10)
```
(`harness/mnist/model.py:10-13`). Normalization applied at predict time is
`(p - 0.1307) / 0.3081` on the already-[0,1] pixels (`harness/mnist/test.py:60`). Weights live at
`harness/mnist/mnist_ffnn_model.pth`, which **is not in the repo** — the first size-≥1 run trains
it for 15 epochs. Reported harness accuracy ≈ 0.97.

**Neither the reference submission nor ours uses this architecture** — see §3d and Q5.

---

## 3b. The reference implementation (`submissions/mnist/`)

Working OpenFHE/HEIR example of exactly the contract above. Layout: `src/<stage>.cpp` one
executable each, `include/params.h` mirroring `harness/params.py`, and a 28 699-line
HEIR-generated `src/mlp_openfhe.cpp`.

What it establishes as *convention* (not contract) and that we should mirror where free:

- `io/<inst>/public_keys/{cc,pk,mk,rk}.bin`, `io/<inst>/secret_key/sk.bin`
  (`client_key_generation.cpp:35-58`). `secret_key/` is **not** measured — only `public_keys/` is.
- `io/<inst>/ciphertexts_upload/cipher_input_<i>.bin`, one file per image;
  `ciphertexts_download/cipher_result_<i>.bin` likewise.
  **This is a free choice, not a requirement** — the harness only `du`s the directory. It is why
  the reference uploads 500 MB at size 1 and 4.9 GB at size 2.
- `io/<inst>/intermediate/preprocessed_input.txt` and `model_scores.txt` as the inter-stage
  text hand-offs; scores are 10 space-separated floats per line, argmax'd in `client_postprocess`.
- `server_preprocess_model` is a **no-op returning 0** — the reference does all its model work
  inside `server_encrypted_compute`.
- Its own deviation from the harness model, documented in `submissions/mnist/docs/README.md`:
  a **512×784 first FC layer with a polynomial approx-ReLU**, ~95% plaintext accuracy. So the
  precedent for changing the architecture is already set by the reference itself.

### Reference numbers to beat (`measurements/`)

| | size 0 (1 img) | size 1 (100) | size 2 (1000) |
| --- | --- | --- | --- |
| Encrypted computation | 7.61 s | 647.07 s | 6505.61 s |
| Public + eval keys | 1.0 G | 1.0 G | 1.0 G |
| Encrypted input | 5.0 M | 500.3 M | 4.9 G |
| Encrypted results | 1.0 M | 100.2 M | 1001.6 M |
| Encrypted-model accuracy | — | 0.96 | 0.974 |
| Harness-model accuracy | — | 1.00 | 0.981 |

---

## 3c. HEaaN2 inventory

Everything the contract needs is in the public API (`include/HEaaN2/HEaaN2.hpp` umbrella).

| Need | HEaaN2 API |
| --- | --- |
| Params / modulus chain | `paramsUtils::LevelsBuilder` → `Levels` (`setRing`, `initMod`, `buildAbove`) |
| Secret key | `SKGenerator{SKGenParams{log_degree, hw, ntt_alg}}` → `genKey()`, `genHighDegreeKey()` |
| Public (encryption) key | `EncKeyGenerator{EncKeyGenParams{...}}` → `genKey(sk)` → `Ptr<IEncKey>` |
| Relin / rotation keys | `SwKeyGenerator` → `genRelinKey(sk)`, `genRotKeys(sk, steps)` → `RotKeyPtrs` |
| Encode / decode | `EnDecoder{EncodeParams{...}}` → `encode(Message, IPlaintext, level)`, `decode` |
| Encrypt / decrypt | `EnDecryptor::encrypt(ptxt, sk|enckey, ctxt)`, `decrypt(ctxt, sk, ptxt)` — **public-key encryption is supported**, so we need not encrypt symmetrically |
| Add / tensor / relin / rescale / adjust | `HomEval`, `HomEvalFlexible::adjust` |
| Rotation | `HomEval::rot` (keyed), `HomEval::frobMap` (key-less Galois) |
| Matrix-vector | `MatrixVectorEval` — BSGS with double-hoisted rotations, consumes exactly 1 level |
| **Serialization** | `heaan::serial::save/load` for `Message`, `IPlaintext`, `ICiphertext`, `ISecretKey`, `IEncKey`, `ISwKey`, **`RotKeyPtrs`**, `KeyPtrBundle`, `BootKeyPtrs` — both `std::string` filename and `std::ostream` overloads |

**Nothing needed is missing.** In particular `RotKeyPtrs` serializes as one unit, so the whole
rotation-key set is a single file.

### The decisive find: `HEaaN2/mlp/` already implements this exact workload

The checked-out branch carries a complete, tuned MNIST MLP FHE inference, in two variants:

```
fc1 (128x484, +b1)  ->  x^2  ->  fc2 (10x128, +b2)
```

| | `mlp/Halevi–Shoup/MLInference_128.cpp` | `mlp/PCMM/MLInference_Large.cpp` |
| --- | --- | --- |
| Method | HS diagonal matvec + rotation folding | pcmm (GEMM); feature = ct row, image = slot |
| Ring | CI, N=2^17, 65536 slots, 512 coords/image → **128 images per ciphertext** | CI, N=2^13…2^16 selectable |
| README (GPU) | 1148 µs per 128 images (**8.97 µs/image**), 97.66% | 2.54 ms per 1000 images (**2.54 µs/image**), 97.75% ± 0.45 |
| **Measured here, CPU** | **122 ms** per 128 images (**952 µs/image**), **97.66%** | **173 ms** per 1000 images (**173 µs/image**), **97.60%** |
| Best for | sizes 0–1 | sizes 2–3 ("wins as the batch grows") |

Both were run during recon from `HEaaN2/build-cpu` (see below), so these CPU figures are measured
on this machine, not extrapolated. PCMM is 5.5× faster per image than HS at 1000 images, which
confirms the README's claim and the size-2/3 plan. This box is noisy (HS eval ±35 ms over 3 runs),
so treat them as order-of-magnitude, not as measurements.

Shared support code is `mlp/Halevi–Shoup/InferenceCommon.hpp` (652 lines: CSV/idx loaders, the
cleartext reference, `LayerSpec`→`Layer` offline precompute, `homLayer` online eval) and
`mlp/PCMM/Common.hpp`.

Model + weights are checked in at `mlp/Common/weights/` (`W1_128x484.csv`, `b1_128.csv`,
`W2_10x128.csv`, `b2_10.csv`, `manifest.txt`), produced by `mlp/Common/export_weights.py`:
x²+BatchNorm, 30 epochs, seed 0, BN folded into fc1 at export. **Folded plaintext accuracy
97.96%**, max |logit diff| vs the unfolded model 6.1e-05.

### Parameter-security story — this is the good news

`InferenceCommon.hpp:115-131` carries an explicit 128-bit modulus-budget table for
uniform-ternary secrets (`hw = 0`):

```
N=2^13 → 214 bits   2^14 → 430   2^15 → 868   2^16 → 1748   2^17 → 3523
```
sourced from HEaven's `maxBitsPolicy128()` (2^13–2^15) and **ePrint 2024/463** (2^16, 2^17).

The subtlety is already handled in the code, and correctly. `makeLayer` sizes every switching
key from the degree the secret was *sampled* at, not the ring it sits in
(`InferenceCommon.hpp:511-523`):

```cpp
swk_builder.setRing(sk.logDegree(), poly_type);              // keys live in the full ring...
swk_builder.setModUpPrimes(                                   // ...but security is the sampled key's
    maxBits128(sk_log_degree_low - (ntt_alg == CYC_FOR_CI ? 1 : 0)), swk_margin);
```

with the reasoning spelled out in the comment: *"Lifting adds no entropy: a key lifted from 2^l
has the LWE problem of 2^l, not of the ring it now sits in, and CI halves it again because
SKGenerator samples only half the coefficients there."*

For the HS variant: sampled at 2^15, CI → effective dimension 2^14 → budget **430 bits**, with a
5-bit margin. Modulus chain is `initMod(30) + 3×25` ≈ **105 bits**, far under it. Noise is
`DiscreteGaussian(3.2)`.

**Status after review round 1: this is an OPEN GAP, not a claim.** The human's answer was *"the
128-bit security will be provided later and assume it is safe for now"*, and `hw` may still
change. Per passdown §1.4 we therefore **record the gap rather than assert the number**: the
submission README will state the parameters and say the analysis is pending crypto-side review,
citing the above as provisional. `hw`, `SMALL_LOG_DEGREE` and the mod-up budget go in **one
header as named constants** so the review can be applied in one place.

Must be closed before the fork goes public — §1.4 makes it a validity condition. If the review
rejects the lifted-key construction specifically, PCMM (no lifting; security straight from the
RLWE dimension) becomes the better primary variant.

### GPU execution — `srun` is mandatory and shapes the run command

`baedal` is a shared Slurm node; CUDA is only reachable inside an allocation. Bare, the benchmark
aborts with *"CUDA device is not available in the current environment"* (`cudaGetDeviceCount` → 999).

Verified during recon:

| invocation | result |
| --- | --- |
| `./build/mlp/MLInference_128 3 12345` | **fails** — no CUDA device |
| `srun ./build/mlp/MLInference_128 3 12345` | **7.52 µs/image**, 97.66% |
| `srun python3 -c "subprocess.run(['./build/mlp/MLInference_128', …])"` | **7.64 µs/image**, rc=0 |

The third row is the one that matters: the harness invokes our stage binaries through
`subprocess.run`, so a nested child inherits the allocation. **The whole harness runs under one
`srun`**, and no harness change is needed:

```bash
srun python3 harness/run_submission.py 0 --seed 3
```

Run bare, the pipeline fails at stage 7 — loudly, which is the good failure mode.

### Build status verified during recon

- CUDA build (`build/`) works **under `srun`** (see above). This is the submission target.
- Also configured and built a CPU tree at `HEaaN2/build-cpu`
  (`-DBUILD_WITH_CUDA=OFF -DBUILD_MLP=ON`, Release, OpenMP + AVX2 + AVX512) to get the CPU
  baseline numbers above. Kept as a fallback, not the submission target.
- Toolchain note: `heaven-dev-g++` originally lacked `cblas.h` (it shipped `libopenblas.so` only)
  while `hem/src/core/PPMM.cpp` includes `<cblas.h>`. Fixed with
  `conda install conda-forge::openblas` — **verified, header now present**. `heaven-dev-cuda`
  already had the full OpenBLAS dev package.

---

## 4. The passdown's open questions, resolved against source

| | Question | Finding |
| --- | --- | --- |
| **Q1** | model dir name / `--model` choices | **Both parts of the passdown's guess are wrong, in our favour.** `--model` and `--dataset` are plain `type=str` with **no `choices=`** (`utils.py:53-56`); nothing is hardcoded. But the submission directory is keyed on the **dataset**, not the model: `model_exec_dir = exec_dir / dataset_name` (`run_submission.py:44`). `--model` only ever reaches the results JSON as a label. Since `harness/<dataset>/` must also exist (`run_submission.py:50`), the only usable values are `mnist` and `cifar10`. → **We keep `submissions/mnist/` and replace its contents.** No harness edit needed. We may pass `--model heaan2-mlp` freely to label the measurements. |
| **Q2** | stage name drift | Confirmed. README's table (line 316-318) says `client_preprocess_dataset` / `client_encode_encrypt_query`; the harness calls **`client_preprocess_input`** and **`client_encode_encrypt_input`**, matching the reference `CMakeLists.txt`. Source wins. There is no `client_preprocess_dataset` stage at all. |
| **Q3** | `submission/` vs `submissions/` | Real path is **`submissions/<dataset>/`**. README lines 46, 103, 288, 300 saying `submission/` or `submissions/mlp` are stale — there is no `submissions/mlp` in this tree, only `mnist` and `cifar10`. |
| **Q4** | build integration | `build_submission` (`utils.py:95-105`) **unconditionally** runs `scripts/get_openfhe.sh` and then `scripts/build_task.sh ./submissions/mnist`, both with `check=True`. → **`get_openfhe.sh` stays untouched**: `submissions/cifar10/CMakeLists.txt:19` does `find_package(OpenFHE CONFIG REQUIRED)`, so a submission that is not ours still needs it, and the script takes no argument telling it which dataset is being built. It already early-exits when `third_party/openfhe/lib` exists, so the cost is one build per fresh clone. **`build_task.sh` gains a dispatch on its existing `$1`**: `submissions/mnist` → our HEaaN2 path (no libtorch), anything else → today's OpenFHE + libtorch path unchanged. Additive; nothing existing changes behaviour. |
| **Q5** | libtorch dependency | **Only the reference submission needs it**, to read `data/traced_model.pt` in `server_encrypted_compute.cpp:19-20,46`. The harness's own Python side uses the `torch` from `requirements.txt`. HEaaN2's weights are plain CSV → **we can drop libtorch entirely**, deleting the ~2.5 GB download from `build_task.sh`. |
| **Q6** | key/ciphertext sizes are scored | Confirmed — `du -sb` on three fixed directory names. The reference's one-file-per-image choice is what costs it 4.9 GB at size 2; our 128-images-per-ciphertext packing should cut upload by ~2 orders of magnitude. Key size needs measuring (`RotKeyPtrs` for two matvec layers + relin key at N=2^17). |
| **Q7** | optional extras | `server_reported_steps.json` confirmed at `io/<inst>/server_reported_steps.json`, flat name→seconds. `server_get_params` / `server_upload_ek` are `--remote`-only. |

---

## 3d. Plan

### Stage → HEaaN2 mapping

| Stage | What it does |
| --- | --- |
| `client_key_generation <size>` | `LevelsBuilder` → `Levels`; `SKGenerator` at 2^15 → `genHighDegreeKey` → sk at 2^17; `EncKeyGenerator::genKey(sk)` → public enc key; `SwKeyGenerator::genRotKeys` for both matvec layers' BSGS steps + `genRelinKey`. `serial::save` sk → `io/<inst>/secret_key/`, enc key + rot keys + relin key → `io/<inst>/public_keys/`. |
| `server_preprocess_model` *(no argv)* | Parse `W1/b1/W2/b2` CSV, build the `std::map<i32, Message>` diagonals via `buildDiags`, `serial::save` them plus the bias `Message`s to a **size-independent** cache under `submissions/mnist/build/model_cache/`. |
| `client_preprocess_input <size>` | Read `test_pixels.txt`; apply `(p - 0.1307)/0.3081`; center-crop 28×28 → 22×22 (484). Write to `io/<inst>/intermediate/`. |
| `client_encode_encrypt_input <size>` | Pack 128 images per `Message` at `slot(c,i) = c*128 + i`; `encode` at top level; `encrypt` with the **public** enc key; save `ceil(batch/128)` ciphertexts to `ciphertexts_upload/`. |
| `server_encrypted_compute <size>` | Load enc keys + cached diagonals, construct the two `MatrixVectorEval`s, then per ciphertext `homLayer(fc1)` → `homLayer(fc2)`. Write `ciphertexts_download/` + `server_reported_steps.json`. |
| `client_decrypt_decode <size>` | Load sk, decrypt/decode, read logit `r` of image `i` from slot `r*128 + i`, write 10 scores per line. |
| `client_postprocess <size>` | argmax → `encrypted_model_predictions.txt`, exactly `batch_size` lines. |

### Code provenance constraint

The submission is written **fresh against the HEaaN2 public API**. No HEaaN2 source is vendored:
`mlp/InferenceCommon.hpp`, `mlp/PCMM/Common.hpp` and the `MLInference_*.cpp` files are read as
*specification*, not copied. The build follows the `zn-mul` branch's `submission/` pattern —
`find_package(HEaaN2 REQUIRED)`, link `HEaaN2::HEaaN2`, own `src/` + `include/`, CUDA on.

Model weights will be **regenerated in this fork** by our own training script rather than copied
from `HEaaN2/mlp/Common/weights/` (see `NOTES_FOR_HUMAN.md` — awaiting confirmation).

### Chosen configuration

Start with the **Halevi–Shoup** variant for all four sizes, chunking the batch into
`ceil(n/128)` ciphertexts. It is the simpler path, is RLWE-only, and already covers sizes 0 and 1
in a single ciphertext. Revisit PCMM for sizes 2–3 as the §5.7 optimization step, once size 0 and
size 1 pass.

- Scheme: CKKS, CI subring, `PolyType::GRAFTED`, `NTTAlgorithm::CYC_FOR_CI`
- Ring N=2^17, 65536 slots, 512 padded coords/image, 128 images/ciphertext
- Levels: `initMod(30)`, 3 levels × 25 bits — `L3` encrypt → fc1 → `L2` → x² → rescale `L1` → fc2 → `L0`
- Secret key sampled at 2^15 (uniform ternary, hw=0), lifted to 2^17 → key-less `frobMap` fold in fc1
- Switching-key budget `maxBits128(14) = 430` bits, margin 5.0, `DiscreteGaussian(3.2)`
- No bootstrapping (depth 3 suffices)

### Packing / encoding

784-dim input → normalize → center-crop to 484 → zero-pad to 512 coordinates → slot `c*128 + i`
for coordinate `c`, image `i`. fc1 is the rectangular 128×512 shape (4:1 fold via 2 key-less
automorphisms); fc2 is squared to 128×128 so its cosets ride the matvec giant steps
(64 baby steps). Weight diagonals are `Message`s encoded by `MatrixVectorEval` at the input scale.

### Activation

**No approximation is involved.** The model was *trained* with an exact `x²` activation
(`export_weights.py:48`, `FFNN(act="square", bn=True)`), so the homomorphic circuit computes the
network exactly — one `tensor` + `relin` + `rescale`. This removes the whole class of
approximation error the passdown §1.6 warns about; residual error is CKKS noise only
(max slot diff 0.17 on logits of order 10).

### Where our library forces a different design from the reference

1. **No `CryptoContext` object.** OpenFHE serializes one `cc.bin` that every stage loads; HEaaN2
   has no such object — each stage reconstructs `Levels`/`EnDecoder`/`HomEval` deterministically
   from constants in a shared header. Those constants become the de-facto context and must be
   identical across stages.
2. **Rotation keys are per-`(ring, modulus)`, generated against a specific level.** `makeLayer`
   currently generates keys *inside* the model build, from `sk`. The harness forbids that: the
   server has no `sk`. We must split `makeLayer` into a client half (key generation, stage 2.2)
   and a server half (diagonals + `MatrixVectorEval` construction from loaded keys).
3. **`MatrixVectorEval` encodes its diagonals in its constructor, on the device of `rot_keys`**
   — so it cannot be built in `server_preprocess_model`, which does not know the instance size
   and therefore cannot find `io/<inst>/public_keys`. Consequence: preprocessing splits into a
   size-independent plaintext part (stage 3) and a key-dependent part that necessarily lands in
   stage 7's wall-clock. We will report the split honestly via `server_reported_steps.json`.
   The reference has the same shape (its stage 3 is a no-op), so this is not a scoring advantage.
4. **Batched packing, not one ciphertext per image.** A free choice the harness permits, and the
   main source of the expected bandwidth win.
5. **Model architecture** differs from `harness/mnist/model.py` — see below.

### Deviations that must be documented in the submission README (§1.3, §7)

Full step-by-step accounting of which side every operation lands on — and the one item that is
genuinely contestable — is in **`NOTES_FOR_HUMAN.md`, "the cleartext/ciphertext boundary"**.
Summary:

| Deviation | Nature |
| --- | --- |
| Architecture `484→128→10` with `x²`, vs the harness's `784→128→64→10` with ReLU | Different trained model. The reference submission also deviates (512×784 + approx-ReLU). |
| **Center-crop 28×28 → 22×22 before encryption** | **Cleartext, client-side — the one contestable item.** The model was *trained* on cropped input, so the crop is part of the model definition and no model computation leaves the encrypted stage. But it does shrink the encrypted input 784 → 484, which is what puts 128 rather than 64 images in a ciphertext: **a 2× throughput gain traceable to a cleartext step.** Document prominently, including that consequence. |
| `(p − 0.1307)/0.3081` normalization before encryption | Cleartext client-side, identical to the reference (`client_preprocess_input.cpp:47-51`) and to the harness's own `test.py:60`. |
| argmax over 10 decrypted logits | Cleartext client-side postprocessing, identical to the reference's `client_postprocess`. §1.3 permits it explicitly. |
| BatchNorm folded into fc1 | Cleartext, **offline, model-only** — outside the per-run boundary. |

Note `client_preprocess_dataset` (upstream README line 316) **is never invoked by the harness** —
it does not exist as a stage. Our per-run client cleartext hook is `client_preprocess_input`.

### Expected outcome

Extrapolating the measured CPU eval times (HS: 122 ms per 128-image ciphertext) against the
reference's `Encrypted computation` column:

| | size 0 (1) | size 1 (100) | size 2 (1000) | size 3 (10000) |
| --- | --- | --- | --- | --- |
| ciphertexts (HS, 128/ct) | 1 | 1 | 8 | 79 |
| projected HS eval | 0.12 s | 0.12 s | ~1.0 s | ~9.6 s |
| projected PCMM eval | — | — | ~0.17 s | ~1.7 s |
| reference `Encrypted computation` | 7.61 s | 647.07 s | 6505.61 s | — |

i.e. roughly **5000×** on size 1 and **6000×** on size 2 for the crypto itself. Two caveats worth
stating before anyone quotes these:

- The harness's `Encrypted computation` number is wall-clock around the whole process
  (`utils.py:116`), so it will include key deserialization and ciphertext I/O, which at N=2^17
  will likely *dominate* the eval. The headline harness numbers will be much worse than the table
  above, and that is the honest number.
- Accuracy should land ~0.976, within noise of the harness plaintext model (~0.97). Passdown §1.6
  is satisfied: no large gap either way, so nothing here points at an approximation bug. The
  architecture deviation still gets documented per §7, but the accuracy delta itself is not
  worth commentary.

---

## Status

Phase 1 (recon) and Phase 2 (implementation) complete. The submission lives in
`submissions/mnist/`, is built against the HEaaN2 public API only, and passes all four instance
sizes through the unmodified harness on GPU:

| size | encrypted-model accuracy | harness plaintext model |
| --- | --- | --- |
| 0 single | `PASS (expected=7, got=7)` | — |
| 1 small | 0.9800 | 0.9700 |
| 2 medium | 0.9890 | 0.9820 |
| 3 large | 0.9794 | 0.9776 |

Outstanding, both tracked in `NOTES_FOR_HUMAN.md`:

- the ≥128-bit security justification, recorded as an open gap per §1.4 and awaiting crypto-side
  review;
- no official measurements generated — `measurements/` still holds the reference OpenFHE numbers.
