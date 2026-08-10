# Notes for the human — HEaaN2 ml-inference submission

Companion to `RECON.md`. Phase 1 (recon), Phase 2 (HS implementation for single/small) and Phase 3
(PCMM implementation for medium/large, added on your instruction) are complete; all four instance
sizes pass through the unmodified harness. Latest round's decisions and questions are answered
first; Phase 2's round-1 notes follow below that.

---

## Phase 3: PCMM added for medium/large, on your instruction

You asked for PCMM at sizes 2/3 specifically because you had already measured it as faster than HS
on GPU, on hardware other than this shared dev box, and separately noted this dev machine should be
treated as CPU-only for this round. Both are now true: implemented, and every claim below is
**correctness**-verified; performance is reported honestly including one number that came out
looking wrong and was not chased down (see the GPU flag below).

### What changed

- `include/mlp_pcmm.hpp` + `src/mlp_pcmm.cpp` (new): the PCMM scheme, written independently against
  HEaaN2's public API (`HomEvalMatrix::pcmm`, `ICtMatrix`/`IPtMatrix`, `Matrix<Real>`) — no HEaaN2
  source vendored, same rule as Phase 2. HEaaN2's own `mlp/PCMM` benchmark was read as a design
  reference (the pcmm circuit, the section-6.3 coeff/slot relabeling trick, the bias-fold-as-extra-
  column layout), the same relationship Phase 2 had with `mlp/Halevi–Shoup`.
- `include/mlp_params.hpp`: added a `mlp::pcmm` sub-namespace (ring/level/security constants) and
  `mlp::usePcmm(InstanceSize)`; extended the shared `maxBits128` table with a 2^12 entry PCMM needs.
- All seven stage `.cpp` files now start with `if (usePcmm(size)) { ... } else { ... }` (four of
  them: `client_key_generation`, `client_encode_encrypt_input`, `server_encrypted_compute`,
  `client_decrypt_decode`). `client_preprocess_input` and `client_postprocess` needed **no changes
  at all** — both schemes agree on the input/output file formats, only the crypto in between
  differs. `server_preprocess_model` gained a few lines to also cache raw (unpadded) weights, since
  it runs unconditionally regardless of size.
- `CMakeLists.txt`: one line, `src/mlp_pcmm.cpp` added to the existing `mlp_pipeline` library.
  Nothing else about the build changed.

### Correctness — verified on CPU (all sizes) and GPU (accuracy only, size 2)

Ran the **full, unmodified harness** — not just the standalone binaries — for every combination:

| size | scheme | machine | encrypted-model accuracy | harness plaintext model |
| --- | --- | --- | --- | --- |
| 0 single | HS | GPU (regression) | `PASS (expected=7, got=7)` | — |
| 1 small | HS | CPU + GPU (regression) | 0.9800 (both) | 0.9700 |
| 2 medium | PCMM | CPU | **0.9890** (989/1000) | 0.9820 |
| 2 medium | PCMM | GPU | **0.9880** (988/1000) | 0.9820 |
| 3 large | PCMM | CPU | **0.9796** (9796/10000) | 0.9776 |

The CPU runs used a **separate HEaaN2 install** (`~/HEaaN2/install-cpu`, `BUILD_WITH_CUDA=OFF`) so
as not to disturb the GPU install your official measurements will use; I stashed and restored
`submissions/mnist/build` (the GPU one) around the CPU testing so it ends this session exactly as
it was before. Every test run also clobbered `measurements/<size>/results-*.json`; all were
restored via `git checkout -- measurements/`, and a stray `measurements/large/` the size-3 run
created (it didn't exist before) was deleted. `measurements/` is clean, still the reference's.

I also independently re-derived the packing/level-schedule arithmetic while writing this (it's in
the "Something I caught myself" note below) rather than trusting a first draft — worth knowing
since PCMM's level schedule has one more moving part than HS's (the relin key sits at a *different*
level than its input, because `x^2` here runs tensor→rescale→relin instead of the more usual
tensor→relin→rescale).

### The one thing that looked wrong: PCMM's GPU eval time

Size 2 on GPU: **9.34 s** of server-reported eval, against **0.267 s** for the identical operation
sequence on CPU — GPU **~35× slower**, the wrong direction, and inconsistent with HEaaN2's own
`MLInference_Large` benchmark (2.54 ms/1000 images on its own hardware) and with your own prior
finding that PCMM beats HS on GPU. Accuracy was correct both ways (0.988 GPU vs 0.989 CPU — a
half-point gap from independent encryption noise on the same seed, not a bug), so this reads as a
performance anomaly, not a correctness one.

**I did not chase this down.** Time-boxing reasons: you'd already validated PCMM-beats-HS-on-GPU
yourself, on different hardware, which is why PCMM was worth building at all — so this dev box's
GPU number disagreeing with that doesn't call the design into question, it calls *this shared
node's GPU* into question, and that's exactly the kind of investigation that's cheap to go down and
expensive to be wrong about without instrumentation I didn't have time to add (proper CUDA
profiling, ruling out node contention from other users). My best guess, offered as a hypothesis and
nothing more: PCMM's `pcmm`/`tensor`/`relin` calls at N=2^13 are individually tiny next to HS's at
N=2^17, so fixed per-kernel-launch overhead that HS's larger ops amortize away might dominate here
— and this RTX 5090 (sm_120/Blackwell) is new enough that a research library's kernels may not be
tuned for it yet. **Please re-measure PCMM on the actual bench server before quoting any GPU number
for it**, and don't take this dev box's 9.34 s as representative of anything.

### Encryption asymmetry — disclosed, not a shortcut

HEaaN2's public `EnDecryptor` has a public-key overload for plain ciphertexts but **only a
secret-key overload for matrices** (confirmed by reading `EnDecryptor.hpp`: no `IEncKey` overload
for `ICtMatrix` exists in the public API). So HS encrypts under a public key (as before); PCMM
necessarily encrypts under the client's own secret key. Both are exclusively client-side — the
secret key never leaves `seckeydir()`, which the harness doesn't measure — but it's a real,
API-forced difference between the two schemes, not a design choice, and it's called out plainly in
the submission README rather than left for a reviewer to notice.

### Security — same "open gap" status as Phase 2, now covering PCMM too

PCMM's parameters are *simpler* to eventually justify than HS's: no lifted-key construction to
review (the secret key is sampled directly at its working ring, N=2^13), security following
straight from `maxBits128(12) = 106` bits against a ~94-bit modulus chain. This was flagged back in
round 1 as the fallback if HS's lifted-key construction got rejected — it's now built, not just a
contingency plan. Still an open gap either way, per your round-1 answer ("assume it is safe for
now"): the README's security section covers both schemes' parameters with the same "not signed
off" framing, nothing asserted as reviewed.

### Cleanup

Two scratch build directories I used only to verify the CPU path (`submissions/mnist/build-cpu`,
`build-cpu-harness`) were deleted at the end of this session — they were never staged and aren't
part of the submission. The CPU HEaaN2 install (`~/HEaaN2/install-cpu`) was left in place outside
the repo in case you want to re-run the CPU comparison later without a fresh 15-ish-minute rebuild.

---

## Phase 2 status: implemented, HS covers single/small

The submission is written and working end to end through the **unmodified harness** on GPU.

| size | encrypted-model accuracy | harness plaintext model | server-reported eval |
| --- | --- | --- | --- |
| 0 single | `PASS (expected=7, got=7)` | — | 0.031 s |
| 1 small (100) | 0.9800 | 0.9700 | 0.032 s |

(Sizes 2/3 moved to PCMM in Phase 3 above; their original HS-path numbers — 0.9890/0.021s at size 2,
0.9794/0.059s at size 3 — are no longer what ships, kept here only as the historical record of what
was measured before the scheme switch.)

Numerics verified independently against a NumPy forward pass on the size-0 input:
max |decrypted − plaintext logit| = 0.109 on logits spanning [−32, +14], argmax identical.

Key material 174.8 M (reference: 1.0 G). HS's encrypted input at size 1 is 1.6 M; see Phase 3 above
for PCMM's corresponding size 2/3 figures, which are larger, not smaller (bandwidth, unlike compute,
did not obviously favor PCMM here — worth knowing before assuming PCMM wins on every axis).

### Three things you needed to know before the first official run

1. **Every harness run overwrites `measurements/<size>/results-<n>.json`.** My development runs
   clobbered the reference OpenFHE numbers; I restored them with `git checkout -- measurements/`
   after each run and also removed the `measurements/large/` directory the size-3 run created
   (it did not exist before). `measurements/` is currently **clean and still the reference's**.
   Per passdown §6 I have generated no official measurements.

2. **The run environment needs both toolchains on PATH, in this order.** The conda env supplies
   `cmake`; the venv must supply `python3` (the conda env has no torch):
   ```bash
   export PATH=/home/yongwonchoi/miniconda3/envs/heaven-dev-cuda/bin:$PATH
   source bmenv/bin/activate
   export HEAAN2_ROOT=/home/yongwonchoi/HEaaN2
   srun python3 harness/run_submission.py <size> --seed 3
   ```
   Getting the order backwards fails at stage 1 with `ModuleNotFoundError: No module named 'torch'`.

3. **Fixed setup dominates every measured number, and it is the obvious next optimization.**
   The harness's `Encrypted computation` is ~4.6 s at *every* size, of which only 21–59 ms is
   evaluation; the rest is deserializing 174.8 M of rotation keys and encoding the weight
   diagonals, both of which happen inside stage 7 because `server_preprocess_model` is invoked
   with no instance-size argument and so cannot find `io/<size>/public_keys`. Worth discussing
   before measurements: it is entirely legitimate to shrink, but not by moving encrypted work.

### Deliberately not done yet (passdown §5.7 — optimize last)

- ~~PCMM variant for sizes 2–3.~~ **Done in Phase 3** — see the top of this file.
- Reducing HS's setup cost described above.
- Multi-threading the per-ciphertext loop in stage 7 (HS) / investigating PCMM's GPU eval time
  (Phase 3) before either is worth optimizing further.

---

## Resolved in review round 1

| | Decision |
| --- | --- |
| **Execution** | GPU. Shared machine → every run goes through `srun`. Verified: `srun ./build/mlp/MLInference_128 3 12345` → **7.52 µs/image**, 97.66%. |
| **Category** | **GPU / hardware-backed** submission. |
| **Licensing** | HEaaN2's **public API only**. No HEaaN2 source is copied into this fork. Pattern to follow: the `zn-mul` branch's `submission/` dir — `find_package(HEaaN2 REQUIRED)`, link `HEaaN2::HEaaN2`, own `src/` + `include/`. I can access the branch; read it. |
| **Security claim** | Recorded as an **open gap**, not asserted. See below. |
| **Submission dir** | `submissions/mnist/`. |
| **cblas** | `conda install conda-forge::openblas` into `heaven-dev-g++` — **done and verified**, `cblas.h` present. |
| **Plaintext accuracy** | Won't belabour the delta. Architecture deviation still gets documented (§7 requires it); the accuracy comparison just won't be editorialised. |

### The `srun` consequence — worth knowing before you run anything

The harness calls our stage binaries **directly**, so `submissions/mnist/build/server_encrypted_compute`
run bare will abort with *"CUDA device is not available in the current environment"*.

Verified fix, requires **no harness change**: launch the whole harness under `srun` and every
nested child inherits the allocation. Tested `srun → python3 → subprocess → CUDA binary`, rc=0,
7.64 µs/image.

```bash
srun python3 harness/run_submission.py 0 --seed 3
```

This becomes the documented run command in the submission README. Without `srun` the run fails at
stage 7 — loudly, which is the good case.

### Security claim — recorded as an open gap, per passdown §1.4

Your answer: *"the 128-bit security will be provided later and assume it is safe for now."*

Passdown §1.4 says that if we cannot derive a justification we **record the gap rather than invent
a claim**, so that is what I will do. Concretely:

- The submission README will **not** assert ≥128-bit security as settled. It will state the
  parameters, state that the security analysis is pending crypto-side review, and cite what the
  in-code table currently claims (HEaven `maxBitsPolicy128()` for N=2^13–2^15, ePrint 2024/463 for
  2^16–2^17) as provisional.
- `hw` may change → I will make hardness/`hw`, `SMALL_LOG_DEGREE`, and the mod-up budget **named
  constants in one header**, so a later review can change them in one place without touching the
  pipeline.
- **This must be closed before the submission goes public.** A benchmark submission's parameter
  claim is read adversarially, and §1.4 makes it a validity condition. Please ping me when the
  crypto-side review lands so I can replace the placeholder text with the real justification.

Flagging one knock-on: if the review rejects the **lifted-key construction** (sample at 2^15 →
`genHighDegreeKey` to 2^17, enabling the key-less `frobMap` fold), the HS variant loses its main
optimization at single/small. This is no longer purely hypothetical scope: **PCMM is now built**
(Phase 3, top of this file) and covers medium/large already, taking security straight from the RLWE
dimension with no lifting to review. If HS's lifted key is rejected, the honest next question is
whether PCMM should also take over single/small, or whether single/small should get a fresh,
non-lifted HS parameter set instead — both are options, neither is done, and it's your call once
the review lands.

---

## Your question: `client_preprocess_dataset`, and the cleartext/ciphertext boundary

### Short answer on `client_preprocess_dataset`

**We don't handle it, because it does not exist.** It appears only in the upstream README's stage
table (line 316), described as *"(Optional) Any in the clear computations the client wants to apply
over the dataset/model."* `run_submission.py` never calls it — grep the harness and there is no
such invocation. If we created `submissions/mnist/client_preprocess_dataset.py`, it would simply
never run. Same for `client_encode_encrypt_query` (README's name for
`client_encode_encrypt_input`). This is the passdown's Q2 drift.

The harness's real per-run client-side cleartext hook is **`client_preprocess_input <size>`**, and
that is where our normalization and crop go.

### The full accounting — every step, which side, cleartext or ciphertext

Harness-owned steps marked *(harness)* are not ours to place.

| # | Step | Side | Domain | Questionable? |
| --- | --- | --- | --- | --- |
| 1 | `generate_dataset` — download MNIST 10k | *(harness)* | cleartext | no |
| 2.2 | `client_key_generation` — sk, enc key, rot/relin keys | client | cleartext (key material) | no |
| 3 | `server_preprocess_model` — parse W/b CSV, validate, cache in padded p×q layout | server | cleartext | no — the server **owns** the model in this benchmark; it was never secret |
| 7a | build + encode the weight diagonals (in stage 7; stage 3 gets no instance size) | server | cleartext model → plaintexts | no — model-only, no input involved |
| 4 | `generate_input` — draw the batch, write `test_pixels.txt` | *(harness)* | cleartext | no |
| 5a | normalize `(p − 0.1307)/0.3081` | client | cleartext | no — reference does the identical thing (`client_preprocess_input.cpp:47-51`), as does the harness's own `test.py:60` |
| 5b | **center-crop 28×28 → 22×22 (784 → 484)** | client | cleartext | **YES — see below** |
| 6 | pack 128 images into slots, encode, encrypt | client | cleartext → ciphertext | no — data layout, not computation |
| 7 | fc1 matvec · fold · +b1 · **x²** · rescale · fc2 matvec · +b2 | server | **ciphertext** | no — this is the whole inference, all of it encrypted |
| 8 | decrypt, decode, extract 10 logits/image from slots | client | ciphertext → cleartext | no |
| 9 | `client_postprocess` — argmax over 10 logits | client | cleartext | no — §1.3 explicitly permits post-decryption processing; reference does the same |

Also cleartext but **offline and model-only**, so outside the per-run boundary: BatchNorm is folded
into fc1 at weight-export time (standard eval-mode BN folding; `export_weights.py:63-72`).

### The one questionable item: the crop

I want this stated plainly rather than buried, because it is the thing a reviewer will challenge.

**What it is.** The client crops 3 pixels off each side in the clear before encrypting, so the
encrypted input is 484-dimensional, not 784.

**The case that it is legitimate.** The model was *trained* on 22×22 input — the crop is part of
the model definition, not an optimization. fc1 is literally a 128×484 matrix; there is no 784-input
model to run. An FHE implementation of a 484-input network must feed it 484 inputs. Nothing that
the model computes has been moved out of the encrypted stage: the entire circuit
(fc1 → x² → fc2) runs on ciphertext.

**The case against, which is not frivolous.** The benchmark is nominally "MNIST inference", and
MNIST is 784 pixels. A skeptic reads a cleartext dimensionality reduction from 784 → 484 as making
the encrypted problem smaller, and the objection has teeth because **it buys real throughput**:
484 pads to 512 coordinates rather than 1024, which is exactly what puts 128 images in a
ciphertext at N=2^17 instead of 64. That is a **2× throughput gain traceable to a cleartext step**.

**My recommendation.** Document it prominently — not in a footnote — with the framing above, and
be explicit about the 2× packing consequence rather than letting a reader discover it. If you want
the objection gone entirely, the alternative is training a 784-input variant and eating the 2×;
that is your call and I'd want it made before measurements, not after. Passdown §1.3 permits the
crop as long as it is documented, so this is a presentation-risk judgement, not a validity one.

---

## Your question: why I said "rewrite `scripts/get_openfhe.sh`"

I was reasoning from the passdown's Q4, which assumed we would **drop OpenFHE entirely** — in
which case `get_openfhe.sh` clones and builds OpenFHE for nothing on every fresh checkout, and the
harness calls it unconditionally with `check=True` (`utils.py:103`), so it cannot just be deleted.

**That reasoning was incomplete, and I'm dropping the plan.** I checked:
`submissions/cifar10/CMakeLists.txt:19` does `find_package(OpenFHE CONFIG REQUIRED)`. The cifar10
submission — which is not ours and which passdown §8 says to leave alone — **needs** OpenFHE.
Neutering `get_openfhe.sh` would break it, and the script takes no arguments so it cannot tell
which dataset is being built.

**Revised plan:**
- **`scripts/get_openfhe.sh` — leave completely untouched.** It already early-exits when
  `third_party/openfhe/lib` exists, so the cost is one OpenFHE build per fresh clone. Wasted for
  our submission, harmless, and it keeps cifar10 working.
- **`scripts/build_task.sh` — dispatch on its `$1` (the task dir)**, which it already receives:
  `submissions/mnist` → our HEaaN2/CMake path; anything else → the existing OpenFHE + libtorch
  path, byte-identical to today. Additive change, no existing behaviour altered.

Net effect: the libtorch download stays for cifar10 but is skipped for us, and no unrelated
submission breaks. If you'd rather pay zero OpenFHE cost on a fresh clone, the only clean lever is
an env-var guard (`SKIP_OPENFHE=1`) that defaults to today's behaviour — say the word and I'll add
it, but I'd rather not without a reason.

---

## Still open — one licensing detail I need to get right

You said: *public functions and classes are fine, but nothing from HEaaN2's actual code base.*
Understood — I will write the inference pipeline fresh against the public API rather than copying
`mlp/InferenceCommon.hpp`, `mlp/PCMM/Common.hpp`, or the `MLInference_*.cpp` files.

**The model weights are the grey area.** The submission needs a model, and
`HEaaN2/mlp/Common/weights/*.csv` are CryptoLab-produced artifacts. Two options:

1. **Regenerate in this fork** with our own training script (the recipe is public: 2-layer
   784→128→10, x² activation, BatchNorm, 30 epochs, seed 0, BN folded at export, center-crop to
   484). Clean provenance, and it makes the fork self-contained and reproducible — which §7 wants
   anyway. Retraining gives slightly different but equivalent weights.
2. Copy the CSVs and treat them as data rather than code.

**I plan on (1)** unless you object — it costs one training run and removes the question entirely.
Tell me if the trained weights are considered proprietary regardless of who runs the script.

---

## Other non-blocking findings (unchanged from round 1)

### The passdown is stale where it describes the tree

- There is **no `submissions/mlp/`** — this fork has `submissions/mnist/` and
  `submissions/cifar10/` (upstream PR #25, `add-resnet20`).
- The submission dir is keyed on the **dataset**, not the model
  (`model_exec_dir = submissions/<dataset_name>`, `run_submission.py:44`). `--model` is a free-text
  label that only reaches the results JSON. Passdown Q1's worry does not arise.
- `--dataset`/`--model` have **no `choices=`** despite the upstream README's usage line.

### A harness trap

`utils.run_exe_or_python` silently does **nothing** if neither `<stage>.py` nor `build/<stage>`
exists — no exception, no message. A misnamed target yields a green run with stale output rather
than a build error. I'll add a post-build check that all seven stage binaries exist.

### Measurement hygiene

`measurements/` untouched. Per §6 no official measurements until you confirm the environment.
Note `harness/mnist/mnist_ffnn_model.pth` is **not** in the repo, so the first `size ≥ 1` run
trains the harness's comparison model for 15 epochs before it can score us.
