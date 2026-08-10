# Notes for the human — HEaaN2 ml-inference submission

Companion to `RECON.md`. Phase 1 (recon), Phase 2 (HS for single/small), Phase 3 (PCMM for
medium/large) and Phase 4 (aligning with the Zn-multiplication precedent) are complete; all four
instance sizes pass through the unmodified harness. Newest phase first.

**Read Phase 4 first if you read nothing else** — it retracts a Phase 3 measurement, root-causes it
to a mis-built CUDA library that was also inflating every other timing in this file, and flags two
decisions I need from you.

---

## Phase 4: aligned with the Zn-multiplication precedent — and found two real problems doing it

You pointed me at the [`zn-mul` HEaaN2 branch](https://github.com/CryptoLabInc/HEaaN2/tree/zn-mul)
and the [Zn-multiplication submission](https://github.com/CryptoLabInc/Zn-multiplication/tree/CryptoLabInc).
I read both in full. The structural lesson transferred cleanly; two things I checked *while*
transferring it turned out to be broken, and both mattered more than the restructuring did.

### What I adopted from the precedent

The sibling submission vendors a **prebuilt HEaaN2** in-tree (`submission/install/` — headers, the
`.so`, and the CMake package config) plus a `LICENSE` permitting redistribution solely for
benchmark reproduction. Its `build_task.sh` is then five lines: no repo cloning, no SSH key, no
`nvcc` hunting. That is a much better replication story than what we had, so:

- **`submissions/mnist/install/`** now holds a prebuilt HEaaN2 (headers + `libheaan2.so.0.2.0`,
  84 MB). A bare clone of this fork builds with **no private-repo access, no SSH key, and no HEaaN2
  checkout.** Verified by building with `HEAAN2_ROOT`/`HEAAN2_DIR` unset.
- **`LICENSE`** — copied **verbatim** from the sibling submission. It is CryptoLab's own legal text;
  editing it was not mine to do. Please confirm it is the current version you want to ship.
- **`LICENSE-THIRD-PARTY`** — *not* copied, because ours is a **CUDA** build and theirs is CPU-only
  (their `build_task.sh` passes `-DBUILD_WITH_CUDA=OFF`, which is why their `.so` is 5.5 MB against
  our 84 MB). I inspected our actual binary (`ldd`, `nm -D`, `strings`) and listed what it really
  depends on: CUDA runtime + cuBLAS, `libtcmalloc`, OpenMP, libstdc++. Neither of the components
  their file lists (GNU Quadmath, the Keccak/tweetfips202 code) appears in our binary's symbol
  table, so copying their list would have been wrong in both directions.
- **`build_task.sh`** now defaults to the vendored install; setting `HEAAN2_ROOT` opts back into the
  from-source path (kept intact for rebuilding for a different GPU, or CPU-only).
- **`CMakeLists.txt`** defaults `CMAKE_PREFIX_PATH` to `${CMAKE_CURRENT_SOURCE_DIR}/install`, same
  as theirs.

I did **not** adopt their directory layout (`submission/` at repo root, binaries in
`target/release/`). Their harness is a *different* harness — it hardcodes
`params.rootdir/"submission"/"target"/"release"` — whereas ours resolves
`submissions/<dataset_name>/build/<stage>`. Copying their layout would break our harness contract.
Same reasoning for their `<instance>`-string argv: ours is passed a numeric size.

### Problem 1: the vendored library was built for the wrong GPU — and that was the "anomaly"

Before shipping the binary I ran `cuobjdump` on it rather than trusting how it had been configured.
It contained **sm_52 cubins and `compute_52` PTX only** — CMake's *default* architecture — despite
the build being configured with `native` on a 5090.

This is precisely the trap documented in `BUILDING.md`: HEaven's architecture fallback

```cmake
if(DEFINED CACHE{CMAKE_CUDA_ARCHITECTURES}) ... else() "75-real;80-real;89-real;120-real"
```

is unreachable, because CMake seeds that cache entry with `52` the instant CUDA is enabled. So the
library still ran on the 5090 — by **JIT-compiling PTX at load**.

**That is the real explanation for the ~9.3 s PCMM GPU evaluation I reported in Phase 3 and wrote up
as an unexplained transient.** It was one-time PTX JIT, cached afterward in `~/.nv/ComputeCache`,
which is exactly why it looked unreproducible when I re-ran it. My Phase 3 write-up guessed at
kernel-launch overhead on a new architecture and said it "was not chased down." The guess was wrong
and the number was an artifact of a mis-built dependency, not a property of the code.

Rebuilt with `CMAKE_CUDA_ARCHITECTURES=120-real` (20/20 cubins sm_120, no PTX) and re-vendored:

| size 2 PCMM evaluation | sm_52 + PTX JIT | sm_120 native |
| --- | --- | --- |
| first run, cold JIT cache | ~9.3 s | **0.046 s** |
| subsequent runs | 0.109–0.124 s | 0.040–0.046 s |

So the mis-build cost ~2.6× once warm *and* hid a multi-second first-run cliff whose absence
depended on a cache outside the repo. **Every timing in `DESIGN.md` is now from the sm_120 build**,
and the retraction is recorded there in full rather than quietly corrected.

Worth noting the same mis-build was inflating the HS numbers too (size 0/1 evaluation went
0.031 s → 0.016 s), so the Phase 2/3 figures were all measured on a hobbled library.

### The A/B you should actually look at

With a correct library, on the same node, same session, HS forced at sizes 2/3 for comparison:

| | HS setup | HS eval | HS scored | PCMM setup | PCMM eval | PCMM scored |
| --- | --- | --- | --- | --- | --- | --- |
| size 2 (1000) | 4.34–4.39 s | **0.018 s** | 4.55–4.60 s | 0.19–0.22 s | 0.040–0.046 s | **0.44 s** |
| size 3 (10000) | 4.35 s | 0.044 s | 4.66 s | 0.20 s | **0.042 s** | **0.57 s** |

Your instruction to use PCMM at medium/large holds, but the mechanism is worth being precise about:
**PCMM wins on setup, not arithmetic.** It needs no rotation keys at all (54 K of key material vs
HS's 174.8 M), so it skips the ~4.4 s of key deserialization + diagonal encoding HS re-pays every
run. On *pure evaluation* HS is still 2.4× faster at 1000 images, and the two only reach parity at
10000. Since the harness scores wall-clock around the whole stage-7 process, PCMM is 8–10× ahead on
the number that counts — but if HS's setup were ever amortized, HS would be the better choice at
size 2. Also: PCMM's encrypted **input is 3.4× larger** (44.5 M vs 13.1 M), so it does not dominate
on every axis.

### Problem 2: the sibling submission's security citation does not transfer to our parameters

Their security section closes with a real citation:
[sparse-key-estimate](https://github.com/jdumezy/sparse-key-estimate/blob/master/Precomputed-Tables/128bits_security.md),
and their numbers check out exactly — N = 2^16, hw = 32, log(PQ) = 114 ≤ the table's
`logn=16, h=32` entry of 349. I fetched the table to see whether we could lean on the same source.

**We cannot, as currently parameterised.** That table is indexed by *sparse* Hamming weights
(columns h ∈ {32, 64, 128, 192, 256, 512, 1024}). Both our schemes use `hw = 0`, which in HEaaN2
means a **uniform-ternary (dense)** secret — h ≈ 2n/3, off the right edge of the table. There is no
row to read.

Our `maxBits128` values sit near its densest column but don't match it (our 2^13 entry 214 equals
its `h=1024` entry exactly; our 2^14 is 430 vs its 426; 2^15 is 868 vs 854), consistent with their
documented provenance in HEaven's `maxBitsPolicy128()` rather than in this table. A dense key *is*
at least as hard as an h=1024 one at the same (n, q), so reading off that column would be
conservative — but that is an argument a reviewer accepts, not a citation, and the numeric drift
shows the two are different analyses.

Two coherent ways to close it, and it's a decision, not an oversight:

1. **Keep `hw = 0`** and cite something valid for a dense key — a lattice-estimator run at our exact
   (n, q, σ) points, or the HE standard. This is what the pending crypto review needs to produce.
2. **Switch to a sparse key** (hw = 32) so the sibling submission's citation applies directly. This
   is a real parameter change, not a docs edit: it changes noise growth, so the level schedule,
   accuracy, and bottom-modulus headroom all need re-validating.

Written up in `DESIGN.md §5`. Flagging it here because "the sibling submission already solved the
security citation" is the natural assumption, and it isn't true for our parameter choice.

### On the `hem-heaan-mlinf-submission` branch you offered

**Not needed — I created no HEaaN2 branch.** I checked whether anything required a library change:
the `zn-mul` branch's own diff against `dev` touches *only* files under `submission/` (its
`git diff --stat origin/dev...origin/zn-mul -- . ':!submission'` is empty), i.e. even the sibling
submission needed no core library edits. Ours likewise builds entirely against the public API. The
one library-side thing we depend on, `HomEval::frobMap`, already exists on `hem-heaan-mlinf` (and on
`hem-heaan-mlinf-frobmap` at `5502b04`, which the vendored binary was built from). If the crypto
review later forces a parameter change that needs new library surface, that's when the branch will
be worth making.

### Two things I need you to confirm

1. **The bench server's GPU.** The vendored library is sm_120-only, with no PTX fallback — so on any
   other architecture it will fail at load rather than run slowly. If the bench server isn't a 5090,
   `install/` must be rebuilt for it before measurements (`HEAAN2_CUDA_ARCH=<arch>`, instructions in
   `BUILDING.md`). Given what Problem 1 turned out to be, I'd verify with `cuobjdump` on the bench
   server rather than assuming.
2. **The 84 MB binary in git.** That's over GitHub's 50 MB per-file warning threshold (under the
   100 MB hard limit). The sibling submission's is 5.5 MB because it is CPU-only. If an 84 MB blob
   in the fork is unacceptable, the options are Git LFS, a release-asset download in
   `build_task.sh`, or shipping CPU-only and requiring a from-source build for GPU — all of which
   change the replication story, so I didn't pick one unilaterally.

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
### — SUPERSEDED, see Phase 4 Problem 1: this was a mis-built library, root-caused and fixed

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
