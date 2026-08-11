# Bench-server handoff — HEaaN2 MNIST submission

**Working doc, not part of the submission. Delete or leave uncommitted before opening the PR.**

You are on the **benchmark server** (1× RTX 5090, sm_120, exclusive access). Your job is to take
the official measurements for CryptoLab's HEaaN2 MNIST submission and commit them.

---

## 1. Context

This is a fork of the HomomorphicEncryption.org [`ml-inference`](https://github.com/fhe-benchmarking/ml-inference)
benchmark. Branch **`heaan2-mnist-submission`** replaces the reference OpenFHE implementation in
`submissions/mnist/` with one built on **HEaaN2**, CryptoLab's proprietary CKKS library. The
harness is unmodified and must stay that way.

Two circuits, same model (`fc1(128×484) → x² → fc2(10×128)`, BN folded), selected by instance
size alone (`mlp::usePcmm`):

| Size | Circuit | Why |
| --- | --- | --- |
| 0–1 (1, 100 images) | Halevi–Shoup rotation-folded matvec | faster arithmetic at small batches |
| 2–3 (1000, 10000) | PCMM (GEMM-based) | no rotation keys → ~20× less setup, which is what the harness scores |

Read [`submissions/mnist/DESIGN.md`](submissions/mnist/DESIGN.md) for how it works,
[`submissions/mnist/BUILDING.md`](submissions/mnist/BUILDING.md) for build details.

**A sibling submission is the template for conventions**: CryptoLab's
[Zn-multiplication](https://github.com/CryptoLabInc/Zn-multiplication/tree/CryptoLabInc)
(vendored prebuilt library + verification-only LICENSE + committed measurements for all sizes).

---

## 2. Precondition — get the latest code

The previous session left **8 files modified and uncommitted** on the dev machine. If those were
not committed and pushed before you started, you do not have the warm-up or the timer
synchronization, and any measurements you take will be wrong.

Verify first:

```bash
git log --oneline -3
git status --short
grep -c cudaDeviceSynchronize submissions/mnist/src/server_encrypted_compute.cpp   # must be >= 1
grep -c "Warm-up" submissions/mnist/src/server_encrypted_compute.cpp              # must be >= 1
```

If `cudaDeviceSynchronize` is absent, **stop** and ask the user to push the dev-machine changes.

### What those changes were

1. **Warm-up pass in stage 7** (both schemes) — one full inference, result discarded, before the
   timed evaluation. Mirrors HEaaN2's own mlp benchmarks. Reported as its own line in
   `server_reported_steps.json`.
2. **`cudaDeviceSynchronize()` in the stage-7 timers** — kernel launches are async; without this
   the evaluation timer closed before the GPU finished and under-reported PCMM by ~4.7×. Only the
   submission's self-reported breakdown was affected; the harness's `Encrypted computation` was
   always honest.
3. Six stale `README.md` → `DESIGN.md` cross-references in code comments.
4. `weights/manifest.txt` — corrected the `/255` normalization note and a training-repo path.
5. Top-level `README.md` — rewritten as a submission page, **with the two timing rows left
   deliberately blank for you to fill in** (see step 5 below).

---

## 3. Build

**No HEaaN2 source checkout, private-repo access or SSH key is needed.** The vendored
`submissions/mnist/install/` holds a prebuilt HEaaN2 v0.2.0 **built for sm_120**, which is exactly
this machine. Just build:

```bash
python -m venv bmenv && source ./bmenv/bin/activate
pip install -r requirements.txt

./scripts/build_task.sh ./submissions/mnist
```

If using conda for the toolchain: **activate conda first, the venv second** — the harness spawns
stages via `python3` resolved through `PATH`, and a non-active venv leaves them without `torch`.

### Pre-flight: confirm the GPU and the library agree

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv     # expect compute_cap 12.0
cuobjdump --list-elf submissions/mnist/install/lib/libheaan2.so.0.2.0 \
  | grep -oE 'sm_[0-9]+' | sort -u                       # expect exactly: sm_120
```

The vendored library has **sm_120 cubins only and no PTX** — no JIT fallback. On any other
architecture the first homomorphic op dies with "no kernel image is available for execution on the
device". A PTX-JIT library would report multi-second first-run costs as evaluation time and is not
a valid source of timings, which is why this check gates everything.

**Do not rebuild HEaaN2 from source here.** (On the 4090 dev box that was necessary because the
vendored lib is sm_120-only *and* needs glibc 2.38. Irrelevant on this machine.)

**A stale build tree is sticky**: if anything looks wrong, `rm -rf submissions/mnist/build` and
rebuild — `build_task.sh` reuses the CMake cache otherwise.

---

## 4. Take the measurements

This is the deliverable. Upstream's process ([`measurements/README.md`](measurements/README.md)):
run each variant with `--num_runs 3`; the average of the three is what gets reported.

```bash
# prefix each with srun if this machine is behind a scheduler
python3 harness/run_submission.py 0 --seed 3 --num_runs 3    # single (1 image)
python3 harness/run_submission.py 1 --seed 3 --num_runs 3    # small  (100)
python3 harness/run_submission.py 2 --seed 3 --num_runs 3    # medium (1000)
python3 harness/run_submission.py 3 --seed 3 --num_runs 3    # large  (10000)
```

Seed 3 is what every documented figure in DESIGN.md uses — keep it for consistency.

Results land in `measurements/{single,small,medium,large}/results-{1,2,3}.json`.

**Important — `measurements/` currently holds the reference OpenFHE numbers from February 2026,
committed upstream.** Your runs overwrite `single/`, `small/`, `medium/`; `large/` does not exist
upstream and will be created (Zn-multiplication committed all four, so do the same). Overwriting
them is correct and intended — that is how the fork reports its own numbers.

Sanity-check as you go:
- Accuracy should be ≈ 0.98 at sizes 1–3 (size 0 is a single image — the harness reports PASS and
  writes no Quality block).
- `server_reported_steps.json` should show three lines: model setup, warm-up, encrypted computation.
- HS (sizes 0–1) setup is expected to dominate — on this machine roughly 4–5 s, against a warm
  evaluation on the order of a millisecond. That is expected and explained in DESIGN.md §4.

---

## 5. Update the docs with the real numbers

### a. Top-level `README.md`

The results table has two intentionally blank rows — **Encrypted computation** and **Total
latency**. Fill both from the committed runs (mean of the three). Bandwidth and accuracy rows are
already filled and are machine-independent; correct them if your runs disagree.

Also confirm the stated hardware line matches this machine.

### b. `submissions/mnist/DESIGN.md` §4

The "As shipped" table there is **doubly stale**: taken before the warm-up and before the timer
sync, so its evaluation figures are both cold and unsynchronized. Re-take every number from your
runs. Then update the note directly beneath it, which currently says the measurements are pending.

The §4 line "**Not official measurements**: `measurements/` still holds the reference OpenFHE
numbers" becomes false once you commit — fix it.

### c. Leave alone

- **§5 Security** — the ≥128-bit justification is genuinely unfinished (uniform-ternary `hw = 0`
  is off the edge of the public sparse-key table that Zn-multiplication cites). It is deliberately
  disclosed as pending in README, DESIGN and `mlp_params.hpp`. Not a measurement task; do not
  quietly soften the wording.
- The **cross-check subsection** in §4 comparing against HEaaN2's own mlp benchmarks was measured
  on the 4090 dev box and says so explicitly. Leave it, or re-take it here only if you also build
  the library's benchmarks (not required).

---

## 6. Commit

```bash
git add measurements/ README.md submissions/mnist/DESIGN.md
git commit    # do NOT commit BENCH_HANDOFF.md
```

Keep `submissions/mnist/build/` out (already gitignored).

---

## 7. Known-good reference points (1× RTX 4090 dev box, sm_89 source build)

Not comparable to this machine — a 4090 measured ~3× slower than a 5090 at instance size 0 — but
useful as a shape check. Same weights, seed 3, warm and synchronized:

| | Setup | Warm-up | Evaluation |
| --- | ---: | ---: | ---: |
| size 0 (HS) | 9.36 s | 0.047 s | 1.44 ms |
| size 2 (PCMM, 1000 img) | 0.12 s | 0.056 s | 1.19 ms |

Accuracy at size 2 was 0.989 (989/1000), harness plaintext model 0.980.

Those evaluation figures match HEaaN2's own benchmarks on the same box (1.434 ± 0.044 ms HS,
1.378 ± 0.105 ms PCMM), which is the evidence that the submission adds no overhead over calling
the library directly.

---

## 8. Open items you are *not* expected to resolve

- **Security analysis** (§5) — pending, deliberately disclosed.
- **Submission location** — Zn-multiplication lives in the CryptoLabInc org
  (`CryptoLabInc/Zn-multiplication`, branch `CryptoLabInc`); this one is on a personal fork
  (`yongwonchoi-Cryptolab/ml-inference`, branch `heaan2-mnist-submission`). Whether it must move
  or be mirrored to a CryptoLabInc-org fork before registration is the user's call.

## Ground rules

- **Never modify `harness/`** — an unmodified harness is the core claim of the submission.
- Report what you measure. If a run is anomalous, say so and re-run; do not average away a
  problem or quote a figure you did not observe.
- If the pre-flight architecture check fails, stop and report — do not work around it.
