<!-- /autoplan restore point: /c/Users/Owner/.gstack/projects/DJ-suite/mvp-mixing-console-autoplan-restore-20260811-135511.md -->
# AI Mixing Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the two deterministic mixing decisions in `public/host/engine.js` (`_pickNextTrack()`, `_transitionPlan()`) with optional learned models — a transition-parameter model distilled from a real-DJ-trained generator (DJtransGAN), and a track-selection model imitation-trained on the existing rules — both gated behind `creativeFlags` kill switches that default to the current deterministic behavior.

**Architecture:** Two offline training pipelines in a new `ml/` directory (Python + one vitest-driven data-export script) produce small ONNX models checked into `public/host/models/`. At runtime, `engine.js` loads them lazily via `onnxruntime-web` (same bare-URL-import pattern already used for `web-audio-beat-detector`) and branches on `creativeFlags.aiSelection` / `creativeFlags.aiTransition`; any load or inference failure falls back to the existing deterministic code path for the rest of the session.

**Tech Stack:** Python 3.12 + PyTorch (CPU) for training, ONNX for model interchange, `onnxruntime-web` (wasm) for browser inference, Vitest for the JS-side data export and integration tests. No new npm runtime dependency beyond `onnxruntime-web`, loaded via the project's existing bundler-less `esm.sh` pattern.

## Global Constraints

- Design spec: `docs/superpowers/specs/2026-08-06-ai-mixing-models-design.md` — every task below implements a section of it. Read it first if anything here is ambiguous.
- Both new `creativeFlags` entries (`aiSelection`, `aiTransition`) MUST default to `false`. No behavior change on merge.
- No UI control for the new flags — devtools-only, matching every existing `creativeFlags` entry (`sampling`, `novelty`, `fx`, `loopRoll`, `stems`).
- `engine.js` has no bundler (`public/host/index.html:109` loads `app.js` as a plain `<script type="module">`; `engine.js:8` imports `web-audio-beat-detector` via a bare `https://esm.sh/...` specifier). Any new browser dependency (`onnxruntime-web`) must follow the same pattern — do not add a bundler as part of this work.
- `engine.js` is itself an ES module with a `https://` bare specifier, so plain `node some-script.mjs` cannot `import` it directly (Node has no `https:` resolver). Any script that needs the real `MusicEngine` class must run through Vitest, which already mocks that import (`test/setup.js`).
- Model output values are unbounded in principle (regression/ranking nets) and MUST be clamped to the existing deterministic constants' ranges before use — a live party can't tolerate a pathological model output.
- Model quality is not unit-testable — validated by ear later, like every other mixing feature in this codebase's history. Tests in this plan cover code correctness (feature assembly, clamping, fallback), not "does the AI mix sound good."

---

### Task 1: DJtransGAN inference spike (decision gate)

**Files:**
- Create: `ml/.gitignore`
- Create: `ml/setup_djtransgan.sh`
- Create: `ml/spike_transition_inference.py`
- Create: `ml/README.md`

**Interfaces:**
- Produces: a working local DJtransGAN environment at `ml/vendor/djtransgan/` (gitignored) with pretrained weights downloaded, and a recorded answer to "does real automation-curve extraction work" that Tasks 6-8 depend on.

This task is a spike with an explicit gate, not a normal build-and-move-on task — DJtransGAN is a 2022 research repo pinning `torch==1.9.0`/`numpy==1.18.5`/`tensorflow==2.3.0`/`cupy-cuda102` in its full `requirements.txt`, which will not install on Python 3.12. Investigation during design showed the *model/mixer/frontend* code (what inference actually needs) only imports `torch`, `torchaudio`, `torchlibrosa`, `nnAudio`, and `asteroid-filterbanks` — none of the CUDA-pinned or `essentia`/`madmom` packages, which are only used by the repo's own *dataset-building* pipeline (which we don't use; we supply our own audio and cue points). This task verifies that minimal-dependency theory actually works before three downstream tasks are built on it.

- [ ] **Step 1: Create the `ml/` directory and gitignore its generated/vendored content**

```bash
mkdir -p ml
cat > ml/.gitignore << 'EOF'
venv/
vendor/
data/*
!data/.gitkeep
export/
*.pt
*.wav
*.npz
EOF
```

- [ ] **Step 2: Write the setup script**

```bash
cat > ml/setup_djtransgan.sh << 'EOF'
#!/usr/bin/env bash
# One-time setup: clone DJtransGAN, create a minimal-dependency venv, download
# pretrained weights. Deliberately does NOT use DJtransGAN's own requirements.txt
# (pins torch==1.9.0/numpy==1.18.5/tensorflow==2.3.0/cupy-cuda102 — won't install
# on modern Python). Installs only what djtransgan/model + djtransgan/mixer +
# djtransgan/frontend actually import.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d vendor/djtransgan ]; then
  mkdir -p vendor
  git clone --depth 1 https://github.com/ChenPaulYu/DJtransGAN.git vendor/djtransgan
fi

python -m venv venv
# shellcheck disable=SC1091
source venv/Scripts/activate 2>/dev/null || source venv/bin/activate

pip install --upgrade pip
pip install torch torchaudio torchlibrosa nnAudio asteroid-filterbanks \
            numpy librosa soundfile gdown onnx pandas scikit-learn

cd vendor/djtransgan
python -c "from djtransgan.utils import download_pretrained; download_pretrained()"
echo "Setup complete. Pretrained weights in vendor/djtransgan/pretrained/"
EOF
chmod +x ml/setup_djtransgan.sh
```

- [ ] **Step 3: Run the setup script**

Run: `bash ml/setup_djtransgan.sh`
Expected: completes without error; `ml/vendor/djtransgan/pretrained/djtransgan_minmax.pt` exists and is ~140MB.

- [ ] **Step 4: Write the spike script**

This runs DJtransGAN's real generator on two real tracks from `playlist/`, and — critically — inspects `mix_out` (the second return value of `generator.infer()`, a dict of `{'prev': {'band': ..., 'fader': ..., 'mask': ...}, 'next': {...}}` per `djtransgan/mixer/mixer.py`'s `Mixer.forward()`) since that's the actual automation-curve data Task 7 needs, not just the rendered audio `inference.py` saves by default.

```python
cat > ml/spike_transition_inference.py << 'EOF'
"""Spike: verify DJtransGAN's pretrained generator produces usable EQ/fader
automation curves (not just rendered audio) on our own library's tracks.
Run from ml/ with the venv active: python spike_transition_inference.py
"""
import os
import sys

# DJtransGAN's load_pt() is a bare `torch.load(in_path)` with no weights_only
# guard (checked: djtransgan/utils/utils.py). Forcing weights_only mode via
# this env var (read by torch.load internally) avoids unpickling arbitrary
# objects from a third-party checkpoint downloaded off Google Drive, without
# needing to fork/patch their code. Must be set before load_pt() is called.
os.environ.setdefault("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vendor", "djtransgan"))

import torch
import soundfile as sf
from djtransgan.utils import load_pt, load_audio, squeeze_dim, out_audio
from djtransgan.model import get_generator
from djtransgan.process import preprocess, postprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAYLIST = os.path.join(REPO_ROOT, "playlist")
OUT_DIR = os.path.join(os.path.dirname(__file__), "vendor", "djtransgan", "spike_out")
os.makedirs(OUT_DIR, exist_ok=True)

WEIGHTS = os.path.join(os.path.dirname(__file__), "vendor", "djtransgan", "pretrained", "djtransgan_minmax.pt")


def main():
    tracks = sorted(f for f in os.listdir(PLAYLIST) if f.lower().endswith((".mp3", ".wav")))
    if len(tracks) < 2:
        raise SystemExit(f"Need at least 2 tracks in {PLAYLIST}, found {len(tracks)}")
    prev_path = os.path.join(PLAYLIST, tracks[0])
    next_path = os.path.join(PLAYLIST, tracks[1])
    print(f"prev: {tracks[0]}")
    print(f"next: {tracks[1]}")

    generator = get_generator()
    generator.load_state_dict(load_pt(WEIGHTS))
    generator.eval()

    prev_audio = load_audio(prev_path)
    next_audio = load_audio(next_path)
    # Cue points: 20s from the end of prev, 20s from the start of next — same
    # 20s convention UnmixDB's own cue-region docs use, arbitrary but reasonable
    # for a first spike. Real cue points come from our own beat grid in Task 6.
    prev_cue = max(0, prev_audio.shape[-1] / 44100 - 20)
    next_cue = 20

    (pair_audio, timestamps), (pair_audio_for_g, cue_for_g) = preprocess(
        prev_audio, next_audio, prev_cue, next_cue
    )
    mix_audio, mix_out = generator.infer(*pair_audio_for_g, cue_region=cue_for_g)

    print("\n--- mix_out structure ---")
    for data_type, curves in mix_out.items():
        print(f"{data_type}:")
        for key, val in curves.items():
            if torch.is_tensor(val):
                arr = val.detach().cpu().numpy()
                print(f"  {key}: shape={arr.shape} dtype={arr.dtype} "
                      f"min={arr.min():.4f} max={arr.max():.4f} "
                      f"has_nan={bool((arr != arr).any())}")
            else:
                print(f"  {key}: {type(val)}")

    out_wav = os.path.join(OUT_DIR, "spike_mix.wav")
    out_audio(squeeze_dim(mix_audio).to(torch.float32), out_wav)
    data, sr = sf.read(out_wav)
    rms = (data ** 2).mean() ** 0.5
    print(f"\nOutput wav: {out_wav}, duration={len(data)/sr:.1f}s, rms={rms:.4f}")
    if rms < 1e-6:
        raise SystemExit("SPIKE FAILED: output audio is silent")
    print("\nSPIKE PASSED: automation curves extracted, output audio is non-silent.")


if __name__ == "__main__":
    main()
EOF
```

- [ ] **Step 5: Run the spike and record the result**

Run (with the venv active): `cd ml && source venv/Scripts/activate && python spike_transition_inference.py`

Expected: prints the `mix_out` shape/dtype/range breakdown for both `prev` and `next`, `band` and `fader` keys, ends with `SPIKE PASSED`.

- [ ] **Step 6: DECISION GATE**

If Step 5 raised an error specifically from `weights_only=True` rejecting the
checkpoint's contents (error mentioning `weights_only` or an unpickling class
allowlist): the pretrained `.pt` contains more than plain tensors. Inspect
what it's carrying (`torch.load(path, weights_only=False)` — acceptable as a
one-time manual inspection step, not something this plan's scripts do
routinely) before deciding whether to trust it; do not silently drop back to
`weights_only=False` in the committed scripts.

If Step 5 printed `SPIKE PASSED`: record the exact shapes printed (e.g. `fader: shape=(1, 1, N)`) in `ml/README.md` under a "Verified shapes" heading — Task 7's `distill_labels.py` is written generically (squeeze/flatten) but was validated against these shapes; if they differ meaningfully, Task 7 may need adjustment. Proceed to Task 2.

If Step 5 failed (dependency install failure, `load_state_dict` shape mismatch, silent/NaN output, or any unrecoverable error): **STOP. Do not proceed to Tasks 6, 7, or 8.** Report the specific failure to the user and ask whether to (a) debug further, or (b) fall back to imitation-learning the transition model too (same approach as the selection model, using the deterministic `_transitionPlan()` as the imitation target) — this was the user's non-preferred option during design, so it needs explicit re-confirmation, not a silent pivot. Tasks 2-5 (design-doc fix, export action, selection pipeline, browser integration groundwork) are independent of this gate and can still proceed while the decision is pending.

- [ ] **Step 7: Write `ml/README.md` and commit**

```markdown
cat > ml/README.md << 'EOF'
# ml/ — offline training pipelines for AI mixing models

Not shipped to the browser. See `docs/superpowers/specs/2026-08-06-ai-mixing-models-design.md`
for the full design.

## Setup

    bash setup_djtransgan.sh

Clones DJtransGAN into `vendor/djtransgan/` (gitignored), creates `venv/` with a
minimal dependency set (NOT DJtransGAN's own `requirements.txt` — see
`spike_transition_inference.py`'s docstring for why), downloads pretrained weights.

## Security note: loading the pretrained checkpoint

`djtransgan.utils.load_pt()` is an unguarded `torch.load()` — DJtransGAN's
pretrained `.pt` file comes from Google Drive, a third-party source. Every
script here sets `TORCH_FORCE_WEIGHTS_ONLY_LOAD=1` before calling it, which
restricts unpickling to tensors instead of arbitrary Python objects. Our own
checkpoints (`export/*.pt`, written and read within the same pipeline) use
`torch.load(..., weights_only=True)` explicitly for the same reason. The
`np.load(..., allow_pickle=True)` calls on `data/transition_curves.npz` are
safe despite `allow_pickle` — that file is written by `build_transition_dataset.py`
in this same pipeline, never from a third-party source; `allow_pickle` is
required there only because the per-pair automation curves are ragged
(variable-length) arrays, not because the source is untrusted.

## Verified shapes (Task 1 spike, <DATE>)

<paste the mix_out shape/dtype/range output from Step 5 here>
EOF

git add ml/.gitignore ml/setup_djtransgan.sh ml/spike_transition_inference.py ml/README.md
git commit -m "spike: verify DJtransGAN pretrained generator produces usable automation curves"
```

---

### Task 2: Fix design doc class name

**Files:**
- Modify: `docs/superpowers/specs/2026-08-06-ai-mixing-models-design.md`

The committed design doc's architecture diagram says `TempoEngine`; the actual exported class in `public/host/engine.js` is `MusicEngine` (`export class MusicEngine` — confirmed via `test/engine.test.js:2`, `import { MusicEngine } from '../public/host/engine.js'`). This doc is the source of truth other tasks/implementers reference — fix it before it propagates further.

- [ ] **Step 1: Fix the name**

Find and replace `TempoEngine` with `MusicEngine` in `docs/superpowers/specs/2026-08-06-ai-mixing-models-design.md` (appears once, in the `simulate_selection.mjs` bullet under "Components").

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-08-06-ai-mixing-models-design.md
git commit -m "docs: fix class name in AI mixing models design spec (MusicEngine, not TempoEngine)"
```

---

### Task 3: Export library analysis (engine method + devtools helper)

**Files:**
- Modify: `public/host/engine.js`
- Modify: `public/host/app.js`
- Test: `test/engine.test.js`

**Interfaces:**
- Produces: `MusicEngine.prototype.exportLibraryAnalysis()` — `async () => Array<{name: string, bpm: number, bpmFallback: boolean, energy: number, duration: number, beatGrid: number[]|null, beatGridBpm: number|null, structure: {segments: Array<{start:number,end:number,energy:number}>}|null}>`. Task 4's selection-dataset generator and Task 6's transition-dataset builder both consume this shape (Task 6 via the JSON file it's dumped to, not the live method). `bpmFallback` is included (unlike `peaks`/the file handle) because Task 7's `distill_labels.py` needs it to filter out tracks whose BPM detection failed — a track stuck at the hardcoded 120 BPM default is otherwise indistinguishable from one genuinely analyzed at 120, which would corrupt the `bpmDelta` training feature.

The engine already caches exactly this per-track data in IndexedDB (`TRACK_CACHE_STORE`, keyed `"${name}|${size}|${lastModified}"`, engine.js:372-396) — this task adds a way to read it back out as plain JSON, following the project's existing devtools-only convention (no UI button) for anything dev/pilot-facing that isn't a party-night feature.

- [ ] **Step 1: Write the failing test**

```javascript
// Add to test/engine.test.js
import { MusicEngine } from '../public/host/engine.js';

describe('exportLibraryAnalysis', () => {
  it('reads back cached per-track analysis as plain JSON, without peaks or the file handle', async () => {
    const engine = makeEngine();
    const cacheKey = 'track-one.mp3|12345|1700000000000';
    const entry = {
      bpm: 128,
      bpmFallback: false,
      energy: 0.6,
      duration: 210.5,
      beatGrid: [0.5, 1.0, 1.5],
      beatGridBpm: 128,
      structure: { segments: [{ start: 0, end: 30, energy: 0.4 }] },
      peaks: new Array(2000).fill(0.1),
    };
    await engine._idbSetForExportTest(cacheKey, entry);

    const result = await engine.exportLibraryAnalysis();

    expect(result).toEqual([
      {
        name: 'track-one.mp3',
        bpm: 128,
        bpmFallback: false,
        energy: 0.6,
        duration: 210.5,
        beatGrid: [0.5, 1.0, 1.5],
        beatGridBpm: 128,
        structure: { segments: [{ start: 0, end: 30, energy: 0.4 }] },
      },
    ]);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- engine.test.js -t "exportLibraryAnalysis"`
Expected: FAIL — `engine._idbSetForExportTest is not a function` (or `exportLibraryAnalysis is not a function`).

- [ ] **Step 3: Implement the export method**

In `engine.js`, add a module-level cursor-based "get all" helper near the existing `_idbGet`/`_idbSet` (around line 143), and a `_idbSetForExportTest` test seam, then the public method on `MusicEngine`:

```javascript
// Add after _idbSet (engine.js, near line 143-152)
async function _idbGetAll(store) {
  const db = await _idbOpen();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(store, 'readonly');
    const req = tx.objectStore(store).openCursor();
    const out = [];
    req.onsuccess = () => {
      const cursor = req.result;
      if (cursor) {
        out.push({ key: cursor.key, value: cursor.value });
        cursor.continue();
      } else {
        resolve(out);
      }
    };
    req.onerror = () => reject(req.error);
  });
}
```

```javascript
// Add as a MusicEngine method, near loadLibraryFromDirectory / _preAnalyze.
/**
 * Reads back every cached per-track analysis entry as plain JSON — no
 * FileSystemFileHandle (not serializable), no `peaks` (2000-entry array
 * per track, irrelevant to training, keeps the export small). Devtools-only
 * dev tool for the ml/ training pipeline (see docs/superpowers/specs/
 * 2026-08-06-ai-mixing-models-design.md) — no UI wiring, matches the
 * existing creativeFlags devtools-only convention.
 */
async exportLibraryAnalysis() {
  const entries = await _idbGetAll(TRACK_CACHE_STORE);
  return entries.map(({ key, value }) => {
    const name = key.slice(0, key.lastIndexOf('|', key.lastIndexOf('|') - 1));
    const { bpm, bpmFallback, energy, duration, beatGrid, beatGridBpm, structure } = value;
    return { name, bpm, bpmFallback, energy, duration, beatGrid, beatGridBpm, structure };
  });
}

/** Test seam only — real cache keys are written by _preAnalyze via _idbSet. */
async _idbSetForExportTest(cacheKey, entry) {
  await _idbSet(TRACK_CACHE_STORE, cacheKey, entry);
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test -- engine.test.js -t "exportLibraryAnalysis"`
Expected: PASS

- [ ] **Step 5: Add the devtools download helper to app.js**

```javascript
// Add near the bottom of app.js, after `connect();`
// Devtools-only dev tool (no UI button, matches creativeFlags convention):
// run `downloadLibraryAnalysis()` in the browser console after a library
// has loaded to export cached per-track analysis for the ml/ training
// pipeline (docs/superpowers/specs/2026-08-06-ai-mixing-models-design.md).
window.downloadLibraryAnalysis = async () => {
  const data = await engine.exportLibraryAnalysis();
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'library-analysis.json';
  a.click();
  URL.revokeObjectURL(url);
  return `Exported ${data.length} tracks.`;
};
```

- [ ] **Step 6: Commit**

```bash
git add public/host/engine.js public/host/app.js test/engine.test.js
git commit -m "feat: add devtools-only library analysis export for ml training pipeline"
```

---

### Task 4: Selection dataset generator

**Files:**
- Create: `test/generate-selection-dataset.gen.js`
- Create: `ml/data/.gitkeep`
- Modify: `public/host/engine.js` (export two constants — see Step 0)

**Interfaces:**
- Consumes: `MusicEngine` (real class, `public/host/engine.js`), specifically `_pickNextTrack()`, `_energyTarget()`, `_noveltyPenalty()`, `_recordPick()` (all already exist, unmodified — only their exported-constants surface grows), plus the newly-exported `BPM_PENALTY_WEIGHT`/`NOVELTY_WEIGHT`.
- Produces: `ml/data/selection_dataset.json` — `Array<{pickId: number, candidateEnergy: number, candidateBpm: number, currentEnergy: number, currentBpm: number, energyTarget: number, noveltyPenalty: number, rank: number}>`. Task 5's `train_selection.py` consumes this file directly. `pickId` (added during this task's review fix) is a monotonically increasing id shared by every candidate row scored for the same pick — since group sizes now vary (the candidate pool shrinks as each simulated set plays out, not a constant 59), Task 5 groups rows by `pickId` rather than assuming a fixed stride.

This is not a Vitest *test* in the assertion sense — it's a data-generation script that runs *through* Vitest because `engine.js`'s bare `https://esm.sh/...` import (engine.js:8) means only Vitest's existing mock (`test/setup.js`) can load the real class outside a browser. Naming it `.gen.js` (not `.test.js`) keeps it out of `npm test`'s normal run; it's invoked explicitly.

**Design correction, found during Task 4's review:** the original version of this task's `runOnce()` reset `engine._recentHistory = []` on every iteration and never called `_recordPick()`, so `_noveltyPenalty()`'s early-return-on-empty-history path fired for all 118,000 generated rows — `noveltyPenalty` was a dead, constant-zero column. At inference, real sets build up real history across consecutive picks, so a model trained on always-zero novelty would receive genuinely nonzero novelty values it never learned from — the same train/inference mismatch class as the energy/rising fix earlier in this plan. The fix: simulate actual sequential sets (a run of consecutive real picks, each one calling `_recordPick()` before the next), not one-off independent snapshots. This also removes the need to hardcode `0.01`/`0.3` score-weight literals that risked drifting from `engine.js`'s real `BPM_PENALTY_WEIGHT`/`NOVELTY_WEIGHT` constants (an Important finding from the same review) — those two constants are now exported from `engine.js` and imported here instead.

- [ ] **Step 0: Export the two score-weight constants from engine.js**

```javascript
// engine.js — change these two existing const declarations (lines ~11, ~52)
// to `export const`, no other change:
export const BPM_PENALTY_WEIGHT = 0.01; // per BPM of difference from the current track
// ... (all other consts between these two stay exactly as they are today) ...
export const NOVELTY_WEIGHT = 0.3; // comparable magnitude to the existing energy/BPM score terms
```

Run `npm test` after this one-line-per-constant change — 32/32 must still pass (exporting a previously-module-private const doesn't change behavior).

- [ ] **Step 1: Write the generator script**

```javascript
// test/generate-selection-dataset.gen.js
import { writeFileSync, mkdirSync } from 'node:fs';
import { MusicEngine, BPM_PENALTY_WEIGHT, NOVELTY_WEIGHT } from '../public/host/engine.js';

/** Deterministic small synthetic library spanning a spread of energy/BPM
 * combinations — enough variety for _pickNextTrack()'s scoring to produce
 * a non-trivial ranking, without needing a real analyzed library on disk. */
function makeSyntheticLibrary(n = 60) {
  const tracks = [];
  for (let i = 0; i < n; i++) {
    tracks.push({
      name: `track-${i}.mp3`,
      bpm: 100 + (i % 12) * 4, // 100..144
      energy: (i % 10) / 9, // 0..1
    });
  }
  return tracks;
}

/** One simulated "set": a run of consecutive picks that calls _recordPick()
 * for real after each one, so _noveltyPenalty() sees genuine, non-empty
 * history for picks after the first few — matching how a real party set
 * actually accumulates _recentHistory, unlike a one-off snapshot per row. */
function runSet(engine, rows, stepsPerSet, pickIdRef) {
  engine.played = new Set();
  engine._recentHistory = [];
  engine.setStartedAt = Date.now() - Math.floor(Math.random() * 40 * 60 * 1000);
  engine.current = engine.library[Math.floor(Math.random() * engine.library.length)];
  engine.played.add(engine.current.name);

  for (let step = 0; step < stepsPerSet; step++) {
    const target = engine._energyTarget();
    const candidates = engine.library.filter((t) => !engine.played.has(t.name));
    if (candidates.length === 0) break;

    const scored = candidates
      .map((track) => {
        const energyDelta = Math.abs(track.energy - target);
        const bpmDelta = Math.abs(track.bpm - engine.current.bpm);
        const novelty = engine._noveltyPenalty(track);
        return { track, novelty, score: -energyDelta - bpmDelta * BPM_PENALTY_WEIGHT - novelty * NOVELTY_WEIGHT };
      })
      .sort((a, b) => b.score - a.score);

    const pickId = pickIdRef.value++; // shared by every candidate row from this one scoring event — Task 5 groups on this, not a fixed stride
    scored.forEach(({ track, novelty }, rank) => {
      rows.push({
        pickId,
        candidateEnergy: track.energy,
        candidateBpm: track.bpm,
        currentEnergy: engine.current.energy,
        currentBpm: engine.current.bpm,
        energyTarget: target,
        noveltyPenalty: novelty,
        rank,
      });
    });

    const picked = scored[0].track; // argmax — matches the deterministic engine with creativeFlags.sampling off
    engine._recordPick(picked); // real history accumulation, the fix this round is about
    engine.played.add(picked.name);
    engine.current = picked;
  }
}

export function generate({ setsCount = 200, stepsPerSet = 10 } = {}) {
  const engine = new MusicEngine({});
  engine.library = makeSyntheticLibrary();
  const rows = [];
  const pickIdRef = { value: 0 };
  for (let s = 0; s < setsCount; s++) runSet(engine, rows, stepsPerSet, pickIdRef);

  mkdirSync('ml/data', { recursive: true });
  writeFileSync('ml/data/selection_dataset.json', JSON.stringify(rows));
  return rows.length;
}

// Run directly via `npx vitest run test/generate-selection-dataset.gen.js --no-coverage`
// (executed as a plain module, not a describe/it suite — see package.json script).
if (process.env.GENERATE_SELECTION_DATASET === '1') {
  const count = generate({});
  // eslint-disable-next-line no-console
  console.log(`Wrote ${count} rows to ml/data/selection_dataset.json`);
}
```

Row count is no longer a fixed `iterations × 59` formula, since each set now plays out `stepsPerSet` sequential picks against a shrinking candidate pool (`60 - step` candidates at step `step`, once `played` grows) rather than a constant 59 every time — expect roughly `200 × 10 × ~55` (~110,000) rows, not exactly 118,000; Step 3 below checks the real printed count instead of asserting an exact number.

- [ ] **Step 2: Add the npm script to invoke it**

In `package.json`, add to `"scripts"`:

```json
"gen:selection-dataset": "cross-env GENERATE_SELECTION_DATASET=1 vitest run test/generate-selection-dataset.gen.js"
```

This needs `cross-env` since the project has no existing cross-platform env-var-setting convention (Windows dev machine per the environment this plan targets). Install it:

Run: `npm install --save-dev cross-env`

- [ ] **Step 3: Run it and verify output**

Run: `npm run gen:selection-dataset`
Expected: prints `Wrote N rows to ml/data/selection_dataset.json` with N in the ballpark of 100,000-120,000 (see Step 1's note on why this isn't an exact fixed number anymore), and the file exists.

Verify shape and that novelty is no longer dead: `node -e "const d = require('./ml/data/selection_dataset.json'); console.log(d.length, d[0]); console.log('distinct noveltyPenalty values:', new Set(d.map(r => r.noveltyPenalty)).size)"`
Expected: an object with keys `candidateEnergy, candidateBpm, currentEnergy, currentBpm, energyTarget, noveltyPenalty, rank`, and the distinct-values count is well above 1 (confirms the Step 0/1 fix — a constant-zero column would print `1`).

- [ ] **Step 4: Commit**

`ml/data/.gitkeep` is exempted from `ml/.gitignore`'s `data/*` rule by its `!data/.gitkeep` line (Task 1) — create it before adding, since Step 3's run only wrote the gitignored `selection_dataset.json` into that directory, not this tracked placeholder:

```bash
touch ml/data/.gitkeep
git add test/generate-selection-dataset.gen.js public/host/engine.js package.json package-lock.json ml/data/.gitkeep
git commit -m "feat: generate imitation-learning dataset for track-selection model"
```

---

### Task 5: Train and export the selection model

**Files:**
- Create: `ml/train_selection.py`
- Create: `ml/lib/__init__.py`
- Create: `ml/lib/onnx_export.py`

**Interfaces:**
- Consumes: `ml/data/selection_dataset.json` (Task 4).
- Produces: `public/host/models/selection.onnx`, `public/host/models/selection-model.meta.json` (`{"inputOrder": [...], "mean": [...], "std": [...]}`). Task 9's browser integration consumes both by exact filename and the `inputOrder`/`mean`/`std` keys.

- [ ] **Step 1: Write the shared ONNX-export helper**

```python
# ml/lib/onnx_export.py
import json
import os

import torch


def export_model(model, sample_input, out_dir, name, input_order, mean, std):
    """Exports a trained torch model to ONNX plus a sidecar .meta.json with
    the exact feature order and normalization stats the browser must apply
    before inference — the model's ONNX graph has no memory of feature names."""
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, f"{name}.onnx")
    torch.onnx.export(
        model,
        sample_input,
        onnx_path,
        input_names=["features"],
        output_names=["output"],
        dynamic_axes={"features": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    meta_path = os.path.join(out_dir, f"{name}-model.meta.json")
    with open(meta_path, "w") as f:
        json.dump({"inputOrder": input_order, "mean": mean, "std": std}, f, indent=2)
    return onnx_path, meta_path
```

- [ ] **Step 2: Write the training script**

Ranking loss: every candidate row scored for the same pick shares a `pickId` (added to the dataset during Task 4's review fix — group sizes vary now, since the candidate pool shrinks as each simulated set plays out, so grouping is by this id, not a fixed stride). The model should score each group so its ordering matches the deterministic engine's `rank` column. Implemented as a pairwise margin loss within each group (sampled, not full O(n²), to keep this fast on CPU).

```python
# ml/train_selection.py
import json

import numpy as np
import torch
import torch.nn as nn

from lib.onnx_export import export_model

FEATURES = ["candidateEnergy", "candidateBpm", "currentEnergy", "currentBpm", "energyTarget", "noveltyPenalty"]


class SelectionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(len(FEATURES), 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_dataset(path):
    with open(path) as f:
        rows = json.load(f)
    x = np.array([[r[k] for k in FEATURES] for r in rows], dtype=np.float32)
    rank = np.array([r["rank"] for r in rows], dtype=np.float32)
    pick_id = np.array([r["pickId"] for r in rows], dtype=np.int64)
    return x, rank, pick_id


def build_groups(pick_id):
    """Returns a list of row-index arrays, one per distinct pickId, in
    first-occurrence order. Replaces fixed-GROUP_SIZE stride slicing now that
    group sizes vary (candidate pool shrinks within each simulated set)."""
    groups, order = {}, []
    for i, pid in enumerate(pick_id.tolist()):
        if pid not in groups:
            groups[pid] = []
            order.append(pid)
        groups[pid].append(i)
    return [np.array(groups[pid]) for pid in order]


def pairwise_margin_loss(scores, ranks, groups, n_pairs=4):
    """Within each group (rows sharing one pickId), sample n_pairs (better,
    worse) pairs by rank and penalize the model if it doesn't score the
    better one higher. `groups` here holds indices local to `scores`/`ranks`
    (already remapped for whichever subset — train or val — is passed in)."""
    loss = torch.tensor(0.0)
    count = 0
    for idx in groups:
        if len(idx) < 2:
            continue
        group_scores = scores[idx]
        group_ranks = ranks[idx]
        for _ in range(n_pairs):
            i, j = np.random.choice(len(idx), size=2, replace=False)
            if group_ranks[i] == group_ranks[j]:
                continue
            better, worse = (i, j) if group_ranks[i] < group_ranks[j] else (j, i)
            loss = loss + torch.clamp(1.0 - (group_scores[better] - group_scores[worse]), min=0)
            count += 1
    return loss / max(count, 1)


def split_train_val(groups, val_fraction=0.1, seed=0):
    """Splits by GROUP (pick), never by row — a group must never straddle
    train/val. Returns (train_row_idx, train_groups, val_row_idx, val_groups),
    where *_groups hold indices local to the *_row_idx-gathered subset."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    n_val = max(1, int(val_fraction * len(groups)))
    val_group_positions = set(order[:n_val].tolist())

    train_row_idx, val_row_idx = [], []
    train_groups, val_groups = [], []
    for gi, idx in enumerate(groups):
        if gi in val_group_positions:
            start = len(val_row_idx)
            val_row_idx.extend(idx.tolist())
            val_groups.append(np.arange(start, start + len(idx)))
        else:
            start = len(train_row_idx)
            train_row_idx.extend(idx.tolist())
            train_groups.append(np.arange(start, start + len(idx)))
    return train_row_idx, train_groups, val_row_idx, val_groups


def main():
    x, ranks, pick_id = load_dataset("data/selection_dataset.json")
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1.0
    x_norm = (x - mean) / std

    groups = build_groups(pick_id)
    train_row_idx, train_groups, val_row_idx, val_groups = split_train_val(groups)
    x_train, x_val = x_norm[train_row_idx], x_norm[val_row_idx]
    r_train, r_val = ranks[train_row_idx], ranks[val_row_idx]

    model = SelectionNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    x_train_t = torch.tensor(x_train)
    r_train_t = torch.tensor(r_train)
    x_val_t = torch.tensor(x_val)
    r_val_t = torch.tensor(r_val)

    best_val = float("inf")
    patience, patience_left = 10, 10
    for epoch in range(200):
        model.train()
        opt.zero_grad()
        scores = model(x_train_t)
        loss = pairwise_margin_loss(scores, r_train_t, train_groups)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = pairwise_margin_loss(model(x_val_t), r_val_t, val_groups).item()
        if val_loss < best_val:
            best_val = val_loss
            patience_left = patience
            torch.save(model.state_dict(), "export/selection_best.pt")  # state_dict is tensors only — safe with weights_only=True below
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stop at epoch {epoch}, best val loss {best_val:.4f}")
                break
        if epoch % 20 == 0:
            print(f"epoch {epoch}: train_loss={loss.item():.4f} val_loss={val_loss:.4f}")

    model.load_state_dict(torch.load("export/selection_best.pt", weights_only=True))  # our own checkpoint, tensors only
    model.eval()
    sample_input = torch.zeros(1, len(FEATURES))
    onnx_path, meta_path = export_model(
        model, sample_input, "../public/host/models", "selection",
        FEATURES, mean.tolist(), std.tolist(),
    )
    print(f"Exported {onnx_path}, {meta_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run it**

Run: `cd ml && mkdir -p export && source venv/Scripts/activate && python train_selection.py`
Expected: prints decreasing train/val loss over epochs, ends with `Exported ../public/host/models/selection.onnx, ../public/host/models/selection-model.meta.json`.

- [ ] **Step 4: Smoke-test the exported ONNX file**

```python
python -c "
import onnxruntime as ort
import numpy as np
sess = ort.InferenceSession('../public/host/models/selection.onnx')
out = sess.run(None, {'features': np.zeros((3, 6), dtype=np.float32)})
print('output shape:', out[0].shape)
assert out[0].shape == (3,), 'expected one score per row'
print('OK')
"
```

(Requires `onnxruntime` — add it to `ml/setup_djtransgan.sh`'s pip install list, or install directly: `pip install onnxruntime`.)
Expected: `output shape: (3,)`, `OK`.

- [ ] **Step 5: Commit**

```bash
git add ml/train_selection.py ml/lib/ public/host/models/selection.onnx public/host/models/selection-model.meta.json
git commit -m "feat: train and export track-selection model (imitates deterministic engine)"
```

---

### Task 6: Transition dataset builder

**Files:**
- Create: `ml/build_transition_dataset.py`

**Interfaces:**
- Consumes: `playlist/*.mp3` (real audio, already in the repo), DJtransGAN's pretrained generator (Task 1), `ml/data/library-analysis.json` (Task 3's `window.downloadLibraryAnalysis()` export — a manual browser step, same as what Task 7 originally required; moved earlier to this task, see Step 0 below).
- Produces: `ml/data/transition_curves.npz` — arrays `pair_names`, `energy` (float, per-sample), `rising` (float 0/1, per-sample), `bpm_delta` (float, per-sample), `fader_curves` (object array of variable-length 1D float arrays), `band_curves` (same). Task 7's `distill_labels.py` consumes this file directly — it only reduces the curves; it does not recompute `energy`/`rising`/`bpmDelta`.

**Design correction, found during Task 3's review:** the original version of this task derived `energy`/`rising` at a single fixed cue point (`track duration - 20s`) per pair, and Task 7 separately re-derived them from whole-track average energy. Both of those measure something the browser's `_transitionPlan()` (engine.js:774) never actually computes at inference time — it reads the *structural segment containing the current playback position* (`seg.energy`, and `rising = next ? next.energy > seg.energy : false`), or, when a track has no real structure data, a synthetic sine curve unrelated to any audio measurement. Training a model on track-average energy or an arbitrary fixed cue point, then feeding it real segment-level energy at inference, is a train/inference mismatch that would make the model's behavior arbitrary. The fix: this task now samples cue points *at real structural segment boundaries* and computes `energy`/`rising` with the exact same segment-lookup rule `_transitionPlan()` uses — and Task 9's AI transition path is gated to only run when real structure data exists (never on the synthetic-curve fallback), so training data never needs to represent that regime at all. `hasRealStructure` is therefore dropped as a model input entirely (it was Task 8's 4th feature in an earlier version of this plan) — it's a gate on whether the AI path runs, not something the model needs to be told.

Only pairs where `prev` has at least one real structural segment, and neither `prev` nor `next` has `bpmFallback: true` (a fallback-120-BPM track would corrupt the `bpmDelta` feature — this is why Task 3 was amended to include `bpmFallback` in its export), are usable. Sampling at real segment boundaries also naturally covers a spread of energy/rising/bpmDelta combinations, replacing the earlier "shuffle track permutations" diversity strategy.

- [x] **Step 0: Get `ml/data/library-analysis.json`**

This requires a real browser session (uses IndexedDB, unavailable under Python):
1. Run the app (`npm start` from the repo root, open `public/host/index.html` per the project's existing dev flow), load a music library folder that includes the same tracks as `playlist/`.
2. Open devtools console, run `await downloadLibraryAnalysis()`.
3. Move the downloaded `library-analysis.json` to `ml/data/library-analysis.json`.

- [x] **Step 1: Write the dataset builder**

**Task 1's spike found that `djtransgan.process.preprocess()` — this task's
originally planned entry point — is unusable on Python 3.12**: it
unconditionally imports `djtransgan/process/beat.py`, which imports `madmom`,
which hard-fails on Python 3.12 (`collections.MutableSequence` and `np.float`
were both removed, independent of any dependency pinning). The verified
working replacement, confirmed in `ml/spike_transition_inference.py`
(committed in Task 1), is to call `djtransgan.dataset.select_audio_region`
directly — this task's script follows that same pattern instead of the
original `preprocess()`-based one.

```python
# ml/build_transition_dataset.py
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vendor", "djtransgan"))

import numpy as np
import torch
from djtransgan.config import settings
from djtransgan.utils import load_audio, normalize
from djtransgan.dataset import select_audio_region
from djtransgan.model import get_generator

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAYLIST = os.path.join(REPO_ROOT, "playlist")
WEIGHTS = os.path.join(os.path.dirname(__file__), "vendor", "djtransgan", "pretrained", "djtransgan_minmax.pt")
MAX_SAMPLES = 150  # keeps this runnable in minutes on CPU; revisit if the model underfits
MAX_SEGMENTS_PER_PAIR = 3  # caps how many of one pair's segment boundaries get sampled, for pair diversity


def load_library(path):
    with open(path) as f:
        tracks = json.load(f)
    return {t["name"]: t for t in tracks}


def usable_tracks(lib):
    """Tracks with real structural segments and trustworthy BPM — the same
    two conditions Task 9's AI transition path requires at inference time."""
    return {
        name: t for name, t in lib.items()
        if not t.get("bpmFallback") and t.get("structure") and len(t["structure"].get("segments", [])) > 0
    }


def segment_energy_rising(segments, idx):
    """Exactly engine.js's _transitionPlan() resolution (engine.js:774-786):
    the segment at idx, and whether the next segment (if any) is higher energy."""
    seg = segments[idx]
    nxt = segments[idx + 1] if idx + 1 < len(segments) else None
    return seg["energy"], (nxt["energy"] > seg["energy"]) if nxt else False


def build_pair_inputs(prev_path, next_path, prev_cue_point):
    """Same construction as ml/spike_transition_inference.py's verified working
    path: select_audio_region() instead of djtransgan.process.preprocess()
    (which requires madmom, broken on Python 3.12 — see Task 1's report).
    prev_cue_point is now a real structural segment boundary (Step 0's data),
    not an arbitrary fixed offset."""
    prev_audio = normalize(load_audio(prev_path))
    next_audio = normalize(load_audio(next_path))

    next_cue_point = 20  # next-side entry point is not feature-relevant, arbitrary is fine
    prev_cues = [max(0, prev_cue_point - 16), prev_cue_point]
    next_cues = [max(0, next_cue_point - 16), next_cue_point]

    prev_audio_for_g, _, _ = select_audio_region(prev_audio, prev_cues, settings.N_TIME, True, 0)
    next_audio_for_g, next_cues_for_g, _ = select_audio_region(next_audio, next_cues, settings.N_TIME, True, 1)

    pair_audio_for_g = [prev_audio_for_g.unsqueeze(0), next_audio_for_g.unsqueeze(0).to(torch.float32)]
    cue_for_g = next_cues_for_g.unsqueeze(0).to(torch.float32)
    return pair_audio_for_g, cue_for_g


def main():
    lib_path = "data/library-analysis.json"
    if not os.path.exists(lib_path):
        raise SystemExit(f"{lib_path} not found — see this task's Step 0.")
    lib = usable_tracks(load_library(lib_path))
    if len(lib) < 2:
        raise SystemExit(f"Only {len(lib)} tracks have real structure + trustworthy BPM — need at least 2.")

    tracks_on_disk = {f for f in os.listdir(PLAYLIST) if f.lower().endswith((".mp3", ".wav"))}
    names = [n for n in lib if n in tracks_on_disk]

    # Build the full sample plan (pair, segment index) up front so shuffling
    # gives pair AND segment-position diversity, not just pair diversity.
    rng = random.Random(42)
    plan = []
    for prev_name in names:
        segments = lib[prev_name]["structure"]["segments"]
        seg_indices = list(range(len(segments)))
        rng.shuffle(seg_indices)
        for idx in seg_indices[:MAX_SEGMENTS_PER_PAIR]:
            next_name = rng.choice([n for n in names if n != prev_name])
            plan.append((prev_name, next_name, idx))
    rng.shuffle(plan)
    plan = plan[:MAX_SAMPLES]

    generator = get_generator()
    # load_pt() (djtransgan's own wrapper) is a bare torch.load() with no
    # weights_only guard — call torch.load directly instead, same mitigation
    # Task 1 used, since this checkpoint is a third-party download.
    state_dict = torch.load(WEIGHTS, weights_only=True, map_location="cpu")
    generator.load_state_dict(state_dict)
    generator.eval()

    pair_names, energies, risings, bpm_deltas, fader_curves, band_curves = [], [], [], [], [], []
    for i, (prev_name, next_name, seg_idx) in enumerate(plan):
        prev_path = os.path.join(PLAYLIST, prev_name)
        next_path = os.path.join(PLAYLIST, next_name)
        try:
            segments = lib[prev_name]["structure"]["segments"]
            energy, rising = segment_energy_rising(segments, seg_idx)
            bpm_delta = abs(lib[prev_name]["bpm"] - lib[next_name]["bpm"])
            prev_cue_point = segments[seg_idx]["start"]

            pair_audio_for_g, cue_for_g = build_pair_inputs(prev_path, next_path, prev_cue_point)
            _, mix_out = generator.infer(*pair_audio_for_g, cue_region=cue_for_g)

            fader = mix_out["prev"]["fader"].detach().cpu().numpy().reshape(-1)
            band = mix_out["prev"]["band"].detach().cpu().numpy().reshape(-1)

            pair_names.append(f"{prev_name}__{next_name}__seg{seg_idx}")
            energies.append(energy)
            risings.append(1.0 if rising else 0.0)
            bpm_deltas.append(bpm_delta)
            fader_curves.append(fader)
            band_curves.append(band)
            print(f"[{i+1}/{len(plan)}] {prev_name}[seg{seg_idx}] -> {next_name}: "
                  f"energy={energy:.3f} rising={rising} bpmDelta={bpm_delta:.1f} fader={fader.shape} band={band.shape}")
        except Exception as e:
            print(f"[{i+1}/{len(plan)}] SKIPPED {prev_name}[seg{seg_idx}] -> {next_name}: {e}")

    os.makedirs("data", exist_ok=True)
    np.savez(
        "data/transition_curves.npz",
        pair_names=np.array(pair_names, dtype=object),
        energy=np.array(energies, dtype=np.float32),
        rising=np.array(risings, dtype=np.float32),
        bpm_delta=np.array(bpm_deltas, dtype=np.float32),
        fader_curves=np.array(fader_curves, dtype=object),
        band_curves=np.array(band_curves, dtype=object),
    )
    print(f"Wrote {len(pair_names)} samples to data/transition_curves.npz")


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Run it**

Run: `cd ml && source venv/Scripts/activate && python build_transition_dataset.py`
Expected: prints one line per sample (up to 150), each showing real `energy`/`rising`/`bpmDelta` values, ends with `Wrote N samples to data/transition_curves.npz`. Curve shapes should match Task 1's recorded `mix_out` shapes in `ml/README.md`.

- [ ] **Step 3: Commit** _(pending — awaiting explicit commit approval per repo policy)_

```bash
git add ml/build_transition_dataset.py
git commit -m "feat: build transition automation-curve dataset from DJtransGAN on our own library"
```

(Note: `ml/data/transition_curves.npz` and `ml/data/library-analysis.json` are both gitignored — regenerable/large, not source.)

---

### Task 7: Distill transition labels

**Files:**
- Create: `ml/distill_labels.py`

**Interfaces:**
- Consumes: `ml/data/transition_curves.npz` (Task 6) — `energy`/`rising`/`bpm_delta` arrays are already computed there (matching `_transitionPlan()`'s exact segment-lookup logic), this task does not recompute them or touch `library-analysis.json`.
- Produces: `ml/data/transition_dataset.json` — `Array<{energy: number, rising: number, bpmDelta: number, transitionMs: number, duckDb: number, fxIntensity: number}>`. Task 8's `train_transition.py` consumes this file directly. Three inputs, not four — `hasRealStructure` was dropped as a model feature during Task 3's review (see Task 6's "Design correction" note): it's a gate on whether Task 9's AI transition path runs at all, not something the model needs as an input, since training data only ever contains the real-structure regime.

This is the one lossy translation step named in the design doc: DJtransGAN's continuous per-timestep automation curves get reduced to the three scalars `_transitionPlan()` already predicts.

- [x] **Step 1: Write the distillation script**

```python
# ml/distill_labels.py
import json
import os

import numpy as np

EQ_MIN_DB = -18  # matches engine.js's EQ_MIN_DB constant — same clamp range on both sides


def reduce_curve(fader, band):
    """Reduces one sample's raw automation curves to (transitionMs, duckDb,
    fxIntensity) — the three scalars engine.js's _transitionPlan() already
    predicts. Assumes fader/band are 1D float arrays over the transition's
    time axis, normalized 0..1 by DJtransGAN's sigmoid output layer
    (generator.py sets last_activate='sigmoid')."""
    # Duration: span where the fader is actively moving (not pinned at 0 or 1),
    # as a fraction of the curve length, scaled to a plausible transition
    # duration range (15s-90s, matching engine.js's PEAK_BLEND_SEC..VALLEY_BLEND_SEC).
    moving = np.abs(np.diff(fader)) > 0.01
    active_fraction = moving.mean() if len(moving) > 0 else 0.5
    transition_ms = float(np.clip(15 + active_fraction * 75, 15, 90) * 1000)

    # Duck depth: how far the band curve dips below its own start, scaled to
    # engine.js's EQ_MIN_DB..0 range (band curve is 0..1, sigmoid output).
    band_dip = float(np.clip(band.max() - band.min(), 0, 1))
    duck_db = -band_dip * abs(EQ_MIN_DB)

    # FX intensity: normalized variance of the combined curve — a mostly-flat
    # curve reads as a plain linear fade (low intensity); a highly modulated
    # one reads as an intentional, expressive move (high intensity).
    combined = np.concatenate([fader, band])
    fx_intensity = float(np.clip(combined.std() * 4, 0, 1))

    return transition_ms, duck_db, fx_intensity


def main():
    # allow_pickle=True is required here because fader_curves/band_curves are
    # variable-length per sample (ragged, stored as numpy object arrays) — but
    # this file is written by build_transition_dataset.py in this same
    # pipeline (Task 6), never from a third-party or user-supplied source, so
    # the arbitrary-code-execution risk pickle normally carries doesn't apply.
    curves = np.load("data/transition_curves.npz", allow_pickle=True)

    rows = []
    for energy, rising, bpm_delta, fader, band in zip(
        curves["energy"], curves["rising"], curves["bpm_delta"], curves["fader_curves"], curves["band_curves"]
    ):
        transition_ms, duck_db, fx_intensity = reduce_curve(np.asarray(fader), np.asarray(band))
        rows.append({
            "energy": float(energy),
            "rising": float(rising),
            "bpmDelta": float(bpm_delta),
            "transitionMs": transition_ms,
            "duckDb": duck_db,
            "fxIntensity": fx_intensity,
        })

    os.makedirs("data", exist_ok=True)
    with open("data/transition_dataset.json", "w") as f:
        json.dump(rows, f)
    print(f"Wrote {len(rows)} rows to data/transition_dataset.json")


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Run the distillation script**

Run: `cd ml && source venv/Scripts/activate && python distill_labels.py`
Expected: `Wrote N rows to data/transition_dataset.json`, N equal to however many samples Task 6 produced (no further filtering happens here — Task 6 already filtered to usable tracks).

- [ ] **Step 3: Commit** _(pending — awaiting explicit commit approval per repo policy)_

```bash
git add ml/distill_labels.py
git commit -m "feat: distill DJtransGAN automation curves into engine.js's 3-parameter action space"
```

---

### Task 8: Train and export the transition model

**Files:**
- Create: `ml/train_transition.py`

**Interfaces:**
- Consumes: `ml/data/transition_dataset.json` (Task 7).
- Produces: `public/host/models/transition.onnx`, `public/host/models/transition-model.meta.json`. Task 9 consumes both.

- [ ] **Step 1: Write the training script**

Plain regression (not ranking, unlike selection) — each row has real target values, not just a relative order.

```python
# ml/train_transition.py
import json

import numpy as np
import torch
import torch.nn as nn

from lib.onnx_export import export_model

FEATURES = ["energy", "rising", "bpmDelta"]  # hasRealStructure dropped — it's Task 9's gate on whether the AI path runs, not a model input (see Task 6's "Design correction")
TARGETS = ["transitionMs", "duckDb", "fxIntensity"]


class TransitionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(len(FEATURES), 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, len(TARGETS)),
        )

    def forward(self, x):
        return self.net(x)


def main():
    with open("data/transition_dataset.json") as f:
        rows = json.load(f)
    if len(rows) < 20:
        raise SystemExit(f"Only {len(rows)} rows — too few to train on. Check Task 6/7 output.")

    x = np.array([[r[k] for k in FEATURES] for r in rows], dtype=np.float32)
    y = np.array([[r[k] for k in TARGETS] for r in rows], dtype=np.float32)

    x_mean, x_std = x.mean(axis=0), x.std(axis=0)
    x_std[x_std == 0] = 1.0
    y_mean, y_std = y.mean(axis=0), y.std(axis=0)
    y_std[y_std == 0] = 1.0

    x_norm = (x - x_mean) / x_std
    y_norm = (y - y_mean) / y_std

    n_val = max(1, int(0.15 * len(x)))
    idx = np.random.default_rng(0).permutation(len(x))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    x_train = torch.tensor(x_norm[train_idx])
    y_train = torch.tensor(y_norm[train_idx])
    x_val = torch.tensor(x_norm[val_idx])
    y_val = torch.tensor(y_norm[val_idx])

    model = TransitionNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    patience, patience_left = 15, 15
    for epoch in range(300):
        model.train()
        opt.zero_grad()
        pred = model(x_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_val), y_val).item()
        if val_loss < best_val:
            best_val = val_loss
            patience_left = patience
            torch.save(model.state_dict(), "export/transition_best.pt")  # state_dict is tensors only — safe with weights_only=True below
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stop at epoch {epoch}, best val loss {best_val:.4f}")
                break
        if epoch % 30 == 0:
            print(f"epoch {epoch}: train_loss={loss.item():.4f} val_loss={val_loss:.4f}")

    model.load_state_dict(torch.load("export/transition_best.pt", weights_only=True))  # our own checkpoint, tensors only
    model.eval()

    # Bake de-normalization of the OUTPUT into the exported graph too, so the
    # browser only has to apply input normalization (meta.json), not invert
    # output normalization by hand — a wrapper module composes cleanly here
    # since y_mean/y_std are fixed constants at export time.
    class DenormalizedModel(nn.Module):
        def __init__(self, inner, y_mean, y_std):
            super().__init__()
            self.inner = inner
            self.register_buffer("y_mean", torch.tensor(y_mean, dtype=torch.float32))
            self.register_buffer("y_std", torch.tensor(y_std, dtype=torch.float32))

        def forward(self, x):
            return self.inner(x) * self.y_std + self.y_mean

    export_ready = DenormalizedModel(model, y_mean, y_std)
    export_ready.eval()
    sample_input = torch.zeros(1, len(FEATURES))
    onnx_path, meta_path = export_model(
        export_ready, sample_input, "../public/host/models", "transition",
        FEATURES, x_mean.tolist(), x_std.tolist(),
    )
    print(f"Exported {onnx_path}, {meta_path}")
    print(f"Output order: {TARGETS}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `cd ml && source venv/Scripts/activate && python train_transition.py`
Expected: decreasing loss, ends with `Exported ../public/host/models/transition.onnx, ...` and `Output order: ['transitionMs', 'duckDb', 'fxIntensity']`.

- [ ] **Step 3: Smoke-test the exported ONNX file**

```python
python -c "
import onnxruntime as ort
import numpy as np
sess = ort.InferenceSession('../public/host/models/transition.onnx')
out = sess.run(None, {'features': np.array([[0.5, 1.0, 4.0]], dtype=np.float32)})
print('output:', out[0])
assert out[0].shape == (1, 3)
print('OK')
"
```
Expected: `output: [[<some transitionMs>, <some duckDb>, <some fxIntensity>]]`, `OK`.

- [ ] **Step 4: Commit**

```bash
git add ml/train_transition.py public/host/models/transition.onnx public/host/models/transition-model.meta.json
git commit -m "feat: train and export transition-parameter model (distilled from DJtransGAN)"
```

---

### Task 9: Browser integration — model loading and inference branching

**Files:**
- Modify: `public/host/engine.js`

**Interfaces:**
- Consumes: `public/host/models/{selection,transition}.onnx` + `.meta.json` (Tasks 5, 8).
- Produces: `MusicEngine.prototype._loadAiModels()`, `_scoreCandidateAi(candidate, current, target)`, `_transitionPlanAi(features)` — internal, but Task 10's tests call them directly via the engine instance.

- [x] **Step 1: Add the onnxruntime-web import and wasm path config**

At the top of `engine.js`, alongside the existing `web-audio-beat-detector` import:

```javascript
// engine.js, near line 8
import * as ort from 'https://esm.sh/onnxruntime-web@1.19.2';
// onnxruntime-web fetches its .wasm binaries relative to wherever it thinks
// its own script lives, which is unreliable through esm.sh's CDN rewriting —
// pin explicitly to the same CDN path esm.sh serves the wasm assets from.
ort.env.wasm.wasmPaths = 'https://esm.sh/onnxruntime-web@1.19.2/dist/';
```

- [x] **Step 2: Add the new creativeFlags entries**

```javascript
// engine.js:191 — extend the existing object, don't replace it
this.creativeFlags = {
  sampling: true,
  novelty: true,
  transitionVariety: true,
  fx: true,
  loopRoll: false,
  stems: false,
  // AI mixing models — /autoplan 2026-08-06 AI mixing models plan. Both
  // default off: deterministic behavior is unchanged until a host flips
  // one on from devtools (same kill-switch pattern as every flag above).
  // Falls back to the deterministic path for the rest of the session if
  // the model fails to load or a single inference call throws.
  aiSelection: false,
  aiTransition: false,
};
this._aiModels = { selection: null, transition: null }; // lazy-loaded, see _loadAiModels()
```

- [x] **Step 3: Add model loading**

```javascript
// New method on MusicEngine, near _preAnalyze
/**
 * Lazily loads an ONNX model + its normalization metadata. Any failure
 * (missing file, unsupported wasm, corrupt file) is caught here, logged
 * once, and the corresponding creativeFlags.ai* is turned off for the rest
 * of the session — a live party never sees a thrown error from this.
 */
async _loadAiModel(name) {
  if (this._aiModels[name]) return this._aiModels[name];
  try {
    const [session, meta] = await Promise.all([
      ort.InferenceSession.create(`models/${name}.onnx`),
      fetch(`models/${name}-model.meta.json`).then((r) => r.json()),
    ]);
    this._aiModels[name] = { session, meta };
    return this._aiModels[name];
  } catch (err) {
    console.warn(`[ai-mixing] failed to load ${name} model, falling back to deterministic:`, err);
    this.creativeFlags[name === 'selection' ? 'aiSelection' : 'aiTransition'] = false;
    return null;
  }
}

/** Applies training-time (x - mean) / std normalization, in the exact
 * inputOrder the model's meta.json declares — feature dicts are unordered
 * in JS, the model's ONNX graph is not. */
function _normalizeFeatures(features, meta) {
  return meta.inputOrder.map((key, i) => (features[key] - meta.mean[i]) / meta.std[i]);
}

async function _runInference(modelEntry, featureArray) {
  const { session } = modelEntry;
  const tensor = new ort.Tensor('float32', Float32Array.from(featureArray), [1, featureArray.length]);
  const results = await session.run({ features: tensor });
  return Array.from(results.output.data);
}
```

- [x] **Step 4: Branch `_pickNextTrack()` on `creativeFlags.aiSelection`**

Modify the existing scoring block (engine.js:841-849) — the deterministic path is unchanged, a new AI path runs before it when the flag is on:

```javascript
// Replace the block starting "const target = this._energyTarget();" through
// "scored.sort((a, b) => b.score - a.score);" (engine.js:841-849) with:
const target = this._energyTarget();
let scored;
if (this.creativeFlags.aiSelection) {
  const model = await this._loadAiModel('selection');
  if (model) {
    scored = await Promise.all(
      candidates.map(async (track) => {
        const bpmDelta = this.current ? Math.abs(track.bpm - this.current.bpm) : 0;
        const features = {
          candidateEnergy: track.energy,
          candidateBpm: track.bpm,
          currentEnergy: this.current ? this.current.energy : track.energy,
          currentBpm: this.current ? this.current.bpm : track.bpm,
          energyTarget: target,
          noveltyPenalty: this.creativeFlags.novelty ? this._noveltyPenalty(track) : 0,
        };
        try {
          const normalized = _normalizeFeatures(features, model.meta);
          const [score] = await _runInference(model, normalized);
          return { track, score };
        } catch (err) {
          console.warn('[ai-mixing] selection inference failed for a candidate, using deterministic fallback score:', err);
          return { track, score: -Math.abs(track.energy - target) - bpmDelta * BPM_PENALTY_WEIGHT };
        }
      })
    );
  }
}
if (!scored) {
  scored = candidates.map((track) => {
    const energyDelta = Math.abs(track.energy - target);
    const bpmDelta = this.current ? Math.abs(track.bpm - this.current.bpm) : 0;
    let score = -energyDelta - bpmDelta * BPM_PENALTY_WEIGHT;
    if (this.creativeFlags.novelty) score -= this._noveltyPenalty(track) * NOVELTY_WEIGHT;
    return { track, score };
  });
}
scored.sort((a, b) => b.score - a.score);
```

Note: `_pickNextTrack()` is not currently `async` — it must become `async` for this to work (its only caller sites need `await`). This is a real signature change; grep the codebase for `_pickNextTrack(` call sites and add `await` at each one as part of this step.

- [x] **Step 5: Branch `_transitionPlan()` on `creativeFlags.aiTransition`, gated to the real-structure regime**

`_transitionPlan()` is also not currently `async` — same signature-change note applies. Modify it (engine.js:774-805).

**This gate is load-bearing, not a style choice.** Task 6/7's training data only contains samples where the outgoing track has real structural segments — the model was never shown the synthetic-sine-curve fallback regime (`_energyTrajectory()`'s `0.6 + 0.3*sin(phase)`, a clock function unrelated to any audio measurement), because there's no way to generate real DJ-transition ground truth for a made-up curve. So the AI path below only runs inside the `if (outgoingTrack && outgoingTrack.structure...)` branch — never in the `else` (synthetic trajectory) branch. `hasRealStructure` is therefore not part of the feature vector (3 inputs, matching Task 8's `FEATURES`), because when this code path runs, it's always `true` by construction of the gate itself:

```javascript
// Replace the body of _transitionPlan (engine.js:774-805) — the value/rising
// resolution block stays identical; only what happens after it changes.
async _transitionPlan(outgoingTrack, positionSec) {
  let value;
  let rising;
  let hasRealStructure = false;
  if (outgoingTrack && outgoingTrack.structure && outgoingTrack.structure.segments.length > 0) {
    hasRealStructure = true;
    const segs = outgoingTrack.structure.segments;
    const seg = segs.find((s) => positionSec >= s.start && positionSec < s.end) || segs[segs.length - 1];
    const idx = segs.indexOf(seg);
    const next = segs[idx + 1];
    value = seg.energy;
    rising = next ? next.energy > seg.energy : false;
  } else {
    ({ value, rising } = this._energyTrajectory());
  }

  // Gate, not a flag check: the AI model was only ever trained on samples
  // where the outgoing track had real structural segments (Task 6/7) — it
  // has no basis to predict anything for the synthetic-curve regime, so
  // this path is unreachable when hasRealStructure is false, regardless of
  // creativeFlags.aiTransition.
  if (this.creativeFlags.aiTransition && hasRealStructure) {
    const model = await this._loadAiModel('transition');
    if (model) {
      try {
        const bpmDelta = this.current ? Math.abs(this.current.bpm - (outgoingTrack ? outgoingTrack.bpm : this.current.bpm)) : 0;
        const features = { energy: value, rising: rising ? 1 : 0, bpmDelta };
        const normalized = _normalizeFeatures(features, model.meta);
        const [transitionMsRaw, duckDbRaw, fxIntensityRaw] = await _runInference(model, normalized);
        const transitionMs = Math.min(Math.max(transitionMsRaw, PEAK_BLEND_SEC * 1000), VALLEY_BLEND_SEC * 1000);
        const duckDb = Math.min(Math.max(duckDbRaw, EQ_MIN_DB), 0);
        const fxIntensity = Math.min(Math.max(fxIntensityRaw, 0), 1);
        const nearPeak = rising && value >= PEAK_ENERGY_THRESHOLD - NEAR_PEAK_MARGIN;
        return { transitionMs, duckDb, fxIntensity, nearPeak };
      } catch (err) {
        console.warn('[ai-mixing] transition inference failed, using deterministic fallback:', err);
      }
    }
  }

  let transitionMs = BASE_BLEND_SEC * 1000;
  let duckDb = EQ_MIN_DB;
  let fxIntensity = 0.8;
  if (value >= PEAK_ENERGY_THRESHOLD) {
    transitionMs = PEAK_BLEND_SEC * 1000;
    fxIntensity = 0.6;
  } else if (value <= VALLEY_ENERGY_THRESHOLD) {
    transitionMs = VALLEY_BLEND_SEC * 1000;
    duckDb = EQ_MIN_DB * 0.5;
    fxIntensity = 1;
  } else if (rising) {
    transitionMs = RISING_BLEND_SEC * 1000;
    fxIntensity = 0.85;
  }

  const nearPeak = rising && value >= PEAK_ENERGY_THRESHOLD - NEAR_PEAK_MARGIN;
  return { transitionMs, duckDb, fxIntensity, nearPeak };
}
```

Grep for `_transitionPlan(` call sites and add `await` at each one as part of this step.

- [x] **Step 6: Copy the wasm/onnx model files where the browser can fetch them**

`ort.InferenceSession.create('models/selection.onnx')` resolves relative to `index.html`'s location (`public/host/`), so `public/host/models/` (already populated by Tasks 5 and 8) is already the right path — no copy needed. Verify:

Run: `ls public/host/models/`
Expected: `selection.onnx`, `selection-model.meta.json`, `transition.onnx`, `transition-model.meta.json`.

- [ ] **Step 7: Commit** _(pending — awaiting explicit commit approval per repo policy)_

```bash
git add public/host/engine.js
git commit -m "feat: wire AI selection/transition models into engine.js behind creativeFlags kill switches"
```

---

### Task 10: Integration tests

**Files:**
- Modify: `test/setup.js`
- Test: `test/engine.aiModels.test.js`

**Interfaces:**
- Consumes: `MusicEngine` with `creativeFlags.aiSelection` / `.aiTransition`, `_loadAiModel`, `_pickNextTrack`, `_transitionPlan` (Task 9).

- [x] **Step 1: Mock onnxruntime-web in test/setup.js**

```javascript
// Add to test/setup.js, alongside the existing web-audio-beat-detector mock
vi.mock('https://esm.sh/onnxruntime-web@1.19.2', () => ({
  env: { wasm: {} },
  InferenceSession: {
    create: vi.fn(async () => {
      throw new Error('mock: no model file in test environment');
    }),
  },
  Tensor: class {
    constructor(type, data, dims) {
      this.type = type;
      this.data = data;
      this.dims = dims;
    }
  },
}));
```

This mock's `InferenceSession.create` always rejecting is deliberate — it's the load-failure path, exercised by Step 2 below. A separate per-test override (Step 3) covers the success path.

- [x] **Step 2: Write the failing test — load failure falls back to deterministic**

```javascript
// test/engine.aiModels.test.js
import { describe, it, expect, vi } from 'vitest';
import { MusicEngine } from '../public/host/engine.js';

function makeEngine() {
  return new MusicEngine({});
}

describe('AI model load failure fallback', () => {
  it('flips creativeFlags.aiSelection off and still returns a valid pick when the model fails to load', async () => {
    const engine = makeEngine();
    engine.library = [
      { name: 'a.mp3', bpm: 120, energy: 0.5 },
      { name: 'b.mp3', bpm: 122, energy: 0.6 },
    ];
    engine.creativeFlags.aiSelection = true;
    engine.creativeFlags.sampling = false; // deterministic argmax, easier to assert on

    const picked = await engine._pickNextTrack();

    expect(picked).toBeTruthy();
    expect(engine.creativeFlags.aiSelection).toBe(false);
  });

  it('flips creativeFlags.aiTransition off and still returns valid clamped values when the model fails to load', async () => {
    const engine = makeEngine();
    engine.creativeFlags.aiTransition = true;
    engine.setStartedAt = Date.now();

    // Gate requires real structure to reach the AI branch at all (see Task 9
    // Step 5) — without it, the flag would never get a chance to flip, since
    // the deterministic path is what's supposed to run anyway.
    const outgoingTrack = {
      bpm: 120,
      structure: { segments: [{ start: 0, end: 60, energy: 0.5 }] },
    };
    const plan = await engine._transitionPlan(outgoingTrack, 0);

    expect(engine.creativeFlags.aiTransition).toBe(false);
    expect(plan.transitionMs).toBeGreaterThan(0);
    expect(plan.duckDb).toBeLessThanOrEqual(0);
    expect(plan.fxIntensity).toBeGreaterThanOrEqual(0);
    expect(plan.fxIntensity).toBeLessThanOrEqual(1);
  });
});
```

- [x] **Step 3: Run to verify current behavior**

Run: `npm test -- engine.aiModels.test.js`
Expected: PASS if Task 9 was implemented correctly (load failure already flips the flag and falls through to the deterministic path within `_loadAiModel`/the `if (!scored)` guard). If it FAILS, the fallback wiring from Task 9 Steps 4-5 has a bug — fix there, not here.

- [x] **Step 4: Write the clamping test — success path with a pathological model output**

```javascript
// Add to test/engine.aiModels.test.js
describe('AI transition output clamping', () => {
  it('clamps a model output outside the deterministic safe range', async () => {
    const engine = makeEngine();
    engine.creativeFlags.aiTransition = true;
    engine.setStartedAt = Date.now();

    // Override _loadAiModel for this test only — simulates a successful load
    // whose model outputs a pathological value (e.g. an undertrained model
    // early in development), the real-world case clamping exists for.
    engine._loadAiModel = vi.fn(async () => ({
      session: {
        run: vi.fn(async () => ({
          output: { data: Float32Array.from([999999, -999999, 5]) }, // way outside safe ranges
        })),
      },
      meta: { inputOrder: ['energy', 'rising', 'bpmDelta'], mean: [0, 0, 0], std: [1, 1, 1] },
    }));

    // The AI path is gated on the outgoing track having real structural
    // segments (see Task 9 Step 5's "load-bearing gate" note) — a null
    // outgoingTrack would never reach the AI branch at all, so this test
    // needs a track with a real structure to actually exercise clamping.
    const outgoingTrack = {
      bpm: 120,
      structure: { segments: [{ start: 0, end: 60, energy: 0.5 }] },
    };
    const plan = await engine._transitionPlan(outgoingTrack, 0);

    expect(plan.transitionMs).toBeLessThanOrEqual(90 * 1000); // VALLEY_BLEND_SEC
    expect(plan.transitionMs).toBeGreaterThanOrEqual(20 * 1000); // PEAK_BLEND_SEC
    expect(plan.duckDb).toBeGreaterThanOrEqual(-18); // EQ_MIN_DB
    expect(plan.duckDb).toBeLessThanOrEqual(0);
    expect(plan.fxIntensity).toBeGreaterThanOrEqual(0);
    expect(plan.fxIntensity).toBeLessThanOrEqual(1);
  });
});
```

- [x] **Step 5: Run to verify it passes**

Run: `npm test -- engine.aiModels.test.js`
Expected: PASS.

- [x] **Step 6: Run the full suite to check for regressions from the `async` signature changes**

Run: `npm test`
Expected: all tests pass, including the pre-existing `test/engine.test.js` suite (its call sites for `_pickNextTrack`/`_transitionPlan` need `await` added if any test calls them directly — fix any resulting failures by adding `await`, not by reverting the signature change).

- [ ] **Step 7: Commit** _(pending — awaiting explicit commit approval per repo policy)_

```bash
git add test/setup.js test/engine.aiModels.test.js
git commit -m "test: cover AI model load-failure fallback and output clamping"
```

---

### Task 11: Final wiring check and rollout note

**Files:**
- Modify: `TODOS.md`

- [ ] **Step 1: Verify default-off behavior end to end**

Run: `npm test`
Expected: full suite passes with `creativeFlags.aiSelection`/`aiTransition` at their default `false` — confirms merge is a no-op for existing behavior.

- [ ] **Step 2: Add a TODOS.md entry for the listening-pass validation**

This mirrors every other "not heard live yet" entry already in the file (loop-roll, stems, structural signal) — the AI models need the same real-audio validation before their default should ever change.

```markdown
<!-- Append to TODOS.md, under "## Mixing Engine", before "## Completed" -->

### AI mixing models — listening-pass validation

**What:** `creativeFlags.aiSelection` and `creativeFlags.aiTransition` (engine.js)
ship default-off. Neither has been heard live — validate both by ear at a real
session before considering flipping either default, per
`docs/superpowers/plans/2026-08-06-ai-mixing-models.md`.

**Why:** Same bar as every other creative-layer flag in this codebase
(loop-roll, stems, structural signal) — a model that looks correct in a unit
test (clamped output, correct fallback) hasn't been validated for how it
actually sounds.

**Context:** The selection model is presently equivalent to the deterministic
rules it imitates (see the design spec's "Known limitation" section) — it
won't sound different from `aiSelection: false` until fine-tuned on real usage
data. The transition model should sound different immediately, since it's
distilled from DJtransGAN's real-DJ-trained automation curves.

**Effort:** S
**Priority:** P2
**Depends on:** A real party/pilot session with the mixer console in use.
```

- [ ] **Step 3: Commit**

```bash
git add TODOS.md
git commit -m "docs: track AI mixing models listening-pass validation in TODOS.md"
```

---

## Self-Review Notes

- **Spec coverage:** Data pipeline (Tasks 3, 4, 6, 7) ✓, model training (Tasks 5, 8) ✓, browser integration + fallback toggle (Task 9) ✓, testing (Task 10) ✓, the design's "known limitation" caveat ✓ (Task 11 TODOS entry references it directly rather than re-explaining it).
- **Placeholder scan:** no TBD/TODO; the one legitimately provisional detail (exact `mix_out` tensor shapes from DJtransGAN) is resolved by Task 1's spike recording real values into `ml/README.md` before Task 6/7 use them — not a placeholder, a documented dependency on an earlier task's empirical output.
- **Type/signature consistency:** `_pickNextTrack()` and `_transitionPlan()` both become `async` in Task 9 — Task 10's tests already `await` them; Step 6 of Task 10 explicitly checks for other call sites needing the same update, since `writing-plans` can't see every caller from this vantage point.
- **Decision gate:** Task 1 is the one task whose failure changes the shape of Tasks 6-8 (transition model would need to switch from distillation to imitation, matching Task 4-5's pattern instead). It's called out explicitly rather than assumed to succeed.

---

## /autoplan Review — 2026-08-11

**Scope note:** Tasks 1-5 are already committed (worktree `ai-mixing-models`, commits through `6d2d5da`). This review treats T1-5 as retrospective (verify what shipped matches the plan) and puts full depth on T6-11 (unbuilt). Codex CLI not installed on this machine — all dual-voice checks below are Claude-subagent-only (`[subagent-only]`), not full consensus. Design phase (Phase 2) skipped: plan states "No UI control for the new flags... devtools-only," confirmed by reading the plan — no UI surface to review. DX phase (Phase 3.5) triggered: `ml/` is a developer-facing offline pipeline (setup script, README, pip installs).

### Decision Audit Trail

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|-------|----------|-----------------|-----------|-----------|----------|
| D1 | CEO | Skip Design phase (Phase 2) | Mechanical | — | Plan explicitly states no UI surface for these flags | — |
| D2 | CEO | Trigger DX phase (Phase 3.5) | Mechanical | — | `ml/` has a setup script, README, npm script, pip installs — developer-facing | — |
| D3 | Eng | Task 9 Step 1's top-level static `import * as ort from 'https://esm.sh/onnxruntime-web@1.19.2'` must become a dynamic `import()` inside `_loadAiModel()` | Mechanical | P5 (explicit over clever) | A static import resolves at module-load time, before any `creativeFlags` check runs — if esm.sh is slow/blocked/unreachable, the whole engine fails to load, not just the opt-in AI feature. Violates the plan's own "never throws into a live set" constraint. Confirmed independently by the Eng subagent voice. | Leaving it static (rejected — defeats the flag's purpose as a kill switch) |
| D4 | Eng | Task 10's file list must explicitly add `test/engine.test.js` (7 un-awaited call sites of `_pickNextTrack`/`_transitionPlan` found in the worktree, lines 91/99/112/191/198/208/213) | Mechanical | P5 (explicit) | Buried in a Step 6 aside instead of declared as a file to modify; an executor following the file list literally misses it until CI fails | — |
| D5 | Eng | Task 9's `_pickNextTrack()` AI path should clamp/sanitize selection scores (NaN/Infinity), matching the clamping already applied to `_transitionPlan()`'s AI output | Mechanical | P1 (completeness) | `Array.sort` with a NaN comparator silently misorders instead of throwing — same failure class already handled on the transition side, left open here | — |
| D6 | Eng | Task 10 should add a test for `session.run()` throwing after a successful model load (not just load-failure and clamping) | Mechanical | P1 (completeness) | Distinct code path (inner try/catch around per-candidate/per-call inference) with no current coverage | — |
| D7 | CEO/Eng/DX | **Should Tasks 6-11 be built and merged before the party (3 days out)?** | **Taste — surfaced at gate, not auto-decided** | P6 (bias toward action) vs. P2 (boil lakes) in tension | All three independent subagent voices (CEO, Eng, DX), each reviewing blind with no shared context, converged on the same recommendation without prompting: don't build/merge T6-11 pre-party. CEO: the only model that could sound different (`aiTransition`) can't be validated by ear before a party that hasn't happened; the only one safe to ship (`aiSelection`) is a provable no-op. Eng: T9 adds a new runtime CDN dependency and async-signature changes to the two functions most load-bearing for "does the music keep playing." DX: the `ml/` pipeline is fragile (undocumented deps, Google Drive-hosted weights, a 2022 repo with dead sub-dependencies) — bit-rot risk, not party risk, but real. | — |

### Cross-Phase Themes

**Theme: this feature's risk is concentrated in the runtime path, not the training pipeline.** CEO and Eng independently flagged the same mechanism (static CDN import + async signature changes landing in `_pickNextTrack`/`_transitionPlan`) as the actual party-night risk, from different angles (business risk vs. code correctness). High-confidence signal precisely because neither subagent saw the other's reasoning.
