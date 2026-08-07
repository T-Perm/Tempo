# AI Mixing Models — Design

Status: approved (pending user spec review)
Branch: mvp-mixing-console

## Problem

The mixing engine (`public/host/engine.js`) makes two live decisions with hand-tuned
deterministic rules: which track plays next (`_pickNextTrack()`) and how to transition
into it (`_transitionPlan()`). Both are rule sets over a handful of features (energy,
BPM delta, novelty, structural segment). The goal is to replace each with a learned
model while keeping the deterministic path fully available as a fallback — not a
one-way migration.

## Non-goals

- No change to audio playback, EQ, FX, or loop-roll mechanics themselves — only the
  *parameters* fed into them (which track, transition duration/duck/FX intensity).
- No live/online learning during a party. Training happens offline; the browser only
  runs inference.
- No UI control for the new flags in this pass — every existing `creativeFlags` entry
  (`sampling`, `novelty`, `fx`, `loopRoll`, `stems`) is a devtools-only kill switch, and
  the new ones follow the same convention.
- No attempt to source a real "which track do real DJs pick next" preference dataset —
  investigated and none exists that isn't tied to specific copyrighted tracks we don't
  have (see Data Sources below).

## Data sources (investigated, with results)

- **UnmixDB** (zenodo.org/records/1422385): ruled out for both models. Its "mixes" are
  synthetically generated — `makemixdb-creation`'s `makemixes.py` produces every
  mechanical rotation ("all subsequences with wraparound") of a source track list, and
  every transition is a fixed 4-measure beat-aligned linear crossfade. Training on it
  would teach a model to reproduce one uniform fade shape — the deterministic-slop
  problem this project is trying to escape, hidden inside weights instead of a formula.
- **DJtransGAN** (github.com/ChenPaulYu/DJtransGAN, MIT license): the authors' real
  training data (Livetracklist DJ mixes) isn't redistributable due to licensing, but
  they publish a **pretrained generator** already trained on real DJs' EQ/fader
  automation curves. This is usable via distillation — run the pretrained model on
  pairs from our own library, treat its output as ground truth for our own small model.
  This is the source for the transition-parameter model.
- **Track-selection preference data**: no viable real-DJ dataset found. The selection
  model instead imitates the existing deterministic `_pickNextTrack()` — see the
  "Known limitation" note below.

## Feature definitions (precise, to avoid ambiguity with existing code)

`_transitionPlan()` (engine.js:774) already folds structural-segment data into its
`value`/`rising` pair when `_analyzeStructure` produced one, falling back to the
synthetic set-level curve otherwise — `value`/`rising` is a single shape-compatible
signal regardless of source, not two separate ones. The transition model's inputs are
three of those, not four (revised during implementation — see below):

- `energy` (`value`, 0-1) — from `_transitionPlan`'s existing value/rising resolution.
- `rising` (bool) — same resolution.
- `bpmDelta` — `|incoming.bpm - outgoing.bpm|`, same as the selection model's input.

**`hasRealStructure` is a gate, not a fourth input feature** (revised from this design's
original version during Task 3's implementation review). There is no way to generate
real DJ-transition ground truth for the synthetic sine-curve fallback regime — DJtransGAN
distills real automation behavior onto real structural segments, never onto a made-up
clock function. So the AI transition path only runs when `outgoingTrack.structure` has
real segments; when it doesn't, `_transitionPlan()` always uses the deterministic rules,
regardless of `creativeFlags.aiTransition`. Training data therefore only ever represents
the real-structure regime, and the model doesn't need a bit telling it so — it's implied
by the fact that it was invoked at all.

The selection model's inputs are the candidate track's `energy`/`bpm`, the current
track's `energy`/`bpm`, the energy target from `_energyTarget()`, and the same novelty
penalty term `_noveltyPenalty()` already computes — i.e. the exact inputs
`_pickNextTrack()`'s scoring function uses today, since this model is imitating it.

## Architecture

```
                     TRAINING TIME (offline, ml/, not shipped)
┌─────────────────────────────────────────────────────────────────────┐
│  app.js "Export analysis" dev action                                  │
│      → JSON dump of IndexedDB TRACK_CACHE_STORE                       │
│      (energy, bpm, beatGrid, structure — from engine.js's own          │
│       _estimateEnergy / _analyzeBeatGrid / _analyzeStructure)          │
│                          │                                            │
│         ┌────────────────┴────────────────┐                          │
│         ▼                                  ▼                          │
│  ml/build_transition_dataset.py    ml/simulate_selection.mjs          │
│  samples (trackA,trackB) pairs,    imports real MusicEngine class     │
│  runs DJtransGAN pretrained        from engine.js, runs               │
│  generator on real audio →         _pickNextTrack() thousands of      │
│  real EQ/fader automation curve    times over randomized set-starts   │
│         │                                  │                          │
│         ▼                                  ▼                          │
│  ml/distill_labels.py reduces      labeled (features → rank) rows,    │
│  curve to (transitionMs,           guaranteed no drift from the       │
│  duckDb, fxIntensity)              rules being imitated               │
│         │                                  │                          │
│         ▼                                  ▼                          │
│  ml/train_transition.py            ml/train_selection.py              │
│  small MLP, MSE loss               small MLP, ranking loss            │
│         │                                  │                          │
│         ▼                                  ▼                          │
│  transition.onnx +                 selection.onnx +                   │
│  transition-model.meta.json        selection-model.meta.json          │
│  (normalization stats)             (normalization stats)              │
└─────────────────────────┬───────────────────────────────────────────┘
                           │  checked into public/host/models/
                           ▼
                     RUNTIME (browser, live party)
┌─────────────────────────────────────────────────────────────────────┐
│  engine.js                                                            │
│  creativeFlags.aiSelection / .aiTransition  (default: false)          │
│                                                                        │
│  _pickNextTrack()          _transitionPlan()                          │
│    if aiSelection:           if aiTransition AND hasRealStructure:    │
│      onnxruntime-web           onnxruntime-web                        │
│      inference → rank          inference → (transitionMs,             │
│      candidates                 duckDb, fxIntensity)                  │
│                                 inputs: energy, rising, bpmDelta       │
│                                 (hasRealStructure is the gate above,   │
│                                 not a model input — see Feature defs)  │
│      else: existing            → clamp to same safe ranges            │
│      deterministic scoring      deterministic constants define        │
│                                 else: existing deterministic rules     │
│                                                                        │
│  Model load/inference failure → console warning once, fall back to    │
│  deterministic path for rest of session. Never throws into a live set.│
└─────────────────────────────────────────────────────────────────────┘
```

## Components

### `ml/` (new top-level directory, Python + one Node script, gitignored intermediates)

- `requirements.txt` — torch, numpy, librosa, soundfile, onnx.
- `build_transition_dataset.py` — samples track pairs at real structural-segment
  boundaries from the exported library JSON (not arbitrary cue points — matches what
  `_transitionPlan()` actually reads at inference), runs each through DJtransGAN's
  pretrained generator, saves raw automation curves alongside the `energy`/`rising`/
  `bpmDelta` computed at that exact segment.
- `distill_labels.py` — reduces each curve to `(transitionMs, duckDb, fxIntensity)`.
  Reduction rules: `transitionMs` = span from fade-in onset to fade-out completion in
  the fader curve; `duckDb` = peak low-band cut in the EQ curve; `fxIntensity` =
  normalized variance of the combined curve. This is the one lossy translation step
  from DJtransGAN's continuous automation space into our existing 3-parameter action
  space.
- `simulate_selection.mjs` — Node script, imports `MusicEngine` from
  `public/host/engine.js` directly and calls `_pickNextTrack()` in a loop with
  randomized set-start timestamps and RNG seeds, logging feature vectors + resulting
  rank.
- `train_transition.py` / `train_selection.py` — PyTorch training scripts. Small MLPs
  (2 hidden layers, ~32 units), dropout + early stopping on a held-out split. Export via
  `torch.onnx.export`.
- `export/` — output ONNX files + `.meta.json` normalization stats before copying into
  `public/host/models/`.

### `public/host/models/` (new, checked into git — small binary assets)

- `selection.onnx`, `selection-model.meta.json`
- `transition.onnx`, `transition-model.meta.json`

### `public/host/engine.js` changes

- New dependency: `onnxruntime-web` (wasm backend).
- New `creativeFlags` entries: `aiSelection: false`, `aiTransition: false`.
- Lazy model loading (once, cached on the engine instance) on first use of either flag.
- `_pickNextTrack()`: branch on `creativeFlags.aiSelection` — AI path assembles the
  trained feature vector (candidate features, current-track features, energy target),
  runs inference, sorts by predicted rank; existing deterministic scoring is otherwise
  untouched.
- `_transitionPlan()`: branch on `creativeFlags.aiTransition` AND `hasRealStructure`
  (the gate — see Feature Definitions section above) — AI path assembles
  `(energy, rising, bpmDelta)`, runs inference, clamps outputs to
  the existing deterministic constants' ranges (`PEAK_BLEND_SEC`..`VALLEY_BLEND_SEC` for
  duration, `EQ_MIN_DB`..`0` for duck, `0`..`1` for FX intensity); existing deterministic
  rules are otherwise untouched, and are the only path when structure data isn't real.
- New `_loadAiModels()` / inference helper functions, wrapped so any load or inference
  failure logs one console warning and disables the relevant flag for the rest of the
  session rather than throwing.

## Error handling

- Missing model files, unsupported wasm, or a throwing inference call: caught, logged
  once, `creativeFlags.ai*` flipped to `false` in memory for the rest of the session,
  deterministic path takes over silently. A live party never sees a crash from this.
- Model output values are clamped to the same safe ranges the deterministic constants
  already enforce, since a regression/ranking model's raw output is unbounded in
  principle and a live set can't tolerate a pathological value (e.g. a multi-minute
  transition).

## Testing

- Vitest unit tests (existing suite, `npm test`): feature-vector assembly correctness,
  clamping boundary behavior, and the load-failure fallback path (mock a rejected model
  load, assert the deterministic path still runs and the flag gets disabled).
- Model *quality* is not something a unit test can grade — like every other mixing
  feature in this codebase's history (per `TODOS.md`'s recurring "not heard live yet"
  notes), validating that the AI picks/transitions actually sound good is a listening
  pass, not part of this design's automated test coverage.
- Python/Node training-side scripts get light smoke tests only (exported ONNX loads and
  produces the expected output shape) — not full pipeline tests, since the pipeline's
  output quality depends on the pretrained DJtransGAN model and library contents, not
  logic this repo controls.

## Known limitation (explicit, not silently accepted)

The selection model, as scoped in this pass, is mathematically equivalent to
`_pickNextTrack()`'s existing rules — it's imitation-trained on the deterministic
engine's own output because no real-DJ track-pairing preference dataset was found that
isn't tied to specific copyrighted tracks. Flipping `creativeFlags.aiSelection` on today
will not sound different from leaving it off. Its value is structural: it becomes the
foundation TODOS.md's "read-the-room feedback" item (guest skip/override signal) can
fine-tune later without re-deriving the deterministic rules from scratch. The transition
model does not have this limitation — it's distilled from real DJ behavior via
DJtransGAN from day one, so `creativeFlags.aiTransition` should sound different
immediately. One caveat specific to the transition model: it only ever activates when
the outgoing track has real structural segments (`_analyzeStructure` succeeded, not the
synthetic fallback curve) — see Feature Definitions. On a track where structure detection
failed or was too short to produce segments, `creativeFlags.aiTransition` has no effect
and the deterministic rules run regardless of the flag.

## Rollout

Both flags default `false`. No behavior change on merge. Enabling either is a manual
devtools action (`engine.creativeFlags.aiTransition = true`), same as every other
`creativeFlags` entry today — validated by ear before ever considering a UI toggle or a
different default.
