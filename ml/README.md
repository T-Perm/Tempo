# ml/ — offline training pipelines for AI mixing models

Not shipped to the browser. See `docs/superpowers/specs/2026-08-06-ai-mixing-models-design.md`
for the full design.

## Setup

    bash setup_djtransgan.sh

Clones DJtransGAN into `vendor/djtransgan/` (gitignored), creates `venv/` with a
minimal dependency set (NOT DJtransGAN's own `requirements.txt` — see
`setup_djtransgan.sh`'s comments and `spike_transition_inference.py`'s
docstring for why), downloads pretrained weights.

Note: `ml/` expects `playlist/` at the repo root (used by the spike script,
gitignored, not tracked in git). In a worktree that doesn't have those audio
files checked out, create a directory junction/symlink pointing at the main
checkout's `playlist/` before running the spike, e.g. on Windows:
`cmd //c mklink /J playlist <path-to-main-repo>\playlist`.

## Security note: loading the pretrained checkpoint

`djtransgan.utils.load_pt()` is an unguarded `torch.load()` — DJtransGAN's
pretrained `.pt` file comes from Google Drive, a third-party source. This
pipeline never calls `load_pt()`; `spike_transition_inference.py` calls
`torch.load(path, weights_only=True, map_location="cpu")` directly instead,
which restricts unpickling to tensors rather than arbitrary Python objects.
It worked without error, confirming the checkpoint only contains plain
tensors. Our own checkpoints (`export/*.pt`, written and read within the same
pipeline) should use `torch.load(..., weights_only=True)` explicitly for the
same reason. The `np.load(..., allow_pickle=True)` calls on
`data/transition_curves.npz` (future Task 6/7 work) are safe despite
`allow_pickle` — that file is written by `build_transition_dataset.py` in
this same pipeline, never from a third-party source; `allow_pickle` is
required there only because the per-pair automation curves are ragged
(variable-length) arrays, not because the source is untrusted.

## Dependency findings (deviations from the original minimal-dependency plan)

The design's premise — that DJtransGAN's `model`/`mixer`/`frontend` code only
needs `torch`, `torchaudio`, `torchlibrosa`, `nnAudio`, and
`asteroid-filterbanks` — was **directionally right but incomplete**. Actually
verified by import-tracing and running the spike:

- **`torch`/`torchaudio` must be pinned to `2.2.0`**, not left unpinned. Newer
  torchaudio removed `set_audio_backend()`, which
  `djtransgan/utils/utils.py` calls unconditionally at import time. 2.2.0 is
  the oldest version pair with a cp312 wheel that still has it (deprecated,
  no-op, but present).
- **`numpy<2` and `scipy<1.14`** are required — torch 2.2.0's compiled
  extensions were built against NumPy's 1.x C-API.
- **Additional genuine runtime imports** not in the original list:
  `pyloudnorm`, `pyrubberband` (both pulled in by
  `djtransgan/utils/manipulate.py`, eager-imported by
  `djtransgan/utils/__init__.py`), `matplotlib`/`ipython` (pulled in by
  `djtransgan/utils/visualize.py`, same eager-import chain), `openunmix`
  (pulled in by `djtransgan/frontend/asteroid.py` — despite the module name,
  it calls `openunmix.transforms.make_filterbanks`, not anything from the
  `asteroid-filterbanks` PyPI package), and `acoustics` (pulled in by
  `djtransgan/dataset/noise.py`, eager-imported by
  `djtransgan/dataset/__init__.py`, needed for `select_audio_region`).
- **`djtransgan.process` (and its `preprocess()`/`postprocess()` convenience
  wrappers) is unusable on Python 3.12.** `djtransgan/process/__init__.py`
  eager-imports `djtransgan/process/beat.py`, which imports `madmom` for
  beat tracking. `madmom` 0.16.1 (the latest PyPI release, 2018) fails to
  import on Python 3.12 for reasons independent of any version pinning done
  here: it uses `from collections import MutableSequence` (that alias was
  removed from `collections` in Python 3.10) and `np.float` (removed from
  NumPy in 1.24). There is no numpy/Python combination that has wheels for
  both cp312 and pre-removal APIs — this is a genuine dead end, not a
  transient install failure. This was **not** in the original minimal-import
  investigation, which only checked `model`/`mixer`/`frontend`; `process` is
  a fourth package the brief's Step 4 script also imported
  (`from djtransgan.process import preprocess, postprocess`).
  `spike_transition_inference.py` bypasses `djtransgan.process` entirely and
  calls `djtransgan.dataset.select_audio_region` (verified madmom-free)
  directly, replicating only the tail of `preprocess()` (loudness
  normalization + cue-region selection). This is also the shape Tasks 6-8
  actually need: `preprocess()`'s beat tracking, BPM sync, and cue-bar
  snapping are exactly what TEMPO's own beat grid already does — Task 6 was
  always going to supply pre-aligned audio and its own cue points, not run
  DJtransGAN's beat tracker.

## Verified shapes (Task 1 spike, 2026-08-06)

Ran against `playlist/01_Obsessed_SpotiDost.mp3` (prev) and
`playlist/02_Perfect_it_could_be_SpotiDost.mp3` (next), cue points 20s from
each track's transition edge, `djtransgan_minmax.pt` weights:

```
prev:
  band: shape=(1, 3, 1, 1025) dtype=float32 min=0.0000 max=1.0000 has_nan=False
  fader: shape=(1, 4, 1, 5168) dtype=float32 min=0.0000 max=1.0000 has_nan=False
  mask: shape=(1, 4, 1, 1025, 5168) dtype=float32 min=0.0000 max=1.0000 has_nan=False
next:
  band: shape=(1, 3, 1, 1025) dtype=float32 min=0.0000 max=1.0000 has_nan=False
  fader: shape=(1, 4, 1, 5168) dtype=float32 min=0.0000 max=1.0000 has_nan=False
  mask: shape=(1, 4, 1, 1025, 5168) dtype=float32 min=0.0000 max=1.0000 has_nan=False
```

Output wav: 60.0s, rms=0.2704 (non-silent). `SPIKE PASSED`.

All values are in `[0, 1]` (min/max both exactly 0.0/1.0 across band, fader,
and mask — consistent with these being gate/mask-style outputs, not raw
dB curves) with no NaNs. Task 7's `distill_labels.py` is written generically
(squeeze/flatten) but was validated against these shapes; if a different
track pair or weights file produces meaningfully different shapes, Task 7
may need adjustment.
