# ml/distill_labels.py
import json
import os

import numpy as np

EQ_MIN_DB = -18  # matches engine.js's EQ_MIN_DB constant — same clamp range on both sides


def _direction_change_rate(x, thresh=0.0003):
    """Fraction of consecutive same-direction-move pairs that flip sign —
    a curve that moves smoothly in one direction has a low rate; one with
    lots of small back-and-forth wobble has a high rate. `thresh` filters
    sub-sample noise (mean abs per-step diff on real curves is ~3e-4) from
    counting as a "move" at all."""
    d = np.diff(x)
    sign = np.sign(np.where(np.abs(d) > thresh, d, 0))
    sign = sign[sign != 0]
    if len(sign) < 2:
        return 0.0
    return float((np.diff(sign) != 0).mean())


def reduce_curve(fader, band):
    """Reduces one sample's raw automation curves to (transitionMs, duckDb,
    fxIntensity) — the three scalars engine.js's _transitionPlan() already
    predicts. Assumes fader/band are 1D float arrays over the transition's
    time axis, normalized 0..1 by DJtransGAN's sigmoid output layer
    (generator.py sets last_activate='sigmoid').

    Design correction, found empirically after Task 6 first produced real
    curves (the original formulas below were written before any real
    DJtransGAN output existed to check them against):
    - The original "moving" threshold (0.01) was ~30x too coarse relative to
      real per-step diffs (mean ~3e-4 on actual curves) — active_fraction
      rounded to ~0 for all 150 samples, pinning transitionMs at its floor
      for every sample regardless of input.
    - band.max()-band.min() (the original duck-depth signal) is 1.0 for
      literally every sample — DJtransGAN's EQ automation saturates to both
      sigmoid extremes in essentially every real transition, so curve RANGE
      carries no information. band.mean() (how much of the transition is
      spent cut, not how far it CAN cut) does vary sample-to-sample and is
      used instead.
    - combined.std()*4 (the original fx-intensity signal) sits in a narrow
      0.47-0.49 band across all 150 samples — an artifact of sigmoid-
      saturated curves generally having std near that value regardless of
      shape, not a measure of how modulated/expressive a given curve is.
      Direction-change rate (how often the curve reverses, not how far it
      spans) is used instead — verified to vary meaningfully across samples.
    All three replacements were checked against the actual 150-sample Task 6
    dataset before being adopted: transitionMs now spans ~17s-56s, duckDb
    ~-15 to -12dB, fxIntensity ~0.26-0.83 (previously: 15.0-15.6s/-18/1.0 for
    all 150 samples — a model trained on the original labels would have
    learned three constants, not a function of its inputs)."""
    # Duration: span where the fader is actively moving (not pinned at 0 or
    # 1), as a fraction of the curve length, scaled to a plausible transition
    # duration range (15s-90s, matching engine.js's PEAK_BLEND_SEC..VALLEY_BLEND_SEC).
    moving = np.abs(np.diff(fader)) > 0.001
    active_fraction = moving.mean() if len(moving) > 0 else 0.5
    transition_ms = float(np.clip(15 + active_fraction * 300, 15, 90) * 1000)

    # Duck depth: how much of the transition is spent with the band cut
    # (1 - mean level), scaled to engine.js's EQ_MIN_DB..0 range.
    duck_depth = float(np.clip(1 - band.mean(), 0, 1))
    duck_db = -duck_depth * abs(EQ_MIN_DB)

    # FX intensity: how often the fader/band curves reverse direction — a
    # smooth one-way fade reads as a plain crossfade (low intensity); a
    # curve with lots of direction changes reads as an intentional,
    # expressive move (high intensity). Scale factor (40) chosen so the
    # observed rate range (~0.006-0.021 combined) maps to a usable 0-1 spread.
    rate = (_direction_change_rate(fader) + _direction_change_rate(band)) / 2
    fx_intensity = float(np.clip(rate * 40, 0, 1))

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
