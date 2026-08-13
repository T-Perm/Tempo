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
