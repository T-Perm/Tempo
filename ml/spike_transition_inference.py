"""Spike: verify DJtransGAN's pretrained generator produces usable EQ/fader
automation curves (not just rendered audio) on our own library's tracks.
Run from ml/ with the venv active: python spike_transition_inference.py

NOTE on djtransgan.process: the brief's original design called
djtransgan.process.preprocess(), but djtransgan/process/__init__.py
unconditionally imports djtransgan/process/beat.py, which imports madmom.
madmom 0.16.1 (the latest PyPI release) does not import on Python 3.12 --
independent of any torch/numpy version pinning here, it uses
`from collections import MutableSequence` (removed in Python 3.10) and
`np.float` (removed in NumPy 1.24). That's not a workaround-able version
skew; it's a dead API surface. But preprocess() only uses madmom for beat
tracking / BPM sync / cue-bar snapping -- exactly the job TEMPO's own beat
grid will do in Task 6 ("Real cue points come from our own beat grid in
Task 6", per the original brief comment below). So this spike bypasses
djtransgan.process entirely and constructs the generator's inputs directly
via djtransgan.dataset.select_audio_region (verified madmom/essentia-free),
replicating only the tail of preprocess() (normalize + cue-region select).
This also matches how Tasks 6-8 will actually call the generator -- they
don't need BPM sync either, since TEMPO already time-stretches decks itself.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vendor", "djtransgan"))

import torch
import soundfile as sf
from djtransgan.config import settings
from djtransgan.utils import load_audio, squeeze_dim, out_audio, normalize
from djtransgan.dataset import select_audio_region
from djtransgan.model import get_generator

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
    # DJtransGAN's own load_pt() is a bare `torch.load(in_path)` with no
    # weights_only guard (djtransgan/utils/utils.py). We call torch.load
    # directly with weights_only=True instead of using their wrapper, since
    # the pretrained .pt comes from Google Drive (third-party source) and
    # this restricts unpickling to plain tensors rather than arbitrary
    # Python objects.
    state_dict = torch.load(WEIGHTS, weights_only=True, map_location="cpu")
    generator.load_state_dict(state_dict)
    generator.eval()

    prev_audio = load_audio(prev_path)
    next_audio = load_audio(next_path)
    prev_audio = normalize(prev_audio)
    next_audio = normalize(next_audio)

    # Cue points: 20s from the end of prev, 20s from the start of next --
    # same 20s convention UnmixDB's own cue-region docs use, arbitrary but
    # reasonable for a first spike. Real cue points come from our own beat
    # grid in Task 6. The 16s window below stands in for
    # djtransgan.process.select_cue_points' "CUE_BAR=8 bars back from the
    # downbeat" snap, which needs madmom's beat tracker; here it's just a
    # fixed-width region since we're validating curve shape/range, not
    # musical alignment quality.
    prev_cue_point = max(0, prev_audio.shape[-1] / settings.SR - 20)
    next_cue_point = 20
    prev_cues = [max(0, prev_cue_point - 16), prev_cue_point]
    next_cues = [max(0, next_cue_point - 16), next_cue_point]

    prev_audio_for_g, _, _ = select_audio_region(prev_audio, prev_cues, settings.N_TIME, True, 0)
    next_audio_for_g, next_cues_for_g, _ = select_audio_region(next_audio, next_cues, settings.N_TIME, True, 1)

    pair_audio_for_g = [prev_audio_for_g.unsqueeze(0), next_audio_for_g.unsqueeze(0).to(torch.float32)]
    cue_for_g = next_cues_for_g.unsqueeze(0).to(torch.float32)

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
