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

python -m pip install --upgrade pip
# Pinned to torch/torchaudio 2.2.0, the oldest cp312 wheel pair still exposing
# torchaudio.set_audio_backend(), which djtransgan/utils/utils.py calls
# unconditionally at import time (removed in newer torchaudio). numpy/scipy are
# pinned to versions torch 2.2.0's compiled extensions were built against
# (numpy>=2 breaks the C-API ABI torch 2.2.0 expects). pyloudnorm, pyrubberband,
# openunmix, matplotlib, and ipython are additional runtime imports discovered
# by tracing djtransgan/utils/__init__.py (eager-imports manipulate.py and
# visualize.py) and djtransgan/frontend/asteroid.py — not covered by the
# `asteroid-filterbanks` PyPI package despite the module's name. acoustics is
# a runtime import of djtransgan/dataset/noise.py, eager-imported by
# djtransgan/dataset/__init__.py (needed for select_audio_region).
#
# NOT installed: madmom. djtransgan/process/__init__.py eager-imports
# djtransgan/process/beat.py, which imports madmom for beat tracking. madmom
# 0.16.1 (latest PyPI release, 2018) does not import on Python 3.12 for
# reasons unrelated to version pinning here -- it uses
# `from collections import MutableSequence` (removed in Python 3.10) and
# `np.float` (removed in NumPy 1.24), both hard API removals with no
# compatible numpy/Python combination available as wheels. This means
# djtransgan.process (and its preprocess()/postprocess() convenience
# wrappers, which also do BPM sync and cue-bar snapping we don't want) is
# unusable as-is. spike_transition_inference.py bypasses it and calls
# djtransgan.dataset.select_audio_region directly, matching how Tasks 6-8
# feed the generator from TEMPO's own beat grid instead.
python -m pip install torch==2.2.0 torchaudio==2.2.0 torchlibrosa nnAudio asteroid-filterbanks \
            "numpy<2" "scipy<1.14" librosa soundfile gdown onnx pandas scikit-learn \
            pyloudnorm pyrubberband openunmix matplotlib ipython acoustics

cd vendor/djtransgan
python -c "from djtransgan.utils import download_pretrained; download_pretrained()"
echo "Setup complete. Pretrained weights in vendor/djtransgan/pretrained/"
