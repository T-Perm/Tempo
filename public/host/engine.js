// TEMPO DJ engine — rules-based (not ML) track sequencing + playback.
// Scope per /office-hours + /plan-eng-review: BPM + a simple loudness-based
// "energy" proxy only. True musical-key detection is deliberately out of
// scope for this 2-week pilot (no simple, reliable browser library exists;
// BPM + energy is enough to test the sequencing claim without over-building).
import { analyze as detectBpm } from 'https://esm.sh/web-audio-beat-detector@8';

const ENERGY_CYCLE_MS = 20 * 60 * 1000; // one build/peak/cool cycle per 20 min of a set
const BPM_PENALTY_WEIGHT = 0.01; // per BPM of difference from the current track
const CROSSFADE_MS = 4000;

export class MusicEngine {
  constructor({ onNowPlaying, onLibraryProgress, onLibraryError }) {
    this.onNowPlaying = onNowPlaying || (() => {});
    this.onLibraryProgress = onLibraryProgress || (() => {});
    this.onLibraryError = onLibraryError || (() => {});

    /** @type {{name: string, handle: FileSystemFileHandle, bpm: number, energy: number}[]} */
    this.library = [];
    this.played = new Set();
    /** Guest-approved tracks waiting to play next — "insert at next gap", per design decision. */
    this.queue = [];
    this.current = null;
    this.setStartedAt = null;

    this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    this.playerA = this._makePlayer();
    this.playerB = this._makePlayer();
    this.activePlayer = this.playerA;
    this.standbyPlayer = this.playerB;
  }

  _makePlayer() {
    const audio = new Audio();
    audio.crossOrigin = 'anonymous';
    const gain = this.audioCtx.createGain();
    const source = this.audioCtx.createMediaElementSource(audio);
    source.connect(gain).connect(this.audioCtx.destination);
    gain.gain.value = 0;
    return { audio, gain };
  }

  /**
   * Chrome/Edge create every AudioContext suspended until it's resumed inside
   * a user gesture. Must be called synchronously at the top of the click
   * handler that starts the library load — resuming later (after the picker
   * and library pre-analysis awaits) can miss the gesture-activation window,
   * leaving playback silent with no error.
   */
  unlockAudio() {
    if (this.audioCtx.state === 'suspended') return this.audioCtx.resume();
    return Promise.resolve();
  }

  /** Guest/host picks a local folder via the File System Access API (Chromium only). */
  async loadLibraryFromDirectory() {
    if (!window.showDirectoryPicker) {
      throw new Error('This browser does not support local folder access. Use Chrome or Edge.');
    }
    const dirHandle = await window.showDirectoryPicker();
    const fileHandles = [];
    for await (const [name, handle] of dirHandle.entries()) {
      if (handle.kind === 'file' && /\.(mp3|wav|m4a|ogg)$/i.test(name)) {
        fileHandles.push({ name, handle });
      }
    }
    if (fileHandles.length === 0) {
      throw new Error('No audio files (.mp3/.wav/.m4a/.ogg) found in that folder.');
    }
    await this._preAnalyze(fileHandles);
  }

  /** Pre-analyzes the whole library once, at setup time — not live during playback. Perf review 4A. */
  async _preAnalyze(fileHandles) {
    const analyzed = [];
    let done = 0;
    for (const { name, handle } of fileHandles) {
      try {
        const file = await handle.getFile();
        const arrayBuffer = await file.arrayBuffer();
        const audioBuffer = await this.audioCtx.decodeAudioData(arrayBuffer.slice(0));
        const bpm = await detectBpm(audioBuffer).catch(() => 120); // fall back to a neutral BPM rather than dropping the track
        const energy = this._estimateEnergy(audioBuffer);
        analyzed.push({ name, handle, bpm, energy, duration: audioBuffer.duration });
      } catch (err) {
        // Bad/missing metadata or corrupt file — skip it, don't crash pre-analysis. Failure-mode T5.
        this.onLibraryError({ name, error: String(err && err.message ? err.message : err) });
      }
      done += 1;
      this.onLibraryProgress({ done, total: fileHandles.length });
    }

    if (analyzed.length === 0) {
      throw new Error('Every file in that folder failed to analyze — pick a different folder.');
    }

    // Normalize energy to 0..1 across the actual library, not an assumed global scale.
    const energies = analyzed.map((t) => t.energy);
    const min = Math.min(...energies);
    const max = Math.max(...energies);
    const range = max - min || 1;
    this.library = analyzed.map((t) => ({ ...t, energy: (t.energy - min) / range }));
  }

  _estimateEnergy(audioBuffer) {
    // Simple RMS loudness proxy across all channels — cheap, no ML, "boring by default".
    let sumSquares = 0;
    let count = 0;
    for (let ch = 0; ch < audioBuffer.numberOfChannels; ch++) {
      const data = audioBuffer.getChannelData(ch);
      // Sample every 100th frame — full-resolution RMS is unnecessary for a coarse energy proxy.
      for (let i = 0; i < data.length; i += 100) {
        sumSquares += data[i] * data[i];
        count += 1;
      }
    }
    return count > 0 ? Math.sqrt(sumSquares / count) : 0;
  }

  _energyTarget() {
    const elapsed = Date.now() - (this.setStartedAt || Date.now());
    return 0.6 + 0.3 * Math.sin((2 * Math.PI * elapsed) / ENERGY_CYCLE_MS);
  }

  /** Rules-based pick: closest to the current energy target, penalized by BPM distance from the current track. */
  _pickNextTrack() {
    if (this.queue.length > 0) return this.queue.shift(); // guest-approved requests take the next gap first

    if (this.library.length === 0) return null;

    let candidates = this.library.filter((t) => !this.played.has(t.name));
    if (candidates.length === 0) {
      // Library exhausted — reset rather than stopping playback dead. Failure-mode T4.
      this.played.clear();
      candidates = this.library;
    }

    const target = this._energyTarget();
    let best = null;
    let bestScore = -Infinity;
    for (const track of candidates) {
      const energyDelta = Math.abs(track.energy - target);
      const bpmDelta = this.current ? Math.abs(track.bpm - this.current.bpm) : 0;
      const score = -energyDelta - bpmDelta * BPM_PENALTY_WEIGHT;
      if (score > bestScore) {
        bestScore = score;
        best = track;
      }
    }
    return best;
  }

  /** Guest request approved (by host or 90s auto-timeout) — matches it to a library track by name and queues it. */
  enqueueRequestedTrack(trackName) {
    const match = this.library.find(
      (t) => t.name.toLowerCase().includes(trackName.toLowerCase())
    );
    if (match) this.queue.push(match);
    // If no local match exists, silently skip rather than crash — the pilot's
    // library is host-provided and may not contain every requested song.
  }

  async start() {
    this.setStartedAt = Date.now();
    await this._playNext();
  }

  async _playNext() {
    const track = this._pickNextTrack();
    if (!track) return; // empty library — nothing to do, already surfaced via loadLibraryFromDirectory errors

    try {
      const file = await track.handle.getFile();
      const url = URL.createObjectURL(file);
      const incoming = this.standbyPlayer;
      incoming.audio.src = url;

      await new Promise((resolve, reject) => {
        incoming.audio.oncanplaythrough = resolve;
        incoming.audio.onerror = () => reject(new Error(`Failed to load "${track.name}"`));
      });

      this.played.add(track.name);
      const previousPlayer = this.activePlayer;
      await this._crossfade(previousPlayer, incoming);
      this.activePlayer = incoming;
      this.standbyPlayer = previousPlayer;
      this.current = track;
      this.onNowPlaying(track);

      // Schedule the next pick to start crossfading before this track ends.
      const msUntilNext = Math.max(0, (track.duration * 1000) - CROSSFADE_MS - 500);
      this._nextTimer = setTimeout(() => this._playNext(), msUntilNext);

      incoming.audio.onerror = () => {
        // Mid-transition failure — skip immediately rather than leave dead air. Failure-mode T3.
        clearTimeout(this._nextTimer);
        this._playNext();
      };
    } catch (err) {
      // This track failed to load — try the next one immediately instead of stopping. Failure-mode T3.
      this.onLibraryError({ name: track.name, error: String(err && err.message ? err.message : err) });
      this._playNext();
    }
  }

  async _crossfade(outgoing, incoming) {
    await incoming.audio.play();
    const start = this.audioCtx.currentTime;
    const duration = CROSSFADE_MS / 1000;
    incoming.gain.gain.setValueAtTime(0, start);
    incoming.gain.gain.linearRampToValueAtTime(1, start + duration);
    outgoing.gain.gain.setValueAtTime(outgoing.gain.gain.value, start);
    outgoing.gain.gain.linearRampToValueAtTime(0, start + duration);
    await new Promise((resolve) => setTimeout(resolve, CROSSFADE_MS));
    outgoing.audio.pause();
  }
}
