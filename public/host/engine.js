// TEMPO DJ engine — rules-based (not ML) track sequencing + beatmatched mixing.
// Scope per /office-hours + /plan-eng-review: BPM + a simple loudness-based
// "energy" proxy for selection. True musical-key detection is deliberately out
// of scope (no simple, reliable browser library exists). Beat-grid detection
// below is a pilot-scale spectral-novelty onset detector, not a production DSP
// library — good enough to phrase-align a mix, not a substitute for real
// beat-tracking research. Flagged as a standing risk in /autoplan's eng review.
import { analyze as detectBpm } from 'https://esm.sh/web-audio-beat-detector@8';

const ENERGY_CYCLE_MS = 20 * 60 * 1000; // one build/peak/cool cycle per 20 min of a set
const BPM_PENALTY_WEIGHT = 0.01; // per BPM of difference from the current track
const CROSSFADE_MS = 4000;
const TEMPO_STRETCH_CAP = 0.06; // ±6% playback-rate adjustment ceiling before we bail on tempo-sync
const EQ_MIN_DB = -18; // cut-only EQ — no boost, avoids needing a limiter (autoplan eng decision)
const PHRASE_BEATS = 32; // 8-bar phrase at 4/4 — theoretical grid from the first detected onset
const BEAT_GRID_MIN_ONSETS = 4; // fewer than this = degenerate, fall back to average BPM

export class MusicEngine {
  constructor({ onNowPlaying, onLibraryProgress, onLibraryError, onManualStateChange, onAutoPilotResumed }) {
    this.onNowPlaying = onNowPlaying || (() => {});
    this.onLibraryProgress = onLibraryProgress || (() => {});
    this.onLibraryError = onLibraryError || (() => {});
    this.onManualStateChange = onManualStateChange || (() => {}); // (armed: boolean)
    this.onAutoPilotResumed = onAutoPilotResumed || (() => {});

    /** @type {{name: string, handle: FileSystemFileHandle, bpm: number, energy: number, beatGrid: number[]|null, beatGridBpm: number|null}[]} */
    this.library = [];
    this.played = new Set();
    /** Guest-approved tracks waiting to play next — "insert at next gap", per design decision. */
    this.queue = [];
    this.current = null;
    this.setStartedAt = null;

    // Manual mixing state — global arm (autoplan Design decision #5c): touching
    // any control arms manual mode for the whole session, not per-control.
    this.manualArmed = false;
    this._deferredAutoTransition = false; // track-end timer fired while armed — replay on backToAuto()

    this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    this.compressor = this.audioCtx.createDynamicsCompressor();
    this.compressor.connect(this.audioCtx.destination);

    // Stable, UI-bindable deck objects (autoplan Eng decision: expose playerA/
    // playerB directly rather than swapping identity on every crossfade).
    this.playerA = this._makePlayer('A');
    this.playerB = this._makePlayer('B');
    this._liveDeckId = 'A'; // which deck is currently the audible "front" deck
  }

  _makePlayer(id) {
    const audio = new Audio();
    audio.crossOrigin = 'anonymous';
    audio.preservesPitch = true; // explicit, not relying on the browser's implicit default
    audio.mozPreservesPitch = true;
    audio.webkitPreservesPitch = true;

    const source = this.audioCtx.createMediaElementSource(audio);
    const low = this.audioCtx.createBiquadFilter();
    low.type = 'lowshelf';
    low.frequency.value = 320;
    const mid = this.audioCtx.createBiquadFilter();
    mid.type = 'peaking';
    mid.frequency.value = 1000;
    mid.Q.value = 0.9;
    const high = this.audioCtx.createBiquadFilter();
    high.type = 'highshelf';
    high.frequency.value = 3200;
    const gain = this.audioCtx.createGain();
    gain.gain.value = 0;

    source.connect(low).connect(mid).connect(high).connect(gain).connect(this.compressor);
    return { id, audio, low, mid, high, gain, eq: { low: 0, mid: 0, high: 0 }, cuePoint: null };
  }

  get liveDeckId() {
    return this._liveDeckId;
  }

  _deckById(id) {
    return id === 'A' ? this.playerA : this.playerB;
  }

  get activePlayer() {
    return this._deckById(this._liveDeckId);
  }

  get standbyPlayer() {
    return this._deckById(this._liveDeckId === 'A' ? 'B' : 'A');
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
        const { beatGrid, beatGridBpm } = this._analyzeBeatGrid(audioBuffer, bpm);
        analyzed.push({ name, handle, bpm, energy, duration: audioBuffer.duration, beatGrid, beatGridBpm });
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

  /**
   * Pilot-scale onset detector: energy-flux novelty function with an adaptive
   * local-mean threshold, peak-picked into onset timestamps. This is a coarse
   * approximation (no spectral/FFT analysis, just frame energy deltas) — good
   * enough to phrase-align a mix, not a substitute for a real beat-tracking
   * library. Sanity-checked against the whole-track average BPM to catch
   * half/double-time misreads before it drives tempo-sync.
   */
  _analyzeBeatGrid(audioBuffer, averageBpm) {
    const onsets = this._detectOnsets(audioBuffer);
    if (onsets.length < BEAT_GRID_MIN_ONSETS) {
      return { beatGrid: null, beatGridBpm: null }; // degenerate — fall back to average-BPM sync
    }

    const intervals = [];
    for (let i = 1; i < onsets.length; i++) intervals.push(onsets[i] - onsets[i - 1]);
    intervals.sort((a, b) => a - b);
    const medianInterval = intervals[Math.floor(intervals.length / 2)];
    if (!medianInterval || medianInterval <= 0) {
      return { beatGrid: null, beatGridBpm: null };
    }

    let beatGridBpm = 60 / medianInterval;
    while (beatGridBpm < 70) beatGridBpm *= 2;
    while (beatGridBpm > 180) beatGridBpm /= 2;

    // Half/double-time sanity check against the independent average-BPM detector.
    const ratio = beatGridBpm / averageBpm;
    if (ratio > 1.8 || ratio < 0.55) {
      return { beatGrid: null, beatGridBpm: null }; // likely misdetection — don't trust the grid
    }

    return { beatGrid: onsets, beatGridBpm };
  }

  _detectOnsets(audioBuffer) {
    const sampleRate = audioBuffer.sampleRate;
    const frameSize = 1024;
    const hop = 512;
    const ch0 = audioBuffer.getChannelData(0);
    const ch1 = audioBuffer.numberOfChannels > 1 ? audioBuffer.getChannelData(1) : null;
    const len = ch0.length;

    const energies = [];
    for (let i = 0; i + frameSize <= len; i += hop) {
      let sum = 0;
      for (let j = 0; j < frameSize; j++) {
        const sample = ch1 ? (ch0[i + j] + ch1[i + j]) / 2 : ch0[i + j];
        sum += sample * sample;
      }
      energies.push(Math.sqrt(sum / frameSize));
    }

    const novelty = [0];
    for (let i = 1; i < energies.length; i++) {
      novelty.push(Math.max(0, energies[i] - energies[i - 1]));
    }

    const localWindow = 43; // ~0.5s at hop=512/44.1kHz — local-mean window for the adaptive threshold
    const minOnsetGapSec = 0.15; // avoid double-counting the same transient
    const onsets = [];
    for (let i = 0; i < novelty.length; i++) {
      const start = Math.max(0, i - localWindow);
      const end = Math.min(novelty.length, i + localWindow);
      let localSum = 0;
      for (let k = start; k < end; k++) localSum += novelty[k];
      const localMean = localSum / (end - start);
      if (novelty[i] > localMean * 1.5 && novelty[i] > 0.01) {
        const t = (i * hop) / sampleRate;
        const last = onsets[onsets.length - 1];
        if (last === undefined || t - last > minOnsetGapSec) onsets.push(t);
      }
    }
    return onsets;
  }

  /** Nearest theoretical phrase boundary (32 beats) at or after `fromSec`, using the track's own beat grid. */
  _phraseBoundaryAfter(track, fromSec) {
    if (!track.beatGrid || !track.beatGridBpm) return null;
    const beatLen = 60 / track.beatGridBpm;
    const phraseLen = beatLen * PHRASE_BEATS;
    const anchor = track.beatGrid[0];
    if (fromSec <= anchor) return anchor;
    const phrasesElapsed = Math.ceil((fromSec - anchor) / phraseLen);
    return anchor + phrasesElapsed * phraseLen;
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

  /**
   * Guest request approved (by host or 90s auto-timeout) — matches it to a
   * library track by word-boundary substring match. Rejects (no match) when
   * multiple tracks match ambiguously rather than silently guessing the
   * first one — this now feeds beat-grid detection + tempo-sync on the hot
   * path, so a bad match has a bigger blast radius than before.
   */
  enqueueRequestedTrack(trackName) {
    const needle = trackName.trim().toLowerCase();
    if (!needle) return;
    const pattern = new RegExp(`\\b${needle.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'i');
    const matches = this.library.filter((t) => pattern.test(t.name));
    if (matches.length === 1) this.queue.push(matches[0]);
    // Zero or ambiguous (2+) matches: silently skip rather than guess — the
    // pilot's library is host-provided and may not contain every requested song.
  }

  async start() {
    this.setStartedAt = Date.now();
    await this._playNext();
  }

  async _playNext() {
    if (this.manualArmed) {
      // Track-end timer fired while the host is driving manually — defer
      // rather than double-fire a transition on top of a manual mix in progress.
      this._deferredAutoTransition = true;
      return;
    }

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

      if (this.manualArmed) {
        // Host armed manual mode while this track was loading — abandon this
        // auto-transition rather than silently overwrite their live mix once
        // the load finishes. Closes most of the race window (loading is the
        // slow part); the crossfade itself is short enough that arming
        // exactly mid-fade is a narrower, documented remaining edge case.
        this._deferredAutoTransition = true;
        return;
      }

      this.played.add(track.name);
      const outgoing = this.activePlayer;
      await this._crossfade(outgoing, incoming, this.current, track);
      incoming.track = track; // tracked per-deck so manual crossfader moves can report the right "now playing"
      incoming.cuePoint = null; // a new track on this deck invalidates any cue point from the previous one
      this._liveDeckId = incoming.id;
      this.current = track;
      this.onNowPlaying(track);

      // Schedule the next pick to start crossfading at a phrase boundary near
      // the end of this track when we have a beat grid, else the old fixed offset.
      const fallbackStart = Math.max(0, track.duration - CROSSFADE_MS / 1000 - 0.5);
      const phraseStart = this._phraseBoundaryAfter(track, Math.max(0, track.duration - 20));
      const mixOutAt = phraseStart !== null && phraseStart < track.duration ? phraseStart : fallbackStart;
      const msUntilNext = Math.max(0, mixOutAt * 1000);
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

  /** Equal-power crossfade with tempo-sync (when both tracks have a trustworthy beat grid) and a bass-swap EQ duck. */
  async _crossfade(outgoing, incoming, outgoingTrack, incomingTrack) {
    const start = this.audioCtx.currentTime;
    const duration = CROSSFADE_MS / 1000;
    const now = this.audioCtx.currentTime;

    // Cancel any leftover scheduled automation on both gains before writing new
    // curves — otherwise a prior ramp's tail can fight this one (autoplan Eng
    // decision: AudioParam race prevention).
    incoming.gain.gain.cancelScheduledValues(now);
    incoming.gain.gain.setValueAtTime(incoming.gain.gain.value, now);
    outgoing.gain.gain.cancelScheduledValues(now);
    outgoing.gain.gain.setValueAtTime(outgoing.gain.gain.value, now);

    // Tempo-sync: match incoming's rate to outgoing's, capped at ±6%. Beyond
    // the cap (or missing beat grids), skip tempo-stretch and phrase-alignment
    // for this one mix — equal-power crossfade only (autoplan Eng decision).
    let mixInOffset = 0;
    let rate = 1;
    const bothGridded = outgoingTrack && outgoingTrack.beatGridBpm && incomingTrack.beatGridBpm;
    if (bothGridded) {
      const idealRate = outgoingTrack.beatGridBpm / incomingTrack.beatGridBpm;
      if (Math.abs(idealRate - 1) <= TEMPO_STRETCH_CAP) {
        rate = idealRate;
        mixInOffset = incomingTrack.beatGrid[0]; // start on the incoming track's first detected onset, not silence
      }
    }
    incoming.audio.playbackRate = rate;
    if (mixInOffset > 0 && mixInOffset < incomingTrack.duration - duration) {
      incoming.audio.currentTime = mixInOffset;
    }

    await incoming.audio.play();

    // Equal-power curve (constant perceived loudness through the fade) instead
    // of a flat linear ramp.
    const steps = 30;
    const fadeIn = new Float32Array(steps + 1);
    const fadeOut = new Float32Array(steps + 1);
    for (let i = 0; i <= steps; i++) {
      const x = i / steps;
      fadeIn[i] = Math.sin(x * Math.PI * 0.5);
      fadeOut[i] = Math.cos(x * Math.PI * 0.5);
    }
    incoming.gain.gain.setValueCurveAtTime(fadeIn, start, duration);
    outgoing.gain.gain.setValueCurveAtTime(fadeOut, start, duration);

    // Bass-swap EQ duck on the outgoing track during the fade (classic DJ
    // mixing move — avoids two competing bass lines) — only when the host
    // hasn't manually set EQ (manual values always win).
    if (outgoing.eq.low === 0) {
      outgoing.low.gain.cancelScheduledValues(now);
      outgoing.low.gain.setValueAtTime(0, start);
      outgoing.low.gain.linearRampToValueAtTime(EQ_MIN_DB, start + duration * 0.6);
    }

    await new Promise((resolve) => setTimeout(resolve, CROSSFADE_MS));
    if (this.manualArmed) return; // host took over mid-fade — don't force-finalize over their live mix
    outgoing.audio.pause();
    outgoing.audio.playbackRate = 1;
    if (outgoing.eq.low === 0) outgoing.low.gain.setValueAtTime(0, this.audioCtx.currentTime);
  }

  // ---- Manual mixer console API ----
  // Global arm: touching any control here arms manual mode for the whole
  // session (not per-control) — simpler state machine for a pilot-scale build.

  armManual() {
    if (this.manualArmed) return;
    this.manualArmed = true;
    const now = this.audioCtx.currentTime;
    // Cancel scheduled automation and re-anchor at current value on every
    // AudioParam a manual control can touch, so a live drag doesn't fight a
    // ramp left over from an in-flight algorithmic crossfade.
    for (const deck of [this.playerA, this.playerB]) {
      for (const node of [deck.gain.gain, deck.low.gain, deck.mid.gain, deck.high.gain]) {
        node.cancelScheduledValues(now);
        node.setValueAtTime(node.value, now);
      }
    }
    clearTimeout(this._nextTimer);
    // Arming always interrupts whatever scheduled auto-transition was
    // pending (fired or not) — returning to auto must always re-trigger a
    // fresh pick, or playback silently stops advancing forever.
    this._deferredAutoTransition = true;
    this.onManualStateChange(true);
  }

  backToAuto() {
    if (!this.manualArmed) return;
    this.manualArmed = false;

    // Reset any manual EQ/tempo changes back to neutral — otherwise a bass
    // cut or tempo nudge from this session silently persists into every
    // future automatic crossfade, and the bass-swap duck stays disabled for
    // whichever deck the host touched.
    const now = this.audioCtx.currentTime;
    for (const deck of [this.playerA, this.playerB]) {
      deck.eq = { low: 0, mid: 0, high: 0 };
      for (const band of ['low', 'mid', 'high']) {
        deck[band].gain.cancelScheduledValues(now);
        deck[band].gain.setValueAtTime(0, now);
      }
      deck.audio.playbackRate = 1;
    }

    // Reset gain nodes to match what _crossfade() assumes on its next run
    // (live deck at full gain, the other silent) — otherwise a leftover
    // manual gain position makes the next automatic crossfade's curve jump
    // instead of fade, since setValueCurveAtTime always starts from the
    // curve's own first sample regardless of the node's current value.
    this._deckById(this._liveDeckId).gain.gain.setValueAtTime(1, now);
    this._deckById(this._liveDeckId === 'A' ? 'B' : 'A').gain.gain.setValueAtTime(0, now);

    this.onManualStateChange(false);
    this.onAutoPilotResumed();
    const shouldResume = this._deferredAutoTransition;
    this._deferredAutoTransition = false;
    if (shouldResume) this._playNext(); // fresh pick from the current position, not a stale plan
  }

  /** value: 0 (full Deck A) .. 1 (full Deck B). Equal-power. */
  setCrossfader(value) {
    if (!this.manualArmed) this.armManual();
    let x = Math.min(1, Math.max(0, value));
    // Never fade toward a deck that has no track loaded yet (e.g. very early
    // in a set before the second deck has ever played) — that would silence
    // the only audible deck with no error or indication anything went wrong.
    if (x >= 0.5 && !this.playerB.track) x = 0.499;
    if (x < 0.5 && !this.playerA.track) x = 0.5;
    const now = this.audioCtx.currentTime;
    this.playerA.gain.gain.setValueAtTime(Math.cos(x * Math.PI * 0.5), now);
    this.playerB.gain.gain.setValueAtTime(Math.sin(x * Math.PI * 0.5), now);

    const newLiveDeckId = x < 0.5 ? 'A' : 'B';
    if (newLiveDeckId !== this._liveDeckId) {
      this._liveDeckId = newLiveDeckId;
      const deck = this._deckById(newLiveDeckId);
      if (deck.track) {
        this.current = deck.track;
        this.onNowPlaying(deck.track); // keep the header in sync with a manual crossfader flip, not just auto transitions
      }
    }
  }

  /** deckId: 'A'|'B'. band: 'low'|'mid'|'high'. db: -18..0 (cut-only). */
  setEQ(deckId, band, db) {
    if (!this.manualArmed) this.armManual();
    const deck = this._deckById(deckId);
    const clamped = Math.min(0, Math.max(EQ_MIN_DB, db));
    deck.eq[band] = clamped;
    deck[band].gain.setValueAtTime(clamped, this.audioCtx.currentTime);
  }

  /** value: -1..1, mapped to the same ±6% cap the auto tempo-sync uses. */
  setTempoNudge(deckId, value) {
    if (!this.manualArmed) this.armManual();
    const deck = this._deckById(deckId);
    const clamped = Math.min(1, Math.max(-1, value));
    deck.audio.playbackRate = 1 + clamped * TEMPO_STRETCH_CAP;
  }

  /** Tap: jump to the stored cue point. No cue point set yet: set one at the current position instead. */
  setCue(deckId) {
    if (!this.manualArmed) this.armManual();
    const deck = this._deckById(deckId);
    if (deck.cuePoint === null) {
      deck.cuePoint = deck.audio.currentTime;
    } else {
      deck.audio.currentTime = deck.cuePoint; // live jump on the audience-facing output — no headphone pre-listen (autoplan D4)
    }
  }
}
