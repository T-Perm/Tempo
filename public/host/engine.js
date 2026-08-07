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
// /autoplan 2026-08-04 round 4: was a flat 4000ms (a ~4-6s tail-fade at the
// very end of the track — confirmed by the product owner, from actually
// listening, as "not mixing, just crossfade"). Round 4 fixed that by
// widening the window, but scaled everything off one base times a
// multiplier, which kept every transition clustered in a narrow ~45-91s
// band — round 5: "appropriate" duration needs real range, not just "make
// it longer." Each energy-context below is now an explicit duration, not a
// multiplier of a shared base, so a peak-energy cut can be genuinely quick
// (PEAK_BLEND_SEC) while a valley blend stays long and gentle
// (VALLEY_BLEND_SEC) — the choice is about which is musically appropriate
// for the moment, not a single "how long should transitions be" dial.
const PEAK_BLEND_SEC = 20; // near an energy peak — quick, punchy cut, keep momentum
const RISING_BLEND_SEC = 45; // building toward a peak — a middle-length blend
const BASE_BLEND_SEC = 60; // flat or declining energy, no strong signal either way
const VALLEY_BLEND_SEC = 90; // in a low-energy stretch — long, gentle blend, let both basslines breathe together
const PHRASE_LOOKBACK_PAD_SEC = 20; // width of the search window (before the reserve deadline) for a phrase-aligned mix-out point — /autoplan 2026-08-04 round 4
const TEMPO_STRETCH_CAP = 0.06; // ±6% playback-rate adjustment ceiling before we bail on tempo-sync
const EQ_MIN_DB = -18; // cut-only EQ — no boost, avoids needing a limiter (autoplan eng decision)
const PHRASE_BEATS = 32; // 8-bar phrase at 4/4 — theoretical grid from the first detected onset
const BEAT_GRID_MIN_ONSETS = 4; // fewer than this = degenerate, fall back to average BPM
// Adaptive-threshold peak-picking constants for _detectOnsets — same technique
// _analyzeStructure's novelty peak-picking reuses (STRUCTURE_NOVELTY_LOCAL_WINDOW/
// STRUCTURE_NOVELTY_THRESHOLD_MULT), named here too so the two don't silently
// drift apart during future tuning. Found by /ship performance specialist, 2026-08-05.
const ONSET_LOCAL_WINDOW_FRAMES = 43; // ~0.5s at hop=512/44.1kHz — local-mean window for the adaptive threshold
const ONSET_MIN_GAP_SEC = 0.15; // avoid double-counting the same transient
const ONSET_THRESHOLD_MULT = 1.5;
const ONSET_MIN_NOVELTY = 0.01;

// "DJ feel" — /autoplan 2026-08-02: bounded, killable creativity on top of the
// deterministic engine above. Each mechanism is independently flagged
// (this.creativeFlags) — flip any one to `false` at runtime (e.g. from
// devtools: `engine.creativeFlags.sampling = false`) to fall back to that
// mechanism's pre-existing deterministic behavior with no redeploy. Default
// ON. NOT included: "read-the-room" guest-feedback bias — deferred per the
// D3 gate decision ("the DJ needs to be able to mix before he can read the
// room") — see TODOS.md.
const TOP_K = 3; // sample from the top-K scoring candidates, not always the single best
const SAMPLE_TEMPERATURE = 0.15; // lower = closer to pure argmax; scores are small deltas (~0-1.4 range)
const NOVELTY_WINDOW = 5; // how many recent picks count against a candidate's novelty
const NOVELTY_WEIGHT = 0.3; // comparable magnitude to the existing energy/BPM score terms
const PEAK_ENERGY_THRESHOLD = 0.75; // above this: quicker, punchier transitions
const VALLEY_ENERGY_THRESHOLD = 0.45; // below this: longer, gentler blends

// "DJ feel" round 2 — /autoplan 2026-08-04: FX layer (filter sweep + echo
// tail) and a single-repeat "echo-stutter" loop-roll, on top of the
// mechanisms above. Same kill-switch pattern: creativeFlags.fx /.loopRoll,
// each independently toggleable at runtime with no redeploy.
const SWEEP_MIN_DB = EQ_MIN_DB; // filter sweep rolls the outgoing high band off to this floor, same range as manual EQ
const ECHO_DELAY_SEC = 0.28; // fixed short echo interval — not tempo-synced this pass, matches most 120-135 BPM material closely enough to read as "an echo," not tempo-locked
const ECHO_FEEDBACK = 0.35; // decaying repeats, not infinite (feedback < 1)
const ECHO_WET_LEVEL = 0.55; // peak echo-send level during a transition
const ECHO_TAIL_RELEASE_SEC = 0.9; // setTargetAtTime time-constant for the tail fading to silence after the transition ends, so it can't bleed into the next track's own transition
const LOOP_ROLL_MIN_BEAT_SEC = 0.05; // guards against a degenerate/garbage beatGridBpm producing a near-zero or negative beat length
const LOOP_ROLL_MAX_BEAT_SEC = 3; // guards against a degenerate/garbage beatGridBpm producing an absurdly long "beat"
const RESUME_END_MARGIN_SEC = 0.05; // safety margin so resuming <audio> after a loop-roll never seeks to/past end-of-file
const MIN_SHOWY_SPACING = 3; // transitions between "showy" technique firings (occasion gate) — /autoplan 2026-08-04 round 3
const NEAR_PEAK_MARGIN = 0.05; // occasion gate treats "rising and within this much of PEAK_ENERGY_THRESHOLD" as near-peak, not just past it

// Real per-track structural signal — /autoplan 2026-08-04 round 6. Replaces
// _energyTrajectory()'s synthetic Date.now() sine wave (zero relation to
// the actual audio) with a Foote-style self-similarity + checkerboard
// novelty segmentation over coarse 3-band energy features. No FFT/chroma —
// this file has no spectral-analysis infra and a full MIR pipeline is out
// of scope for a browser app with no server (same "no simple, reliable
// browser library exists" call already made for key detection, line 3).
// Same honesty bar as _detectOnsets: catches real loudness/density
// transitions, does not semantically label "chorus" vs "verse". Returns
// null (never a guess) when the track is too short or the novelty curve is
// degenerate — every caller must treat null as "fall back to the synthetic
// trajectory," never invent boundaries from noise.
const STRUCTURE_LOW_CUTOFF_HZ = 200;
const STRUCTURE_HIGH_CUTOFF_HZ = 2000;
const STRUCTURE_WINDOW_SEC = 1.5; // texture-window duration for the similarity matrix
const STRUCTURE_MIN_WINDOWS = 20; // fewer than this — track too short for a meaningful similarity matrix
const STRUCTURE_MAX_WINDOWS = 300; // caps matrix size (<=300x300) so long tracks stay tractable
const STRUCTURE_KERNEL_HALF = 8; // checkerboard novelty kernel half-width, in windows (~24s full width at STRUCTURE_WINDOW_SEC)
const STRUCTURE_NOVELTY_LOCAL_WINDOW = 8; // local-mean window (in texture windows) for adaptive peak-picking, same technique as _detectOnsets
const STRUCTURE_NOVELTY_THRESHOLD_MULT = 1.5; // same adaptive-threshold multiplier as _detectOnsets, for consistency
const STRUCTURE_MIN_BOUNDARY_GAP_SEC = 8; // avoid double-counting the same structural shift
const STRUCTURE_MIN_BOUNDARIES = 1; // fewer than this — no real structure detected, don't fabricate one
const STRUCTURE_MAX_BOUNDARIES = 40; // more than this — over-segmented noise, not real structure

// Visual DJ controller — /autoplan 2026-08-02. Read-only waveform/beat-grid/
// meter view for the host's check-in confidence glance.
const WAVEFORM_PEAK_COUNT = 2000; // fixed count regardless of track length — keeps peaks array size and render cost duration-independent

// Remembered-folder + track-analysis cache — plain IndexedDB, two stores in
// one DB. FileSystemDirectoryHandle is structured-cloneable in Chromium, so
// it can be stored directly (permission still has to be re-confirmed via
// queryPermission/requestPermission — storage alone doesn't grant access).
const IDB_NAME = 'tempo-library';
const IDB_VERSION = 1;
const DIR_STORE = 'dirHandle';
const DIR_KEY = 'last';
const TRACK_CACHE_STORE = 'trackCache';

// One connection, opened lazily and reused for the whole page session — a
// naive open-per-call here left up to 2x-library-size IndexedDB connections
// dangling per setup pass (never closed), which also blocks any future
// version bump's onupgradeneeded from ever firing. Found by /review
// adversarial pass, 2026-08-05.
let _idbConnPromise = null;
function _idbOpen() {
  if (!_idbConnPromise) {
    _idbConnPromise = new Promise((resolve, reject) => {
      const req = indexedDB.open(IDB_NAME, IDB_VERSION);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(DIR_STORE)) db.createObjectStore(DIR_STORE);
        if (!db.objectStoreNames.contains(TRACK_CACHE_STORE)) db.createObjectStore(TRACK_CACHE_STORE);
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => {
        _idbConnPromise = null; // don't cache a failed open — the next call should retry
        reject(req.error);
      };
    });
  }
  return _idbConnPromise;
}

async function _idbGet(store, key) {
  const db = await _idbOpen();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(store, 'readonly').objectStore(store).get(key);
    tx.onsuccess = () => resolve(tx.result ?? null);
    tx.onerror = () => reject(tx.error);
  });
}

async function _idbSet(store, key, value) {
  const db = await _idbOpen();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(store, 'readwrite').objectStore(store).put(value, key);
    tx.onsuccess = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

// Cursor-based rather than getAll() so key and value come back paired —
// exportLibraryAnalysis needs the key to recover the track name.
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

/** Seeded PRNG (mulberry32) — deterministic given a seed, so a pilot-night pick sequence can be reconstructed post-mortem from the logged seed. */
function mulberry32(seed) {
  let state = seed >>> 0;
  return function () {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

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

    // "DJ feel" creative layer — see the constants block above. Each flag is a
    // plain object property so it's toggleable live from devtools with no
    // redeploy: the pilot-night kill switch.
    // loopRoll defaults OFF — /autoplan 2026-08-04 round 3: the "fires on
    // every eligible transition" trigger read as mechanical/cheap live, not
    // a deliberate DJ move. Replaced with an occasion gate (_shouldLoopRoll)
    // below, but its feel (spacing threshold, peak-proximity margin) is
    // unverified — no live audio playback in this coding session. Flip on
    // for the next listening pass.
    // stems defaults OFF for the same reason loopRoll does — real audio
    // (Demucs-separated drums, when a track's stems exist) hasn't been
    // heard live yet either. /autoplan 2026-08-04 round 7.
    this.creativeFlags = { sampling: true, novelty: true, transitionVariety: true, fx: true, loopRoll: false, stems: false };
    this._recentHistory = []; // last NOVELTY_WINDOW picks' {energy, bpm} — for the novelty penalty
    // Occasion-gate state for _shouldLoopRoll() — tracks technique firings,
    // not track picks (this.played/_recentHistory), so spacing/variety
    // guards are independent of library-loop-around resets.
    this._transitionCount = 0;
    this._lastShowyAt = -Infinity;
    this._lastShowyTechnique = null;
    const rngSeed = Date.now() >>> 0;
    this._rng = mulberry32(rngSeed);
    console.info(`[TEMPO creative] RNG seed: ${rngSeed} (reconstruct tonight's picks with this if needed)`);

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

    // Fan-out (not serial insertion) for the level meter, tapped AFTER gain —
    // must reflect actual audible signal, not pre-fade decoded audio, or the
    // visual controller's meter would show "signal" on a track that's faded
    // to silence. Small fftSize: this only needs a coarse RMS-ish level, not
    // frequency detail, and runs every animation frame.
    const analyser = this.audioCtx.createAnalyser();
    analyser.fftSize = 256;

    // Echo tail (FX layer, creativeFlags.fx) — a fixed short delay+feedback
    // loop tapped off `high` (post-EQ), with its own wet gain so the tail
    // can keep sounding independently of `gain`'s crossfade curve, then be
    // explicitly released (see _crossfade). Silent (`wet.gain = 0`) unless a
    // transition opens it — costs nothing when creativeFlags.fx is off.
    const delay = this.audioCtx.createDelay(1.0);
    delay.delayTime.value = ECHO_DELAY_SEC;
    const feedback = this.audioCtx.createGain();
    feedback.gain.value = ECHO_FEEDBACK;
    const wet = this.audioCtx.createGain();
    wet.gain.value = 0;

    source.connect(low).connect(mid).connect(high).connect(gain);
    gain.connect(this.compressor);
    gain.connect(analyser);
    high.connect(delay);
    delay.connect(feedback).connect(delay); // feedback loop — decaying repeats
    delay.connect(wet).connect(this.compressor); // reaches the mix bus directly, independent of `gain`'s fade

    return {
      id,
      audio,
      low,
      mid,
      high,
      gain,
      wet,
      analyser,
      eq: { low: 0, mid: 0, high: 0 },
      cuePoint: null,
      _loopSource: null, // active loop-roll AudioBufferSourceNode, if any — see _maybeLoopRoll/armManual
    };
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

  /**
   * Remembered-folder support (testing/pilot QoL — not a mixing-behavior
   * change). Checks IndexedDB for a directory handle saved by a previous
   * `loadLibraryFromDirectory()` call, and whether read permission on it is
   * still usable without re-prompting. Does NOT itself read the folder —
   * callers still call `loadLibraryFromDirectory(dirHandle)` with the result,
   * from a user gesture, since resuming a suspended AudioContext and (if
   * permission is only 'prompt') re-granting access both need one.
   * Returns `{ handle, name, needsPermission }` or `null` if nothing usable
   * is remembered.
   */
  async tryLoadRememberedDirectory() {
    const handle = await _idbGet(DIR_STORE, DIR_KEY).catch(() => null);
    if (!handle) return null;
    const perm = await handle.queryPermission({ mode: 'read' }).catch(() => 'prompt');
    if (perm === 'denied') return null; // user revoked it — treat as nothing remembered
    return { handle, name: handle.name, needsPermission: perm !== 'granted' };
  }

  /**
   * Guest/host picks a local folder via the File System Access API
   * (Chromium only) — or reuses a previously-picked `dirHandle` (from
   * `tryLoadRememberedDirectory()`) so the folder never needs re-picking.
   */
  async loadLibraryFromDirectory(dirHandle) {
    if (!dirHandle) {
      if (!window.showDirectoryPicker) {
        throw new Error('This browser does not support local folder access. Use Chrome or Edge.');
      }
      dirHandle = await window.showDirectoryPicker();
    } else if ((await dirHandle.queryPermission({ mode: 'read' }).catch(() => 'prompt')) !== 'granted') {
      // Re-granting requires the user gesture this method is being called from.
      const perm = await dirHandle.requestPermission({ mode: 'read' });
      if (perm !== 'granted') throw new Error('Folder access was not granted.');
    }
    await _idbSet(DIR_STORE, DIR_KEY, dirHandle).catch(() => {}); // best-effort remember for next time
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

  /**
   * Pre-analyzes the whole library once, at setup time — not live during
   * playback. Perf review 4A. Skips full decode+BPM+beat-grid+peaks analysis
   * for any file whose (name, size, lastModified) was already analyzed in a
   * previous session — the analysis result, not just the folder path, is
   * what "never have to reload tracks again" actually needs, since re-picking
   * the same folder still re-decoded every file before this cache existed.
   */
  async _preAnalyze(fileHandles) {
    const analyzed = [];
    let done = 0;
    for (const { name, handle } of fileHandles) {
      try {
        const file = await handle.getFile();
        const cacheKey = `${name}|${file.size}|${file.lastModified}`;
        const cached = await _idbGet(TRACK_CACHE_STORE, cacheKey).catch(() => null);
        if (cached) {
          analyzed.push({ ...cached, name, handle });
        } else {
          const audioBuffer = await this._decodeAudioFile(file);
          let bpmFallback = false;
          const bpm = await detectBpm(audioBuffer).catch(() => {
            bpmFallback = true; // surfaced in the visual controller so a guessed BPM never looks like a confident reading
            return 120;
          });
          const energy = this._estimateEnergy(audioBuffer);
          const { beatGrid, beatGridBpm } = this._analyzeBeatGrid(audioBuffer, bpm);
          // Real per-track structure (see _analyzeStructure doc) — null when
          // the track is too short or the novelty curve is degenerate;
          // _transitionPlan() falls back to the synthetic energy trajectory
          // in that case. /autoplan 2026-08-04 round 6.
          const structure = this._analyzeStructure(audioBuffer);
          // Waveform peaks for the visual controller — extracted in this same pass
          // over getChannelData() so the decoded AudioBuffer (tens of MB per
          // track) never has to be retained past this loop iteration.
          const peaks = this._extractPeaks(audioBuffer);
          const entry = { bpm, bpmFallback, energy, duration: audioBuffer.duration, beatGrid, beatGridBpm, structure, peaks };
          analyzed.push({ ...entry, name, handle });
          await _idbSet(TRACK_CACHE_STORE, cacheKey, entry).catch(() => {}); // best-effort — cache miss next time just re-analyzes
        }
        // Checked fresh every load, never cached — the offline Demucs batch
        // (/autoplan 2026-08-04 round 7) can complete after a track's
        // analysis was already cached, so baking stem availability into the
        // IndexedDB entry would go stale the moment separation finishes.
        analyzed[analyzed.length - 1].stemsAvailable = await this._checkStemsAvailable(name);
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
      const { bpm, energy, duration, beatGrid, beatGridBpm, structure } = value;
      return { name, bpm, energy, duration, beatGrid, beatGridBpm, structure };
    });
  }

  /** Test seam only — real cache keys are written by _preAnalyze via _idbSet. */
  async _idbSetForExportTest(cacheKey, entry) {
    await _idbSet(TRACK_CACHE_STORE, cacheKey, entry);
  }

  /** Shared decode step for a File — used by both pre-analysis and loop-roll's on-demand re-decode. */
  async _decodeAudioFile(file) {
    const arrayBuffer = await file.arrayBuffer();
    return this.audioCtx.decodeAudioData(arrayBuffer.slice(0));
  }

  /** Same decode step, from a server-hosted URL instead of a local File — used for stems. */
  async _decodeAudioUrl(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Stem fetch failed: ${res.status} ${url}`);
    const arrayBuffer = await res.arrayBuffer();
    return this.audioCtx.decodeAudioData(arrayBuffer);
  }

  /**
   * Whether Demucs stems exist for this track — a HEAD check against the
   * server's /stems route (server/index.js serves the `stems/` output
   * directory read-only). Separation runs offline
   * (`python -m demucs -o stems playlist/*.mp3`), not on this request path.
   * Checked fresh per library load, never cached (see call site). Demucs
   * names the output folder after the source file's basename without
   * extension — same convention `_stemUrl` reconstructs.
   * /autoplan 2026-08-04 round 7.
   */
  async _checkStemsAvailable(trackName) {
    try {
      const res = await fetch(this._stemUrl(trackName, 'drums'), { method: 'HEAD' });
      return res.ok;
    } catch {
      return false; // server unreachable, no stems dir, etc. — same as "not available"
    }
  }

  _stemUrl(trackName, stem) {
    const base = trackName.replace(/\.[^./]+$/, ''); // strip extension — Demucs' own folder-naming convention
    return `/stems/htdemucs/${encodeURIComponent(base)}/${stem}.wav`;
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
   * Fixed-length min/max peak pairs for the visual controller's waveform —
   * WAVEFORM_PEAK_COUNT points regardless of track duration, so array size
   * and render cost never scale with a long file. Uses channel 0 only
   * (visualization, not analysis — mono approximation is fine).
   */
  _extractPeaks(audioBuffer) {
    const data = audioBuffer.getChannelData(0);
    const blockSize = Math.max(1, Math.floor(data.length / WAVEFORM_PEAK_COUNT));
    const peaks = new Float32Array(WAVEFORM_PEAK_COUNT * 2);
    for (let i = 0; i < WAVEFORM_PEAK_COUNT; i++) {
      const start = i * blockSize;
      const end = Math.min(start + blockSize, data.length);
      let min = 0;
      let max = 0;
      for (let j = start; j < end; j++) {
        const v = data[j];
        if (v < min) min = v;
        if (v > max) max = v;
      }
      peaks[i * 2] = min;
      peaks[i * 2 + 1] = max;
    }
    return peaks;
  }

  /**
   * Real per-track structural segmentation (see the STRUCTURE_* constants'
   * doc comment for the honesty/scope caveats). Foote-style: coarse 3-band
   * energy features per frame → averaged into texture windows → cosine
   * self-similarity matrix → checkerboard-kernel novelty curve along the
   * diagonal → adaptive-threshold peak-picking (same technique
   * _detectOnsets already uses) → segment boundaries, each segment given a
   * real (not synthetic) RMS energy level. Returns null on anything
   * degenerate — too short, too few or too many detected boundaries —
   * rather than guessing; callers fall back to the synthetic energy
   * trajectory in that case.
   */
  _analyzeStructure(audioBuffer) {
    const sampleRate = audioBuffer.sampleRate;
    const ch0 = audioBuffer.getChannelData(0);
    const ch1 = audioBuffer.numberOfChannels > 1 ? audioBuffer.getChannelData(1) : null;
    const len = ch0.length;
    const mono = new Float32Array(len);
    for (let i = 0; i < len; i++) mono[i] = ch1 ? (ch0[i] + ch1[i]) / 2 : ch0[i];

    // Two cascaded one-pole lowpass filters split the signal into 3 crude
    // bands (low/mid/high) — no FFT, single pass each, same cost order as
    // the RMS loop in _estimateEnergy.
    const lowpass = (cutoffHz) => {
      const rc = 1 / (2 * Math.PI * cutoffHz);
      const dt = 1 / sampleRate;
      const alpha = dt / (rc + dt);
      const out = new Float32Array(len);
      let prev = 0;
      for (let i = 0; i < len; i++) {
        prev = prev + alpha * (mono[i] - prev);
        out[i] = prev;
      }
      return out;
    };
    const lpLow = lowpass(STRUCTURE_LOW_CUTOFF_HZ);
    const lpMid = lowpass(STRUCTURE_HIGH_CUTOFF_HZ);

    const frameSize = 1024;
    const hop = 512;
    const frameFeatures = [];
    for (let start = 0; start + frameSize <= len; start += hop) {
      let lowSum = 0;
      let midSum = 0;
      let highSum = 0;
      for (let i = 0; i < frameSize; i++) {
        const idx = start + i;
        const low = lpLow[idx];
        const mid = lpMid[idx] - lpLow[idx];
        const high = mono[idx] - lpMid[idx];
        lowSum += low * low;
        midSum += mid * mid;
        highSum += high * high;
      }
      const low = Math.sqrt(lowSum / frameSize);
      const mid = Math.sqrt(midSum / frameSize);
      const high = Math.sqrt(highSum / frameSize);
      const mag = Math.sqrt(low * low + mid * mid + high * high) || 1;
      frameFeatures.push([low / mag, mid / mag, high / mag]);
    }
    if (frameFeatures.length < 8) return null; // far too short to say anything about structure

    // Aggregate frames into texture windows for a tractable similarity matrix.
    const framesPerWindow = Math.max(1, Math.round((STRUCTURE_WINDOW_SEC * sampleRate) / hop));
    const windows = [];
    for (let w = 0; w * framesPerWindow < frameFeatures.length && windows.length < STRUCTURE_MAX_WINDOWS; w++) {
      const start = w * framesPerWindow;
      const end = Math.min(frameFeatures.length, start + framesPerWindow);
      let l = 0;
      let m = 0;
      let h = 0;
      for (let i = start; i < end; i++) {
        l += frameFeatures[i][0];
        m += frameFeatures[i][1];
        h += frameFeatures[i][2];
      }
      const count = end - start;
      // Re-normalize after averaging — averaging unit vectors doesn't yield
      // a unit vector, and the similarity matrix below needs true cosine
      // similarity (dot product of unit vectors), not a magnitude-biased
      // dot product of whatever length the average happened to land at.
      const avgMag = Math.sqrt((l / count) ** 2 + (m / count) ** 2 + (h / count) ** 2) || 1;
      windows.push([l / count / avgMag, m / count / avgMag, h / count / avgMag]);
    }
    if (windows.length < STRUCTURE_MIN_WINDOWS) return null; // too short/degenerate for a meaningful similarity matrix

    // Cosine self-similarity matrix — both vectors are unit-normalized, so
    // the dot product IS the cosine similarity.
    const n = windows.length;
    const sim = new Float32Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        const a = windows[i];
        const b = windows[j];
        const dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        sim[i * n + j] = dot;
        sim[j * n + i] = dot;
      }
    }

    // Checkerboard-kernel novelty curve along the similarity matrix diagonal.
    const K = STRUCTURE_KERNEL_HALF;
    const novelty = new Float32Array(n);
    for (let d = 0; d < n; d++) {
      let score = 0;
      for (let i = -K; i < K; i++) {
        const wi = d + i;
        if (wi < 0 || wi >= n) continue;
        for (let j = -K; j < K; j++) {
          const wj = d + j;
          if (wj < 0 || wj >= n) continue;
          const sameQuadrant = (i < 0) === (j < 0);
          score += sim[wi * n + wj] * (sameQuadrant ? 1 : -1);
        }
      }
      novelty[d] = score;
    }

    // Adaptive local-mean peak-picking — same technique as _detectOnsets.
    const localWindow = STRUCTURE_NOVELTY_LOCAL_WINDOW;
    const boundaries = [];
    for (let d = 0; d < n; d++) {
      const start = Math.max(0, d - localWindow);
      const end = Math.min(n, d + localWindow);
      let localSum = 0;
      for (let k = start; k < end; k++) localSum += novelty[k];
      const localMean = localSum / (end - start);
      if (novelty[d] > localMean * STRUCTURE_NOVELTY_THRESHOLD_MULT && novelty[d] > 0) {
        const t = (d * framesPerWindow * hop) / sampleRate;
        const last = boundaries[boundaries.length - 1];
        if (last === undefined || t - last > STRUCTURE_MIN_BOUNDARY_GAP_SEC) boundaries.push(t);
      }
    }
    if (boundaries.length < STRUCTURE_MIN_BOUNDARIES || boundaries.length > STRUCTURE_MAX_BOUNDARIES) return null;

    // Segments from boundaries + track duration, each with a real (not
    // synthetic) RMS energy scalar, normalized to 0..1 within this track.
    const duration = len / sampleRate;
    const bounds = [0, ...boundaries, duration];
    const segments = [];
    for (let i = 0; i < bounds.length - 1; i++) {
      const segStart = bounds[i];
      const segEnd = bounds[i + 1];
      const startSample = Math.floor(segStart * sampleRate);
      const endSample = Math.min(len, Math.floor(segEnd * sampleRate));
      let sumSquares = 0;
      let count = 0;
      for (let i2 = startSample; i2 < endSample; i2 += 100) {
        sumSquares += mono[i2] * mono[i2];
        count += 1;
      }
      segments.push({ start: segStart, end: segEnd, energy: count > 0 ? Math.sqrt(sumSquares / count) : 0 });
    }
    const energies = segments.map((s) => s.energy);
    const min = Math.min(...energies);
    const max = Math.max(...energies);
    const range = max - min || 1;
    for (const seg of segments) {
      seg.energy = (seg.energy - min) / range;
      seg.level = seg.energy >= 0.66 ? 'high' : seg.energy <= 0.33 ? 'low' : 'mid';
    }

    return { boundaries, segments };
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

    const onsets = [];
    for (let i = 0; i < novelty.length; i++) {
      const start = Math.max(0, i - ONSET_LOCAL_WINDOW_FRAMES);
      const end = Math.min(novelty.length, i + ONSET_LOCAL_WINDOW_FRAMES);
      let localSum = 0;
      for (let k = start; k < end; k++) localSum += novelty[k];
      const localMean = localSum / (end - start);
      if (novelty[i] > localMean * ONSET_THRESHOLD_MULT && novelty[i] > ONSET_MIN_NOVELTY) {
        const t = (i * hop) / sampleRate;
        const last = onsets[onsets.length - 1];
        if (last === undefined || t - last > ONSET_MIN_GAP_SEC) onsets.push(t);
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
    return this._energyTrajectory().value;
  }

  /** Same sine curve as _energyTarget, plus its direction — rising toward a peak or falling toward a valley. */
  _energyTrajectory() {
    const elapsed = Date.now() - (this.setStartedAt || Date.now());
    const phase = (2 * Math.PI * elapsed) / ENERGY_CYCLE_MS;
    return { value: 0.6 + 0.3 * Math.sin(phase), rising: Math.cos(phase) > 0 };
  }

  /**
   * Joint decision for transition duration, EQ duck depth, FX intensity,
   * and the loop-roll occasion gate's "near a peak" reading — one function,
   * one signal, instead of three independent _energyTrajectory() calls
   * that could each land in a different context. Uses the outgoing track's
   * real structure (current segment at `positionSec`) when
   * `_analyzeStructure` produced one; falls back to the synthetic
   * set-level trajectory otherwise — both sources shape-compatible
   * ({value, rising}), so downstream logic doesn't care which it got.
   * /autoplan 2026-08-04 round 6.
   */
  _transitionPlan(outgoingTrack, positionSec) {
    let value;
    let rising;
    if (outgoingTrack && outgoingTrack.structure && outgoingTrack.structure.segments.length > 0) {
      const segs = outgoingTrack.structure.segments;
      const seg = segs.find((s) => positionSec >= s.start && positionSec < s.end) || segs[segs.length - 1];
      const idx = segs.indexOf(seg);
      const next = segs[idx + 1];
      value = seg.energy;
      rising = next ? next.energy > seg.energy : false;
    } else {
      ({ value, rising } = this._energyTrajectory());
    }

    let transitionMs = BASE_BLEND_SEC * 1000;
    let duckDb = EQ_MIN_DB;
    let fxIntensity = 0.8; // moderate default — flat/declining energy, no strong signal either way
    if (value >= PEAK_ENERGY_THRESHOLD) {
      transitionMs = PEAK_BLEND_SEC * 1000; // near a peak — quick, punchy cut
      fxIntensity = 0.6; // a fast cut already reads as decisive — full-depth sweep/echo would just be noise on top
    } else if (value <= VALLEY_ENERGY_THRESHOLD) {
      transitionMs = VALLEY_BLEND_SEC * 1000; // in a valley — long, gentle blend
      duckDb = EQ_MIN_DB * 0.5; // lighter duck, let both basslines breathe together
      fxIntensity = 1; // full FX depth over a long, gentle blend — there's time for it to be heard, not just noticed
    } else if (rising) {
      transitionMs = RISING_BLEND_SEC * 1000; // building toward a peak — a middle-length blend
      fxIntensity = 0.85;
    }

    const nearPeak = rising && value >= PEAK_ENERGY_THRESHOLD - NEAR_PEAK_MARGIN;
    return { transitionMs, duckDb, fxIntensity, nearPeak };
  }

  /**
   * Rules-based pick: closest to the current energy target, penalized by BPM
   * distance from the current track, penalized further for resembling recent
   * picks (novelty). When creativeFlags.sampling is on, weighted-samples from
   * the top-K scoring candidates instead of always taking the single best —
   * same rule set, controlled unpredictability within it. Set
   * creativeFlags.sampling/.novelty to false to fall back to pure argmax.
   */
  _pickNextTrack() {
    if (this.queue.length > 0) {
      const picked = this.queue.shift(); // guest-approved requests take the next gap first
      this._recordPick(picked);
      return picked;
    }

    if (this.library.length === 0) return null;

    let candidates = this.library.filter((t) => !this.played.has(t.name));
    if (candidates.length === 0) {
      // Library exhausted — reset rather than stopping playback dead. Failure-mode T4.
      // Keep the currently-playing track excluded from this reset cycle — otherwise
      // the picker can hand back what's playing right now as the "next" track. Found
      // by /review adversarial pass, 2026-08-02 (top-K sampling made this more likely
      // than it was under pure argmax).
      this.played.clear();
      if (this.current) this.played.add(this.current.name);
      candidates = this.library.filter((t) => !this.played.has(t.name));
      if (candidates.length === 0) {
        // Single-track library — nothing else exists to hand back but the current track.
        this.played.clear();
        candidates = this.library;
      }
    }

    const target = this._energyTarget();
    const scored = candidates.map((track) => {
      const energyDelta = Math.abs(track.energy - target);
      const bpmDelta = this.current ? Math.abs(track.bpm - this.current.bpm) : 0;
      let score = -energyDelta - bpmDelta * BPM_PENALTY_WEIGHT;
      if (this.creativeFlags.novelty) score -= this._noveltyPenalty(track) * NOVELTY_WEIGHT;
      return { track, score };
    });
    scored.sort((a, b) => b.score - a.score);

    const picked = this.creativeFlags.sampling
      ? this._weightedSample(scored.slice(0, Math.min(TOP_K, scored.length)))
      : scored[0].track;
    this._recordPick(picked);
    return picked;
  }

  /** Penalizes a track that closely resembles recent picks (energy + BPM proximity) — keeps a set from looping the same "safe" candidates. No artist field exists in this library, so no artist-similarity term. */
  _noveltyPenalty(track) {
    if (this._recentHistory.length === 0) return 0;
    let penalty = 0;
    for (const past of this._recentHistory) {
      const energyClosness = Math.max(0, 1 - Math.abs(track.energy - past.energy));
      const bpmCloseness = Math.max(0, 1 - Math.abs(track.bpm - past.bpm) / 20);
      penalty += energyClosness + bpmCloseness * 0.5;
    }
    return penalty / this._recentHistory.length;
  }

  _recordPick(track) {
    this._recentHistory.push({ energy: track.energy, bpm: track.bpm });
    if (this._recentHistory.length > NOVELTY_WINDOW) this._recentHistory.shift();
  }

  /** Softmax-weighted sample from a pre-sorted (descending score) top-K list, using the engine's seeded RNG. */
  _weightedSample(scoredTopK) {
    const maxScore = scoredTopK[0].score;
    const weights = scoredTopK.map(({ score }) => Math.exp((score - maxScore) / SAMPLE_TEMPERATURE));
    const total = weights.reduce((sum, w) => sum + w, 0);
    let r = this._rng() * total;
    for (let i = 0; i < scoredTopK.length; i++) {
      r -= weights[i];
      if (r <= 0) {
        if (i > 0) {
          console.debug(
            `[TEMPO creative] sampled rank ${i + 1}/${scoredTopK.length} "${scoredTopK[i].track.name}" over the top pick "${scoredTopK[0].track.name}"`
          );
        }
        return scoredTopK[i].track;
      }
    }
    return scoredTopK[0].track; // floating-point fallback, should not normally hit
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

    this._transitionCount += 1; // occasion-gate spacing clock — counts attempted transitions, not just successful showy firings
    const track = this._pickNextTrack();
    if (!track) return; // empty library — nothing to do, already surfaced via loadLibraryFromDirectory errors

    try {
      const file = await track.handle.getFile();
      const url = URL.createObjectURL(file);
      const incoming = this.standbyPlayer;
      incoming.audio.src = url;
      // Set as soon as the audio element's timeline switches to this track
      // (not after the crossfade completes) — the visual controller reads
      // audio.currentTime directly every frame, so a track/audio mismatch
      // during the multi-second load+fade window would show the new
      // playhead position over the OLD track's waveform/beat-grid/BPM.
      // Found by /review adversarial pass, 2026-08-03.
      incoming.track = track;
      incoming.cuePoint = null; // a new track on this deck invalidates any cue point from the previous one

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

      const outgoing = this.activePlayer;

      // Joint transition plan — duration, EQ duck, and FX intensity all read
      // from ONE signal (real per-track structure when available, else the
      // synthetic set-level trajectory) instead of three independent energy
      // reads that could disagree. Evaluated fresh per transition, right
      // after the pick (autoplan Eng decision: sampled once per
      // transition-decision, not read live at fade time). /autoplan
      // 2026-08-04 round 6.
      // fxIntensity/nearPeak stay real even if transitionVariety (duration
      // variety specifically) is flagged off — those feed the independently
      // flagged `fx`/`loopRoll` mechanisms, and turning one flag off must
      // not silently mute a mechanism gated by a different flag.
      const plan = this._transitionPlan(this.current, outgoing.audio.currentTime);
      if (!this.creativeFlags.transitionVariety) {
        plan.transitionMs = BASE_BLEND_SEC * 1000;
        plan.duckDb = EQ_MIN_DB;
      }

      // Loop-roll ("echo-stutter") — happens BEFORE the crossfade starts, on
      // the outgoing deck's existing chain. Runs to completion (or is cut
      // short by a manual grab) before _crossfade ever begins, so the timer
      // armed at the end of THIS call is computed from when the incoming
      // track actually starts playing — no drift correction needed for the
      // transition-after-next, since nothing about its scheduling assumes
      // the loop didn't happen.
      await this._maybeLoopRoll(outgoing, this.current, plan);
      if (this.manualArmed) {
        this._deferredAutoTransition = true;
        return;
      }

      // Marked played only once we're committed to the crossfade actually
      // happening — the two manualArmed aborts above must not mark a track
      // "played" that never actually reached the audience, or it silently
      // becomes ineligible until the next library-exhaustion reset. Found by
      // /review, 2026-08-05.
      this.played.add(track.name);
      await this._crossfade(outgoing, incoming, this.current, track, plan.transitionMs, plan.duckDb, plan.fxIntensity);
      this._liveDeckId = incoming.id;
      this.current = track;
      this.onNowPlaying(track);

      // Schedule the next pick to start crossfading at a phrase boundary near
      // the end of this track when we have a beat grid, else a fixed offset.
      // The actual duration for THAT transition is only decided at the start
      // of its own _playNext call (same context-freshness reasoning as
      // above) — so the buffer reserved here must cover the WORST-CASE
      // (valley, longest) duration, not the base constant, or a longer
      // transition can outrun the outgoing track's remaining audio and cut
      // to silence mid-fade instead of completing the blend. Found by
      // /review adversarial pass, 2026-08-02.
      const maxReserveSec = VALLEY_BLEND_SEC + 0.5; // worst case (longest possible) blend duration
      const fallbackStart = Math.max(0, track.duration - maxReserveSec);
      // Search window is anchored to the (now much larger) reserve deadline,
      // not a raw "20 seconds before the file ends" — that was correct at
      // the old ~5.6s reserve but silently never found a boundary early
      // enough once the reserve grew to ~91s, permanently falling back to
      // the flat, unaligned mix-out point. Found by /autoplan round 4.
      const phraseStart = this._phraseBoundaryAfter(
        track,
        Math.max(0, track.duration - maxReserveSec - PHRASE_LOOKBACK_PAD_SEC)
      );
      const mixOutAt =
        phraseStart !== null && phraseStart < track.duration - maxReserveSec ? phraseStart : fallbackStart;
      // mixOutAt is an ABSOLUTE position in the incoming track's own timeline,
      // but setTimeout needs a delay from right NOW — and the incoming track's
      // playhead is already well past 0 here: tempo-sync seeks it to its first
      // beat onset before the crossfade starts, then it plays through the
      // whole (now variable, up to 90s) transition while the fade completes.
      // Scheduling `mixOutAt * 1000` directly ignored that elapsed head start,
      // so every automatic transition fired late by roughly one crossfade's
      // worth of seconds — for a valley (long) transition, late enough to
      // land at or past the track's own end, reproducing the exact
      // "cut to silence" bug the worst-case reserve above was built to
      // prevent, just via a different path it never covered. Found by /ship
      // adversarial pass, 2026-08-05.
      const msUntilNext = Math.max(0, (mixOutAt - incoming.audio.currentTime) * 1000);
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

  /**
   * Occasion gate for "showy" techniques (loop-roll today; fx could adopt
   * the same guard later) — /autoplan 2026-08-04 round 3. Replaces "fires
   * whenever eligible" with an approximation of a human DJ's judgment about
   * WHEN to use a flashy move: only near a rising energy peak, spaced apart
   * from the last showy firing, and never the identical move twice in a
   * row. `nearPeak` comes from `_transitionPlan()` — the outgoing track's
   * real structure when available, else the synthetic set-level trajectory
   * (round 6) — computed once per transition-decision and passed in, not
   * read live here, so this stays in sync with whatever duration/FX
   * decision was made for the same transition. Spacing is measured in
   * transitions (`_transitionCount`), not wall-clock, so a long
   * manual-mixer excursion can't silently suppress the next showy moment
   * once auto-pilot resumes.
   */
  _shouldLoopRoll(nearPeak) {
    const spaced = this._transitionCount - this._lastShowyAt >= MIN_SHOWY_SPACING;
    const notRepeat = this._lastShowyTechnique !== 'loopRoll';
    return nearPeak && spaced && notRepeat;
  }

  /**
   * "Echo-stutter" loop-roll (creativeFlags.loopRoll) — repeats the outgoing
   * track's current beat exactly twice via a sample-accurate
   * AudioBufferSourceNode before the crossfade begins. Hard no-ops whenever
   * the beat grid isn't trustworthy, independent of the flag (automatic
   * correctness guard, not just the human kill switch) — this is
   * deliberately narrower than a general variable-length loop engine (see
   * the /autoplan D3 gate decision, 2026-08-04): anchored only to
   * `beatGrid[0]` + median beat-interval math, never a raw interior onset,
   * so a single bad onset read can't put the loop boundary mid-transient.
   *
   * When `creativeFlags.stems` is on and Demucs stems exist for this track
   * (round 7), loops the isolated drums stem instead of the full mixdown —
   * a cleaner, punchier hit with no vocal/bass bleed. Demucs preserves the
   * source track's exact duration/timing, so the existing beat-grid math
   * (computed from the original mixdown) applies unchanged to the stem —
   * only the decoded audio SOURCE changes, not the loop-boundary logic, the
   * sync model, or the handoff back to `outgoing.audio` afterward. Falls
   * back to looping the full mixdown (today's behavior) whenever stems
   * aren't available or the flag is off — same automatic-fallback
   * discipline as everywhere else in this file.
   */
  async _maybeLoopRoll(outgoing, outgoingTrack, plan) {
    if (!this.creativeFlags.loopRoll) return;
    if (!this._shouldLoopRoll(plan.nearPeak)) return; // occasion gate — see _shouldLoopRoll doc above
    if (!outgoingTrack || !outgoingTrack.beatGrid || !outgoingTrack.beatGridBpm) return;
    const beatLen = 60 / outgoingTrack.beatGridBpm;
    if (!(beatLen > LOOP_ROLL_MIN_BEAT_SEC && beatLen < LOOP_ROLL_MAX_BEAT_SEC)) return; // degenerate BPM read — skip rather than loop garbage

    const anchor = outgoingTrack.beatGrid[0];
    if (outgoing.audio.currentTime <= anchor) return; // haven't reached the first trustworthy beat yet — cheap early-out before paying for a decode

    const useStemDrums = this.creativeFlags.stems && outgoingTrack.stemsAvailable;
    let audioBuffer;
    try {
      audioBuffer = useStemDrums
        ? await this._decodeAudioUrl(this._stemUrl(outgoingTrack.name, 'drums'))
        : await this._decodeAudioFile(await outgoingTrack.handle.getFile());
    } catch {
      return; // decode failed — skip the stutter, plain crossfade still happens
    }
    // The beat grid was computed from the original mixdown — trusting it
    // against a stem buffer is only valid if Demucs preserved exact
    // duration (it should, but "should" isn't a guarantee to loop garbage
    // on). A mismatch means the stem's sample timeline doesn't line up
    // with beatGrid's timestamps — bail rather than loop a misaligned hit.
    if (useStemDrums && Math.abs(audioBuffer.duration - outgoingTrack.duration) > 0.5) return;
    if (this.manualArmed) return; // host grabbed control while this decoded — abandon the stutter, not just the crossfade

    // Re-sample position now, not before the decode — decodeAudioData can
    // take real wall-clock time on a full track, and outgoing.audio kept
    // playing through it. Using a pre-decode position here would start the
    // loop behind the actual playhead, an audible backward jump. Found by
    // /review adversarial pass, 2026-08-05.
    const pos = outgoing.audio.currentTime;
    if (pos <= anchor) return; // decode took long enough that we're no longer past the first beat — bail rather than loop garbage
    const beatsElapsed = Math.floor((pos - anchor) / beatLen);
    const loopStart = anchor + beatsElapsed * beatLen; // grid-aligned, never a raw interior onset
    if (loopStart + beatLen * 2 >= outgoingTrack.duration) return; // too close to the file's end to loop safely

    // The <audio> element and a buffer source must never sound at once (same
    // assumption _crossfade's own outgoing.audio.pause() relies on).
    outgoing.audio.pause();
    const src = this.audioCtx.createBufferSource();
    src.buffer = audioBuffer;
    src.loop = true;
    src.loopStart = loopStart;
    src.loopEnd = loopStart + beatLen;
    src.connect(outgoing.low); // reuse the deck's existing EQ/gain/echo chain from `low` onward
    outgoing._loopSource = src;
    // Occasion-gate state updates here, not at the _shouldLoopRoll() check —
    // this is the actual point of no return. An earlier bail (degenerate
    // BPM, decode failure, manualArmed mid-decode) must not burn the
    // spacing budget for a stutter that never audibly happened.
    this._lastShowyAt = this._transitionCount;
    this._lastShowyTechnique = 'loopRoll';

    const startedAt = this.audioCtx.currentTime;
    src.start(startedAt, loopStart);
    src.stop(startedAt + beatLen * 2); // exactly two repeats — "echo-stutter," not a general loop engine

    await new Promise((resolve) => {
      src.onended = resolve;
    });
    outgoing._loopSource = null;
    if (this.manualArmed) return; // armManual() already stopped/disconnected src and resumed outgoing.audio — nothing left to do here

    src.disconnect();
    // Resume the <audio> element where the loop left off (one repeat's worth
    // of consumed track-time past the loop start) so the crossfade that
    // follows picks up the track's real timeline, not the looped one.
    outgoing.audio.currentTime = Math.min(loopStart + beatLen, outgoingTrack.duration - RESUME_END_MARGIN_SEC);
    await outgoing.audio.play();
  }

  /**
   * Equal-power crossfade with tempo-sync (when both tracks have a
   * trustworthy beat grid) and a bass-swap EQ duck. `transitionMs`/`duckDb`
   * default to the base constants — callers with creativeFlags.transitionVariety
   * off (or the manual mixer path, which doesn't pass them) get identical
   * behavior to the pre-creative-layer engine.
   */
  async _crossfade(
    outgoing,
    incoming,
    outgoingTrack,
    incomingTrack,
    transitionMs = BASE_BLEND_SEC * 1000,
    duckDb = EQ_MIN_DB,
    fxIntensity = 1
  ) {
    const duration = transitionMs / 1000;
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

    // Re-anchor `start` to AFTER the seek+play() await resolves, not before —
    // both can take real wall-clock time (cold blob URL, seeking within it),
    // and setValueCurveAtTime with a startTime that's already in the past
    // applies the curve partway-through: an audible gain jump instead of a
    // smooth fade-in, finishing early since `duration` is still counted from
    // the stale `start`. Found by /review, 2026-08-05.
    const start = this.audioCtx.currentTime;

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
      outgoing.low.gain.linearRampToValueAtTime(duckDb, start + duration * 0.6);
    }

    // FX layer (creativeFlags.fx): filter sweep — roll the outgoing high band
    // off across the whole transition, reusing the same highshelf gain
    // AudioParam the manual EQ already writes to ("manual wins" contract:
    // only when the host hasn't touched this band). Echo tail — open the
    // wet send during the fade, then explicitly release it (setTargetAtTime
    // decay) just past the transition's end so it can't bleed into the next
    // transition's own sweep/echo. Both scaled by `fxIntensity` (from the
    // joint _transitionPlan) — a fast, punchy peak-cut already reads as
    // decisive on its own, so it gets a lighter touch than a long valley
    // blend, which has time for the full effect to actually be heard.
    // /autoplan 2026-08-04 round 6.
    if (this.creativeFlags.fx) {
      if (outgoing.eq.high === 0) {
        outgoing.high.gain.cancelScheduledValues(now);
        outgoing.high.gain.setValueAtTime(0, start);
        outgoing.high.gain.linearRampToValueAtTime(SWEEP_MIN_DB * fxIntensity, start + duration);
      }
      outgoing.wet.gain.cancelScheduledValues(now);
      outgoing.wet.gain.setValueAtTime(0, start);
      outgoing.wet.gain.linearRampToValueAtTime(ECHO_WET_LEVEL * fxIntensity, start + duration * 0.7);
      outgoing.wet.gain.setTargetAtTime(0, start + duration, ECHO_TAIL_RELEASE_SEC);
    }

    await new Promise((resolve) => setTimeout(resolve, transitionMs));
    if (this.manualArmed) return; // host took over mid-fade — don't force-finalize over their live mix
    outgoing.audio.pause();
    outgoing.audio.playbackRate = 1;
    // Tempo-sync is scoped to "for this one mix" (see the comment above where
    // `rate` is computed) — incoming.audio.playbackRate was never reset back
    // to 1 after the blend, so a tempo-synced track kept playing up to 6%
    // off its natural tempo for its entire runtime as "current" (not just
    // during the transition), throwing off pitch, duration, and the
    // wall-clock-vs-track-position math in _playNext's next-transition
    // scheduling. Found by /ship adversarial pass, 2026-08-05.
    incoming.audio.playbackRate = 1;
    if (outgoing.eq.low === 0) outgoing.low.gain.setValueAtTime(0, this.audioCtx.currentTime);
    if (outgoing.eq.high === 0) outgoing.high.gain.setValueAtTime(0, this.audioCtx.currentTime);
    // Gated on the CURRENT flag value, not "was fx on for this transition" —
    // if the host flipped creativeFlags.fx off partway through (devtools),
    // don't let this transition's own just-scheduled tail-release decay
    // (a few lines up) keep running past a flag that's now off. When fx is
    // still on, leave the natural decay alone — it's supposed to keep
    // sounding briefly after the transition ends. Found by /review
    // adversarial pass, 2026-08-05.
    if (!this.creativeFlags.fx) {
      outgoing.wet.gain.cancelScheduledValues(this.audioCtx.currentTime);
      outgoing.wet.gain.setValueAtTime(0, this.audioCtx.currentTime);
    }
  }

  // ---- Manual mixer console API ----
  // Global arm: touching any control here arms manual mode for the whole
  // session (not per-control) — simpler state machine for a pilot-scale build.

  armManual() {
    if (this.manualArmed) return;
    // Nothing to take control OF yet — the very first _playNext() is still
    // loading its first track. Arming here would abort that load
    // (_deferredAutoTransition=true, both decks stay silent with `current`
    // never set) and strand playback until the host notices and taps "Back
    // to Auto." Refuse instead: let the first track finish loading, arming
    // works normally from then on. Found by /ship adversarial pass,
    // 2026-08-05.
    if (!this.current) return;
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
      // The echo send has no manual control (unlike gain/EQ, which the host
      // can pick up and continue from wherever they were) — freezing it at
      // whatever level was mid-decay would leave a constant, uncontrollable
      // echo bleeding into the rest of the manual session. Silence it
      // instead. Found by /review adversarial pass, 2026-08-05.
      deck.wet.gain.cancelScheduledValues(now);
      deck.wet.gain.setValueAtTime(0, now);
      // A live manual grab always wins over an in-flight loop-roll — stop it
      // immediately and hand the deck back to its normal <audio> playback
      // rather than let a stuck loop keep sounding under a live manual mix.
      if (deck._loopSource) {
        try {
          deck._loopSource.stop();
        } catch {
          /* already stopped/ended — fine */
        }
        deck._loopSource.disconnect();
        deck._loopSource = null;
        deck.audio.play().catch(() => {});
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
      deck.wet.gain.cancelScheduledValues(now);
      deck.wet.gain.setValueAtTime(0, now); // silence any leftover echo tail before auto-pilot resumes
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
    // Clamping to the 0.5 midpoint (as this used to) is itself a bug: at
    // x=0.5 both decks sit at ~0.71 gain (equal-power midpoint), a ~3dB dip
    // on the one deck that's actually live, not the "stay put" behavior the
    // comment above promises. Clamp all the way back to the live deck's full
    // gain instead. Found by /ship adversarial pass, 2026-08-05.
    if (x >= 0.5 && !this.playerB.track) x = 0; // no B to fade to — stay fully on A
    if (x < 0.5 && !this.playerA.track) x = 1; // no A to fade to — stay fully on B
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
