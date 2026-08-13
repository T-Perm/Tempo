import { describe, it, expect, beforeEach, vi } from 'vitest';
import { MusicEngine } from '../public/host/engine.js';

function makeEngine() {
  return new MusicEngine({});
}

/** Synthetic mono AudioBuffer: silence except for short 440Hz bursts every
 * `intervalSec`, each burst exactly one onset-detector frame (1024 samples)
 * wide so it registers as a single, unambiguous energy spike. */
function burstBuffer({ intervalSec, burstCount, sampleRate = 44100 }) {
  const intervalSamples = Math.round(intervalSec * sampleRate);
  const length = intervalSamples * burstCount + 2048;
  const data = new Float32Array(length);
  for (let b = 0; b < burstCount; b++) {
    const start = b * intervalSamples;
    for (let i = 0; i < 1024 && start + i < length; i++) {
      data[start + i] = 0.95 * Math.sin((2 * Math.PI * 440 * i) / sampleRate);
    }
  }
  return {
    sampleRate,
    numberOfChannels: 1,
    duration: length / sampleRate,
    getChannelData: () => data,
  };
}

describe('_detectOnsets / _analyzeBeatGrid', () => {
  let engine;
  beforeEach(() => {
    engine = makeEngine();
  });

  it('returns null for a degenerate (too few onsets) buffer', () => {
    const silent = { sampleRate: 44100, numberOfChannels: 1, getChannelData: () => new Float32Array(44100) };
    const { beatGrid, beatGridBpm } = engine._analyzeBeatGrid(silent, 120);
    expect(beatGrid).toBeNull();
    expect(beatGridBpm).toBeNull();
  });

  it('detects a beat grid from regularly spaced bursts and matches the expected BPM', () => {
    // 0.5s spacing = 120 BPM.
    const buf = burstBuffer({ intervalSec: 0.5, burstCount: 9 });
    const { beatGrid, beatGridBpm } = engine._analyzeBeatGrid(buf, 120);
    expect(beatGrid).not.toBeNull();
    expect(beatGrid.length).toBeGreaterThanOrEqual(4);
    expect(beatGridBpm).toBeGreaterThan(110);
    expect(beatGridBpm).toBeLessThan(130);
  });

  it('halves a double-time read back into the sane 70-180 BPM range', () => {
    // 0.25s spacing = raw 240 BPM, which _analyzeBeatGrid's octave-correction
    // must halve to ~120 before it ever reaches the sanity check.
    const buf = burstBuffer({ intervalSec: 0.25, burstCount: 12 });
    const { beatGridBpm } = engine._analyzeBeatGrid(buf, 120);
    expect(beatGridBpm).not.toBeNull();
    expect(beatGridBpm).toBeGreaterThan(70);
    expect(beatGridBpm).toBeLessThan(180);
  });

  it('rejects a grid whose BPM disagrees wildly with the independent average-BPM detector', () => {
    const buf = burstBuffer({ intervalSec: 0.5, burstCount: 9 }); // ~120 BPM grid
    const { beatGrid, beatGridBpm } = engine._analyzeBeatGrid(buf, 40); // ratio ~3, way past the 1.8 cap
    expect(beatGrid).toBeNull();
    expect(beatGridBpm).toBeNull();
  });

  it('accepts a grid whose BPM is close to the average-BPM detector', () => {
    const buf = burstBuffer({ intervalSec: 0.5, burstCount: 9 }); // ~120 BPM grid
    const { beatGrid } = engine._analyzeBeatGrid(buf, 125); // close enough
    expect(beatGrid).not.toBeNull();
  });
});

describe('_pickNextTrack — candidate-exhaustion fallback', () => {
  let engine;
  beforeEach(() => {
    engine = makeEngine();
    engine.creativeFlags.sampling = false; // deterministic argmax for these tests
    engine.creativeFlags.novelty = false;
  });

  it('does not hand back the currently-playing track when the rest of the library is exhausted', async () => {
    engine.library = [
      { name: 'a.mp3', energy: 0.5, bpm: 120 },
      { name: 'b.mp3', energy: 0.5, bpm: 120 },
    ];
    engine.current = engine.library[0];
    engine.played = new Set(['a.mp3', 'b.mp3']); // both already played, current = a.mp3
    const picked = await engine._pickNextTrack();
    expect(picked.name).toBe('b.mp3'); // must NOT be 'a.mp3' — the regression this guards against
  });

  it('falls back to the current track only when it is the sole library entry', async () => {
    engine.library = [{ name: 'only.mp3', energy: 0.5, bpm: 120 }];
    engine.current = engine.library[0];
    engine.played = new Set(['only.mp3']);
    const picked = await engine._pickNextTrack();
    expect(picked.name).toBe('only.mp3'); // nothing else exists to hand back
  });

  it('argmax picks the candidate closest to the current energy target when sampling is off', async () => {
    engine.library = [
      { name: 'far.mp3', energy: 0.1, bpm: 120 },
      { name: 'near.mp3', energy: 0.59, bpm: 120 },
    ];
    engine.current = null;
    engine.played = new Set();
    // Force a known energy target instead of depending on wall-clock timing.
    engine._energyTarget = () => 0.6;
    const picked = await engine._pickNextTrack();
    expect(picked.name).toBe('near.mp3');
  });
});

describe('enqueueRequestedTrack — word-boundary match', () => {
  let engine;
  beforeEach(() => {
    engine = makeEngine();
    engine.library = [
      { name: 'One Dance.mp3' },
      { name: 'One Dance (Remix).mp3' },
      { name: 'Runaway.mp3' },
    ];
  });

  it('queues the single unambiguous match', () => {
    engine.enqueueRequestedTrack('Runaway');
    expect(engine.queue).toHaveLength(1);
    expect(engine.queue[0].name).toBe('Runaway.mp3');
  });

  it('silently skips when the request is ambiguous (2+ matches)', () => {
    engine.enqueueRequestedTrack('One Dance');
    expect(engine.queue).toHaveLength(0);
  });

  it('silently skips when nothing matches', () => {
    engine.enqueueRequestedTrack('Nonexistent Song');
    expect(engine.queue).toHaveLength(0);
  });
});

describe('_weightedSample / _noveltyPenalty', () => {
  it('is deterministic for a fixed RNG seed', () => {
    const e1 = makeEngine();
    const e2 = makeEngine();
    const seed = 12345;
    // Re-seed both engines identically — the constructor seeds from Date.now(),
    // which isn't reproducible across two constructions in the same test.
    const mulberry32 = (seed) => {
      let state = seed >>> 0;
      return function () {
        state = (state + 0x6d2b79f5) | 0;
        let t = Math.imul(state ^ (state >>> 15), 1 | state);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
      };
    };
    e1._rng = mulberry32(seed);
    e2._rng = mulberry32(seed);

    const scoredTopK = [
      { track: { name: 'a' }, score: 0.9 },
      { track: { name: 'b' }, score: 0.5 },
      { track: { name: 'c' }, score: 0.1 },
    ];
    const pick1 = e1._weightedSample(scoredTopK);
    const pick2 = e2._weightedSample(scoredTopK);
    expect(pick1.name).toBe(pick2.name);
  });

  it('penalizes a track resembling recent picks over a novel one', () => {
    const engine = makeEngine();
    engine._recentHistory = [{ energy: 0.6, bpm: 120 }, { energy: 0.61, bpm: 121 }];
    const similar = { name: 'similar', energy: 0.6, bpm: 120 };
    const novel = { name: 'novel', energy: 0.1, bpm: 90 };
    expect(engine._noveltyPenalty(similar)).toBeGreaterThan(engine._noveltyPenalty(novel));
  });
});

describe('_transitionPlan — joint duration/duck/FX/nearPeak decision', () => {
  let engine;
  beforeEach(() => {
    engine = makeEngine();
  });

  it('returns a quick, punchy transition near an energy peak', async () => {
    const track = { structure: { segments: [{ start: 0, end: 100, energy: 0.9 }] } };
    const plan = await engine._transitionPlan(track, 10);
    expect(plan.transitionMs).toBe(20000); // PEAK_BLEND_SEC
    expect(plan.fxIntensity).toBe(0.6);
  });

  it('returns a long, gentle blend in a valley', async () => {
    const track = { structure: { segments: [{ start: 0, end: 100, energy: 0.2 }] } };
    const plan = await engine._transitionPlan(track, 10);
    expect(plan.transitionMs).toBe(90000); // VALLEY_BLEND_SEC
    expect(plan.duckDb).toBe(-9); // EQ_MIN_DB * 0.5, lighter duck
    expect(plan.fxIntensity).toBe(1);
  });

  it('flags nearPeak only when rising and within the margin of the peak threshold', async () => {
    const rising = {
      structure: { segments: [{ start: 0, end: 50, energy: 0.72 }, { start: 50, end: 100, energy: 0.9 }] },
    };
    expect((await engine._transitionPlan(rising, 10)).nearPeak).toBe(true);

    const falling = {
      structure: { segments: [{ start: 0, end: 50, energy: 0.72 }, { start: 50, end: 100, energy: 0.1 }] },
    };
    expect((await engine._transitionPlan(falling, 10)).nearPeak).toBe(false);
  });
});

describe('manual mixer console API', () => {
  let engine;
  beforeEach(() => {
    engine = makeEngine();
    // armManual() refuses to arm before a first track has ever loaded
    // (current === null) — set it so these tests reflect a set already in
    // progress, which is when a host can actually reach the mixer UI.
    engine.current = { name: 'now-playing.mp3' };
  });

  it('clamps setEQ to the -18..0 cut-only range', () => {
    engine.setEQ('A', 'low', -100);
    expect(engine.playerA.eq.low).toBe(-18);
    engine.setEQ('A', 'low', 100);
    expect(engine.playerA.eq.low).toBe(0);
  });

  it('clamps setTempoNudge to +/-1 before scaling by the tempo-stretch cap', () => {
    engine.setTempoNudge('A', 5);
    expect(engine.playerA.audio.playbackRate).toBeCloseTo(1.06);
    engine.setTempoNudge('A', -5);
    expect(engine.playerA.audio.playbackRate).toBeCloseTo(0.94);
  });

  it('setCue sets a cue point on the first tap and jumps to it on the second', () => {
    engine.playerA.audio.currentTime = 42;
    engine.setCue('A');
    expect(engine.playerA.cuePoint).toBe(42);
    engine.playerA.audio.currentTime = 90;
    engine.setCue('A');
    expect(engine.playerA.audio.currentTime).toBe(42);
  });

  it('backToAuto resets manual EQ back to neutral and disarms', () => {
    engine.setEQ('A', 'low', -10);
    engine.backToAuto();
    expect(engine.playerA.eq.low).toBe(0);
    expect(engine.manualArmed).toBe(false);
  });
});

describe('armManual — refuses to arm before a first track has loaded', () => {
  it('does nothing when current is null (still loading the first track)', () => {
    const engine = makeEngine();
    expect(engine.current).toBeNull();
    engine.armManual();
    expect(engine.manualArmed).toBe(false);
  });

  it('arms normally once a track is current', () => {
    const engine = makeEngine();
    engine.current = { name: 'a.mp3' };
    engine.armManual();
    expect(engine.manualArmed).toBe(true);
  });
});

describe('setCrossfader — no-track-loaded clamp stays at full gain', () => {
  it('clamps fully back to deck A (not the 0.5 midpoint) when B has no track', () => {
    const engine = makeEngine();
    engine.playerA.track = { name: 'a.mp3' };
    engine.playerB.track = null;
    engine.setCrossfader(1); // try to fade fully to B
    expect(engine.playerA.gain.gain.value).toBeCloseTo(1);
    expect(engine.playerB.gain.gain.value).toBeCloseTo(0);
  });

  it('clamps fully back to deck B (not the 0.5 midpoint) when A has no track', () => {
    const engine = makeEngine();
    engine.playerA.track = null;
    engine.playerB.track = { name: 'b.mp3' };
    engine.setCrossfader(0); // try to fade fully to A
    expect(engine.playerB.gain.gain.value).toBeCloseTo(1);
    expect(engine.playerA.gain.gain.value).toBeCloseTo(0);
  });
});

describe('setCrossfader — deck-flip notification', () => {
  it('flips the live deck and fires onNowPlaying when the crossfader crosses center', () => {
    const onNowPlaying = vi.fn();
    const engine = new MusicEngine({ onNowPlaying });
    engine.playerA.track = { name: 'a.mp3' };
    engine.playerB.track = { name: 'b.mp3' };
    engine.setCrossfader(0.9); // full toward B
    expect(engine.liveDeckId).toBe('B');
    expect(onNowPlaying).toHaveBeenCalledWith(engine.playerB.track);
  });
});

describe('loadLibraryFromDirectory — permission denial path', () => {
  let engine;
  beforeEach(() => {
    engine = makeEngine();
  });

  // Note: tryLoadRememberedDirectory's own denied-permission branch reads a
  // handle from IndexedDB first, which jsdom doesn't implement — exercising
  // that specific branch would require mocking engine.js's module-internal
  // IndexedDB helpers, out of proportion for this pass. This test covers the
  // other permission-denial path: an already-known handle whose permission
  // has to be re-requested and is refused.
  it('throws when re-requesting permission is refused', async () => {
    const refusedHandle = {
      queryPermission: async () => 'prompt',
      requestPermission: async () => 'denied',
      entries: async function* () {},
    };
    await expect(engine.loadLibraryFromDirectory(refusedHandle)).rejects.toThrow(
      'Folder access was not granted.'
    );
  });
});

describe('_shouldLoopRoll — occasion gate', () => {
  let engine;
  beforeEach(() => {
    engine = makeEngine();
  });

  it('refuses to fire when not near a rising peak', () => {
    expect(engine._shouldLoopRoll(false)).toBe(false);
  });

  it('refuses to fire again too soon after the last showy technique', () => {
    engine._transitionCount = 10;
    engine._lastShowyAt = 9; // only 1 transition ago, below MIN_SHOWY_SPACING (3)
    engine._lastShowyTechnique = null;
    expect(engine._shouldLoopRoll(true)).toBe(false);
  });

  it('fires when near peak, spaced out, and not a repeat of the last technique', () => {
    engine._transitionCount = 10;
    engine._lastShowyAt = 5; // 5 transitions ago, past MIN_SHOWY_SPACING (3)
    engine._lastShowyTechnique = null;
    expect(engine._shouldLoopRoll(true)).toBe(true);
  });

  it('refuses to fire the same showy technique twice in a row', () => {
    engine._transitionCount = 10;
    engine._lastShowyAt = 5;
    engine._lastShowyTechnique = 'loopRoll';
    expect(engine._shouldLoopRoll(true)).toBe(false);
  });
});

describe('_weightedSample — floating-point fallback', () => {
  it('falls back to the top-ranked candidate if the accumulator never crosses zero', () => {
    const engine = makeEngine();
    engine._rng = () => 1; // r = total on the nose — the normal loop exit shouldn't ever fire before exhausting
    const scoredTopK = [
      { track: { name: 'top' }, score: 0.9 },
      { track: { name: 'second' }, score: 0.1 },
    ];
    expect(engine._weightedSample(scoredTopK).name).toBe('top');
  });
});

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
        energy: 0.6,
        duration: 210.5,
        beatGrid: [0.5, 1.0, 1.5],
        beatGridBpm: 128,
        structure: { segments: [{ start: 0, end: 30, energy: 0.4 }] },
        bpmFallback: false,
      },
    ]);
  });
});
