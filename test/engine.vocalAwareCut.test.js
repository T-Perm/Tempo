// Covers the vocal-aware mix-out point mechanism: the crossfade point search
// now prefers a phrase boundary that lands in an instrumental gap on the
// outgoing track's vocals stem, instead of always taking the earliest
// boundary in range regardless of whether a lyric is playing there.
import { describe, it, expect, vi } from 'vitest';
import { MusicEngine } from '../public/host/engine.js';

function makeEngine() {
  return new MusicEngine({});
}

/** Synthetic mono buffer, loud (vocal-like) during [loudStart, loudEnd), silent elsewhere. */
function vocalLikeBuffer({ durationSec, loudStart, loudEnd, sampleRate = 44100 }) {
  const length = Math.round(durationSec * sampleRate);
  const data = new Float32Array(length);
  const loudStartSample = Math.round(loudStart * sampleRate);
  const loudEndSample = Math.round(loudEnd * sampleRate);
  for (let i = loudStartSample; i < loudEndSample && i < length; i++) {
    data[i] = 0.9 * Math.sin((2 * Math.PI * 300 * i) / sampleRate);
  }
  return {
    sampleRate,
    numberOfChannels: 1,
    duration: length / sampleRate,
    getChannelData: () => data,
  };
}

describe('_analyzeVocalPresence', () => {
  it('produces a normalized envelope that is low during silence and high during the loud region', async () => {
    const engine = makeEngine();
    const buffer = vocalLikeBuffer({ durationSec: 10, loudStart: 4, loudEnd: 6 });
    engine._decodeAudioUrl = vi.fn(async () => buffer);

    const presence = await engine._analyzeVocalPresence({ name: 'track.mp3' });

    expect(presence.hop).toBe(1); // VOCAL_PRESENCE_WINDOW_SEC
    expect(presence.levels.length).toBe(10);
    expect(presence.levels[0]).toBeLessThan(0.1); // silent window
    expect(presence.levels[4]).toBeGreaterThan(0.5); // inside the loud region
    expect(Math.max(...presence.levels)).toBeCloseTo(1, 5); // normalized against its own peak
  });
});

describe('_bestPhraseBoundary', () => {
  it('falls back to the plain earliest phrase boundary when no vocal signal is available', () => {
    const engine = makeEngine();
    const track = { beatGrid: [0], beatGridBpm: 120 }; // phraseLen = 60/120 * 32 = 16s
    const result = engine._bestPhraseBoundary(track, 10, 50, null);
    expect(result).toBe(engine._phraseBoundaryAfter(track, 10));
  });

  it('prefers a later phrase boundary that lands in a vocal gap over the earliest one', () => {
    const engine = makeEngine();
    const track = { beatGrid: [0], beatGridBpm: 120 }; // phraseLen = 16s, boundaries at 0,16,32,48...
    // Search window [10, 50) contains boundaries at 16, 32, 48.
    const vocalPresence = {
      hop: 1,
      levels: new Array(60).fill(0.1),
    };
    // Make the boundary at 16 land mid-lyric (high energy), 32 land in a gap (low energy).
    for (let t = 14; t <= 18; t++) vocalPresence.levels[t] = 0.9;
    for (let t = 30; t <= 34; t++) vocalPresence.levels[t] = 0.05;

    const result = engine._bestPhraseBoundary(track, 10, 50, vocalPresence);

    expect(result).toBe(32); // not 16, even though 16 is the earliest boundary in range
  });

  it('returns null when the track has no beat grid, same as _phraseBoundaryAfter', () => {
    const engine = makeEngine();
    const track = { beatGrid: null, beatGridBpm: null };
    expect(engine._bestPhraseBoundary(track, 10, 50, { hop: 1, levels: [] })).toBeNull();
  });
});
