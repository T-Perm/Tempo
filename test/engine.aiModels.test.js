// Covers the AI-model load/inference paths added by the AI mixing models
// plan (docs/superpowers/plans/2026-08-06-ai-mixing-models.md, Task 9/10):
// load failure falls back to deterministic behavior and disables the flag,
// a successful load's raw output gets clamped, and a session.run() failure
// after a successful load also falls back cleanly (distinct code path from
// load failure — /autoplan review 2026-08-11, decision D6).
import { describe, it, expect, beforeEach, vi } from 'vitest';

vi.mock('https://esm.sh/onnxruntime-web@1.19.2', () => ({
  InferenceSession: { create: vi.fn() },
  Tensor: vi.fn(function (type, data, dims) {
    this.type = type;
    this.data = data;
    this.dims = dims;
  }),
  env: { wasm: {} },
}));

import { MusicEngine } from '../public/host/engine.js';
import * as ort from 'https://esm.sh/onnxruntime-web@1.19.2';

function makeEngine() {
  return new MusicEngine({});
}

const selectionMeta = {
  inputOrder: ['candidateEnergy', 'candidateBpm', 'currentEnergy', 'currentBpm', 'energyTarget', 'noveltyPenalty'],
  mean: [0, 0, 0, 0, 0, 0],
  std: [1, 1, 1, 1, 1, 1],
};

const transitionMeta = {
  inputOrder: ['energy', 'rising', 'bpmDelta'],
  mean: [0, 0, 0],
  std: [1, 1, 1],
};

beforeEach(() => {
  vi.restoreAllMocks();
  ort.InferenceSession.create.mockReset();
});

describe('_loadAiModel — load failure fallback', () => {
  it('disables the flag and falls back to deterministic picking when the model file is missing', async () => {
    ort.InferenceSession.create.mockRejectedValue(new Error('404 not found'));
    vi.stubGlobal('fetch', vi.fn(async () => ({ json: async () => selectionMeta })));

    const engine = makeEngine();
    engine.creativeFlags.aiSelection = true;
    engine.creativeFlags.sampling = false;
    engine.creativeFlags.novelty = false;
    engine.library = [
      { name: 'far.mp3', energy: 0.1, bpm: 120 },
      { name: 'near.mp3', energy: 0.59, bpm: 120 },
    ];
    engine.current = null;
    engine.played = new Set();
    engine._energyTarget = () => 0.6;

    const picked = await engine._pickNextTrack();

    expect(picked.name).toBe('near.mp3'); // deterministic argmax still ran
    expect(engine.creativeFlags.aiSelection).toBe(false); // disabled for the rest of the session

    vi.unstubAllGlobals();
  });

  it('leaves aiTransition on deterministic behavior when the model file is missing', async () => {
    ort.InferenceSession.create.mockRejectedValue(new Error('404 not found'));
    vi.stubGlobal('fetch', vi.fn(async () => ({ json: async () => transitionMeta })));

    const engine = makeEngine();
    engine.creativeFlags.aiTransition = true;
    const track = { structure: { segments: [{ start: 0, end: 100, energy: 0.9 }] } };

    const plan = await engine._transitionPlan(track, 10);

    expect(plan.transitionMs).toBe(20000); // PEAK_BLEND_SEC — deterministic path
    expect(engine.creativeFlags.aiTransition).toBe(false);

    vi.unstubAllGlobals();
  });
});

describe('_transitionPlan — AI path clamping', () => {
  it('clamps out-of-range model output to the deterministic safe ranges', async () => {
    const run = vi.fn(async () => ({ output: { data: Float32Array.from([999999, 5, -5]) } }));
    ort.InferenceSession.create.mockResolvedValue({ run });
    vi.stubGlobal('fetch', vi.fn(async () => ({ json: async () => transitionMeta })));

    const engine = makeEngine();
    engine.creativeFlags.aiTransition = true;
    const track = { structure: { segments: [{ start: 0, end: 100, energy: 0.9 }] } };

    const plan = await engine._transitionPlan(track, 10);

    expect(plan.transitionMs).toBeLessThanOrEqual(90000); // VALLEY_BLEND_SEC ceiling
    expect(plan.duckDb).toBe(0); // clamped to the 0 ceiling (raw was +5)
    expect(plan.fxIntensity).toBe(0); // clamped to the 0 floor (raw was -5)

    vi.unstubAllGlobals();
  });
});

describe('_pickNextTrack / _transitionPlan — inference throws after successful load', () => {
  it('falls back to a deterministic score for a candidate whose inference call throws', async () => {
    const run = vi.fn(async () => {
      throw new Error('session.run failed');
    });
    ort.InferenceSession.create.mockResolvedValue({ run });
    vi.stubGlobal('fetch', vi.fn(async () => ({ json: async () => selectionMeta })));

    const engine = makeEngine();
    engine.creativeFlags.aiSelection = true;
    engine.creativeFlags.sampling = false;
    engine.creativeFlags.novelty = false;
    engine.library = [
      { name: 'far.mp3', energy: 0.1, bpm: 120 },
      { name: 'near.mp3', energy: 0.59, bpm: 120 },
    ];
    engine.current = null;
    engine.played = new Set();
    engine._energyTarget = () => 0.6;

    const picked = await engine._pickNextTrack();

    expect(picked.name).toBe('near.mp3'); // per-candidate fallback score still ran
    expect(engine.creativeFlags.aiSelection).toBe(true); // load succeeded — flag stays on, only this call fell back

    vi.unstubAllGlobals();
  });

  it('falls back to the deterministic transition plan when session.run() throws after a successful load', async () => {
    const run = vi.fn(async () => {
      throw new Error('session.run failed');
    });
    ort.InferenceSession.create.mockResolvedValue({ run });
    vi.stubGlobal('fetch', vi.fn(async () => ({ json: async () => transitionMeta })));

    const engine = makeEngine();
    engine.creativeFlags.aiTransition = true;
    const track = { structure: { segments: [{ start: 0, end: 100, energy: 0.9 }] } };

    const plan = await engine._transitionPlan(track, 10);

    expect(plan.transitionMs).toBe(20000); // PEAK_BLEND_SEC — deterministic fallback
    expect(engine.creativeFlags.aiTransition).toBe(true); // load succeeded — flag stays on

    vi.unstubAllGlobals();
  });
});
