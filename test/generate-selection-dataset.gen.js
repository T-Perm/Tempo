// test/generate-selection-dataset.gen.js
import { writeFileSync, mkdirSync } from 'node:fs';
import { MusicEngine, BPM_PENALTY_WEIGHT, NOVELTY_WEIGHT } from '../public/host/engine.js';

/** Deterministic small synthetic library spanning a spread of energy/BPM
 * combinations — enough variety for _pickNextTrack()'s scoring to produce
 * a non-trivial ranking, without needing a real analyzed library on disk. */
function makeSyntheticLibrary(n = 60) {
  const tracks = [];
  for (let i = 0; i < n; i++) {
    tracks.push({
      name: `track-${i}.mp3`,
      bpm: 100 + (i % 12) * 4, // 100..144
      energy: (i % 10) / 9, // 0..1
    });
  }
  return tracks;
}

/** One simulated "set": a run of consecutive picks that calls _recordPick()
 * for real after each one, so _noveltyPenalty() sees genuine, non-empty
 * history for picks after the first few — matching how a real party set
 * actually accumulates _recentHistory, unlike a one-off snapshot per row. */
function runSet(engine, rows, stepsPerSet, pickIdRef) {
  engine.played = new Set();
  engine._recentHistory = [];
  engine.setStartedAt = Date.now() - Math.floor(Math.random() * 40 * 60 * 1000);
  engine.current = engine.library[Math.floor(Math.random() * engine.library.length)];
  engine.played.add(engine.current.name);

  for (let step = 0; step < stepsPerSet; step++) {
    const target = engine._energyTarget();
    const candidates = engine.library.filter((t) => !engine.played.has(t.name));
    if (candidates.length === 0) break;

    const scored = candidates
      .map((track) => {
        const energyDelta = Math.abs(track.energy - target);
        const bpmDelta = Math.abs(track.bpm - engine.current.bpm);
        const novelty = engine._noveltyPenalty(track);
        return { track, novelty, score: -energyDelta - bpmDelta * BPM_PENALTY_WEIGHT - novelty * NOVELTY_WEIGHT };
      })
      .sort((a, b) => b.score - a.score);

    const pickId = pickIdRef.value++; // shared by every candidate row from this one scoring event — Task 5 groups on this, not a fixed stride
    scored.forEach(({ track, novelty }, rank) => {
      rows.push({
        pickId,
        candidateEnergy: track.energy,
        candidateBpm: track.bpm,
        currentEnergy: engine.current.energy,
        currentBpm: engine.current.bpm,
        energyTarget: target,
        noveltyPenalty: novelty,
        rank,
      });
    });

    const picked = scored[0].track; // argmax — matches the deterministic engine with creativeFlags.sampling off
    engine._recordPick(picked); // real history accumulation, the fix this round is about
    engine.played.add(picked.name);
    engine.current = picked;
  }
}

export function generate({ setsCount = 200, stepsPerSet = 10 } = {}) {
  const engine = new MusicEngine({});
  engine.library = makeSyntheticLibrary();
  const rows = [];
  const pickIdRef = { value: 0 };
  for (let s = 0; s < setsCount; s++) runSet(engine, rows, stepsPerSet, pickIdRef);

  mkdirSync('ml/data', { recursive: true });
  writeFileSync('ml/data/selection_dataset.json', JSON.stringify(rows));
  return rows.length;
}

// Run directly via `npx vitest run test/generate-selection-dataset.gen.js --no-coverage`
// (executed as a plain module, not a describe/it suite — see package.json script).
if (process.env.GENERATE_SELECTION_DATASET === '1') {
  const count = generate({});
  // eslint-disable-next-line no-console
  console.log(`Wrote ${count} rows to ml/data/selection_dataset.json`);
}
