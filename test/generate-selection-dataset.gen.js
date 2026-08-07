// test/generate-selection-dataset.gen.js
import { writeFileSync, mkdirSync } from 'node:fs';
import { MusicEngine } from '../public/host/engine.js';

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

function runOnce(engine, rows) {
  engine.played.clear();
  engine._recentHistory = [];
  engine.setStartedAt = Date.now() - Math.floor(Math.random() * 40 * 60 * 1000);
  engine.current = engine.library[Math.floor(Math.random() * engine.library.length)];

  const target = engine._energyTarget();
  const candidates = engine.library.filter((t) => t !== engine.current);
  const scored = candidates
    .map((track) => {
      const energyDelta = Math.abs(track.energy - target);
      const bpmDelta = Math.abs(track.bpm - engine.current.bpm);
      const novelty = engine._noveltyPenalty(track);
      return { track, score: -energyDelta - bpmDelta * 0.01 - novelty * 0.3 };
    })
    .sort((a, b) => b.score - a.score);

  scored.forEach(({ track }, rank) => {
    rows.push({
      candidateEnergy: track.energy,
      candidateBpm: track.bpm,
      currentEnergy: engine.current.energy,
      currentBpm: engine.current.bpm,
      energyTarget: target,
      noveltyPenalty: engine._noveltyPenalty(track),
      rank,
    });
  });
}

export function generate({ iterations = 2000 } = {}) {
  const engine = new MusicEngine({});
  engine.library = makeSyntheticLibrary();
  const rows = [];
  for (let i = 0; i < iterations; i++) runOnce(engine, rows);

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
