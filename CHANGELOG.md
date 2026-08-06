# Changelog

All notable changes to TEMPO are documented in this file.

## [0.4.0.0] - 2026-08-05

### Added
- Transitions now read from the outgoing track's real musical structure (a per-track self-similarity/novelty segmentation computed once at library-load time) instead of a synthetic clock, so blend length, EQ duck depth, and FX intensity all respond to where the track actually is, not just how long the set has been running. Falls back to the old synthetic curve when a track is too short or its structure can't be reliably detected.
- Transitions are a genuinely variable, context-appropriate overlap — quick and punchy near an energy peak, long and gentle in a low-energy stretch — replacing a single fixed-length crossfade.
- The "echo-stutter" loop-roll effect is now occasion-gated (near a rising energy peak, spaced apart from the last showy move, never twice in a row) instead of firing on every eligible transition. Still off by default pending a live listening pass.
- Optional local vocal/drums/bass/other stem separation via Demucs, run offline and served by the existing local server. When on and a track's stems are available, loop-roll uses the isolated drums stem for a cleaner hit with no vocal/bass bleed. Off by default.
- A first automated test suite (Vitest) covering the engine's core decision logic: track-selection scoring and the exhaustion fallback, beat-grid detection's half/double-time sanity check, the transition-planning function, the manual mixer console's control surface, and the loop-roll occasion gate.

### Fixed
- The scheduler for the next automatic transition treated an absolute position in the new track's timeline as a countdown from right now, without accounting for the playhead already being partway into the track from the tempo-sync seek and the crossfade itself — every automatic transition was mistimed, in the worst case firing at or past the track's own end and cutting to silence.
- A tempo-synced track kept playing up to 6% off its natural tempo for its entire runtime after a transition, not just during the blend, because only the outgoing deck's playback rate was reset afterward.
- Manually nudging the crossfader while only one deck had a track loaded caused an unintended volume dip on the one audible deck instead of leaving it at full volume.
- Arming manual mixer control before the very first track finished loading could strand both decks in silence with no clear way to tell it apart from a normal manual session.
- A track that never actually reached the audience (because manual control was armed mid-transition) could still get marked "played," making it wait for a full library reset before it could be picked again.
- The Python virtual environment, separated stem audio, and the source playlist folder were untracked but not gitignored, risking an accidental multi-gigabyte commit.

## [0.3.0.0] - 2026-08-03

### Added
- Visual DJ controller on the host screen: per-deck waveform with beat-grid overlay, a live playhead, a post-fade level meter, and a BPM readout, so the host has a check-in confidence glance at what the engine is doing without opening the mixer console. Tracks with no detected beat grid show a dimmed/hatched region instead of blank space; BPM readouts show "~N (est.)" when the number came from the fallback detector rather than a confident reading.
- "DJ feel" creative layer on top of the deterministic mixing engine: weighted sampling from the top few candidate tracks (instead of always picking the single best-scoring one), a novelty penalty against recently-played tracks, and transition variety (shorter punchier crossfades near energy peaks, longer gentler blends in valleys). Each mechanism is independently killable at runtime from devtools (`engine.creativeFlags.sampling` / `.novelty` / `.transitionVariety`) with no redeploy. The engine logs its RNG seed on load so a night's pick sequence can be reconstructed post-mortem.

### Fixed
- The visual controller's playhead, waveform, and BPM readout could show the previous track's data for several seconds during every transition, since the audio element's timeline switched before the track record it was displayed against. Now they switch in lockstep.
- The level meter's `AnalyserNode` buffer was sized to half the intended sample window, silently truncating the RMS read.
- A worst-case (valley) transition duration could exceed the scheduling buffer reserved for it, cutting a crossfade to silence instead of completing the blend.
- The library-exhaustion reset could hand back the currently-playing track as the "next" pick.

## [0.2.0.0] - 2026-08-02

### Added
- Beatmatched mixing: tempo-synced, phrase-aligned crossfades with an equal-power gain curve and a bass-swap EQ duck, replacing the flat linear crossfade.
- Per-track beat-grid detection (onset analysis) with automatic fallback to average-BPM sync when a track's grid is degenerate or fails a half/double-time sanity check.
- Manual mixer console: crossfader, 3-band EQ per deck, cue points, and tempo nudge, with a hold-to-arm gesture (mouse and keyboard) and a one-tap "Back to Auto" to hand control back to the algorithm.

### Changed
- Guest song-request matching now requires a word-boundary match and rejects ambiguous multi-match requests instead of guessing.

### Fixed
- Manual mixing no longer permanently halts automatic playback after returning to auto-pilot.
- Returning to auto-pilot now resets any manual EQ/tempo/crossfader changes back to neutral, so they don't silently persist into future automatic mixes.
- Cue points no longer carry over to the wrong track when a deck is reused later in a set.
- The manual mixer no longer risks silencing the room by crossfading toward a deck with no track loaded yet.
