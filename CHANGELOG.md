# Changelog

All notable changes to TEMPO are documented in this file.

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
