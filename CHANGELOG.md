# Changelog

All notable changes to TEMPO are documented in this file.

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
