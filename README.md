# TEMPO — party pilot

Implements the tasks from `/plan-eng-review` and `/plan-design-review` against
the approved design doc at `~/.gstack/projects/DJ-suite/email-unknown-design-20260801-222140.md`.

## Run it

```
npm install
npm start
```

- Host screen: `http://localhost:8787/host` — open this on the host's own laptop, in **Chrome or Edge** (the local music-folder picker needs the File System Access API, which Safari/Firefox don't support).
- Guest screen: `http://<this-machine's-local-IP>:8787/guest` — share this on the party's wifi or the host's phone hotspot, per the design doc's local-hosting decision.

On the host screen, click **Choose your music folder** and pick a folder of local `.mp3/.wav/.m4a/.ogg` files. The whole library is pre-analyzed (BPM + a loudness-based energy proxy) before the set starts — this runs once, not live during playback, per the performance-review decision.

**Stem separation (optional, offline, one-time).** This MVP is single-laptop only, so "browser-only" was a self-imposed constraint on the DSP, not a real requirement — `/autoplan` 2026-08-04 round 7 added a local Python step for real vocal/drums/bass/other isolation via [Demucs](https://github.com/facebookresearch/demucs), served by the existing Node server at `/stems`, not run live. To separate a playlist folder:

```
python -m venv .venv
.venv/Scripts/pip install demucs numpy   # .venv/bin/pip on macOS/Linux
.venv/Scripts/python -m demucs -o stems playlist/*.mp3
```

CPU-only ran faster than realtime in testing (~0.5x track duration per track). The engine checks `/stems/htdemucs/<track-basename>/{vocals,drums,bass,other}.wav` fresh on every library load (never cached — the batch can finish after a track's analysis was already cached) and sets `stemsAvailable` per track. Currently used by loop-roll: when `engine.creativeFlags.stems` is on and a track's stems exist, the echo-stutter loops the isolated drums stem instead of the full mixdown — no vocal/bass bleed. Defaults **off**, same as `loopRoll` — unheard in this coding session.

The host screen shows a live visual controller (waveform + beat-grid + level meter per deck) as soon as the library loads — a check-in confidence glance, not a control surface. A track with no detected beat grid shows a dimmed/hatched region instead of blank space, and BPM readouts show "~N (est.)" when the number came from the fallback path rather than real detection — both intentional, so a failed analysis never looks like a confident reading.

The host screen also has an optional manual mixer console — click **Mixer** (top of screen) to open the panel, then arm manual control by holding the auto-pilot overlay for a beat (mouse) or pressing Enter/Space on it (keyboard). From there: crossfader, per-deck 3-band EQ, cue points, and tempo nudge. Tap **Back to Auto** at any point to hand control back to the algorithm; manual EQ/tempo/crossfader changes reset to neutral when you do.

**Transitions are an extended overlap, sized to what's musically appropriate for the moment, not a short tail-fade or a single fixed length.** Two tracks play together for a span that varies by context — quick and punchy (~20s) near an energy peak to keep momentum, long and gentle (~90s) in a low-energy stretch, ~45-60s in between — phrase-aligned when a trustworthy beat grid exists. This replaced a ~4-6 second end-of-track crossfade (`/autoplan` 2026-08-04 rounds 4-5) that read as "not mixing" once actually listened to. Known edge case: tracks shorter than ~90s (the worst-case reserve) will crossfade almost immediately instead of playing first — fine for typical full-length songs, worth knowing about for short clips/edits.

**"Context" comes from a real per-track structural signal, not a fake clock.** During pre-analysis, each track gets a Foote-style self-similarity + novelty segmentation (coarse 3-band energy features, no FFT/chroma — see `_analyzeStructure` in `engine.js` for the honesty caveats: it catches real loudness/density transitions, it does not semantically know "chorus" vs "verse"). Transition duration, EQ duck depth, FX intensity, and the loop-roll occasion gate all read from the outgoing track's *current segment* at its *actual playhead position* — one shared signal, not three independent lookups. Falls back to a synthetic energy curve (unrelated to the audio) when a track is too short or its segmentation is degenerate — same fallback behavior as before this pass, never fabricates structure. `/autoplan` 2026-08-04 round 6.

The engine also has a "DJ feel" creative layer on top of the deterministic
selection/mixing described above: weighted-sampling from the top few
candidate tracks instead of always picking the single best-scoring one,
a novelty penalty against recently-played tracks, transition variety
(shorter punchier crossfades near energy peaks, longer gentler blends in
valleys), an FX layer (filter sweep + echo tail on the outgoing track
during a transition), and a single-repeat "echo-stutter" loop-roll (the
outgoing track's current beat, played twice, sample-accurately — only
when its beat grid is trustworthy; silently skipped otherwise). Each
mechanism is independently killable at runtime with no redeploy — open
devtools on the host page and set `engine.creativeFlags.sampling`,
`.novelty`, `.transitionVariety`, `.fx`, or `.loopRoll` to `false` to fall
back to that mechanism's pre-existing deterministic behavior mid-party if
something feels off. The engine logs its RNG seed to the console on load
so a given night's pick sequence can be reconstructed post-mortem if
needed.

**`loopRoll` defaults to `false`.** Firing it on every eligible transition
read as mechanical rather than a deliberate DJ move, so it's now gated by
`_shouldLoopRoll()` — only near a rising energy peak, spaced at least
`MIN_SHOWY_SPACING` transitions from the last showy-technique firing, never
the same technique twice in a row. The gate's feel (spacing, peak margin) is
unverified — no live audio playback happened while building it. Flip
`engine.creativeFlags.loopRoll = true` in devtools for a real listening pass
before relying on it at a party.

The host page also remembers the last music folder you picked (via the
File System Access API + IndexedDB) — after the first pick, the setup
button reads "Use remembered folder" instead of reopening the OS picker,
and previously-analyzed tracks (matched by name/size/last-modified) skip
re-decoding entirely. Click "Use a different folder" to pick a new one.

## Known scope simplifications (intentional, not oversights)

- **No true musical-key detection.** BPM + loudness-based "energy" is the sequencing signal for track selection. A reliable, simple, browser-based key-detection library doesn't exist the way BPM detection does — adding one would be over-engineering for this build. If the pilot validates the core claim, revisit this for the venue product.
- **Beat-grid detection is a pilot-scale approximation, not a production DSP library.** `engine.js`'s `_detectOnsets`/`_analyzeBeatGrid` use a simple energy-flux novelty function with an adaptive threshold — no FFT/spectral analysis. It's sanity-checked against the whole-track average BPM to catch half/double-time misreads, and falls back to average-BPM sync (today's original behavior) whenever the grid is degenerate (fewer than 4 onsets) or fails the sanity check. Flagged explicitly in `/autoplan`'s eng review as the highest-risk, least-tested part of this build — **if it isn't demonstrably stable a few days before a real pilot, ship with average-BPM-only sync for that pilot** rather than debugging tempo-sync live.
- **No "read-the-room" feedback yet.** Guest requests and skips don't bias the ambient energy curve or track selection — deliberately cut from the `/autoplan` "DJ feel" pass; mixing quality (selection variety, transition expressiveness) needs to be validated at a real pilot before crowd-responsiveness is worth the added risk (unauthenticated guest requests also have no rate limit yet, which this would need first). Tracked in `TODOS.md`.
- **Manual mixer console has no headphone pre-listen.** "Cue" jumps the audience-facing track directly (there's one shared `AudioContext.destination`, no second output route) — a deliberate scope decision (`/autoplan` D4), not an oversight. A bad cue point is an audible skip in front of guests, not a private mistake caught on headphones first.
- **Manual mixer console is laptop/tablet-width only.** No dedicated mobile layout this pass — matches the existing "engine runs on the host's own laptop" architecture decision. The guest-facing request page remains the mobile-optimized surface.
- **Guest QR page is functional but not yet design-reviewed.** `/plan-design-review` only ran a full pass on the host-approval screen; the guest request page (`public/guest/`) is built to match the same visual language but hasn't been through its own design pass. Tracked in `TODOS.md`.
- **Beat-grid onset detection runs synchronously per track during library pre-analysis.** For long tracks this can briefly freeze the setup-progress UI (already labeled "one-time setup cost"). Only affects the pre-party setup step, not live mixing. Fixing properly means chunked yielding or a Web Worker — real scope beyond this ship; revisit if it's actually painful at real library sizes.
- **No persistence across server restarts.** Pending requests and the analyzed library live in memory only — acceptable for a single-night pilot; restarting the server mid-party would lose in-flight state.

## Architecture

- `server/index.js` — Express static file server + WebSocket relay. Holds the single source of truth for pending guest requests, including the 90-second auto-timeout decided in `/plan-eng-review`.
- `public/host/engine.js` — the DJ engine: library pre-analysis (BPM, energy, beat-grid), energy-curve track selection, beatmatched crossfade playback (tempo-sync, phrase-aligned mix points, equal-power gain curve, bass-swap EQ duck), a manual mixer console API (crossfader, per-deck 3-band EQ, cue points, tempo nudge, global auto-pilot/manual-arm state machine), and the failure-mode handling for load errors, exhausted libraries, and bad metadata. Scope and every design decision behind this file: `~/.gstack/projects/DJ-suite/email-unknown-plan-mvp-mixing-20260802.md`.
- `public/host/app.js` — host UI: WebSocket wiring, pending-request cards, ambient urgency color, toasts, connectivity dot, and the mixer console wiring (opt-in collapsed panel, hold-to-arm gesture, live control bindings).
- `public/guest/` — guest-facing request form.
