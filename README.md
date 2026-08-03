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

The host screen shows a live visual controller (waveform + beat-grid + level meter per deck) as soon as the library loads — a check-in confidence glance, not a control surface. A track with no detected beat grid shows a dimmed/hatched region instead of blank space, and BPM readouts show "~N (est.)" when the number came from the fallback path rather than real detection — both intentional, so a failed analysis never looks like a confident reading.

The host screen also has an optional manual mixer console — click **Mixer** (top of screen) to open the panel, then arm manual control by holding the auto-pilot overlay for a beat (mouse) or pressing Enter/Space on it (keyboard). From there: crossfader, per-deck 3-band EQ, cue points, and tempo nudge. Tap **Back to Auto** at any point to hand control back to the algorithm; manual EQ/tempo/crossfader changes reset to neutral when you do.

The engine also has a "DJ feel" creative layer on top of the deterministic
selection/mixing described above: weighted-sampling from the top few
candidate tracks instead of always picking the single best-scoring one,
a novelty penalty against recently-played tracks, and transition variety
(shorter punchier crossfades near energy peaks, longer gentler blends in
valleys). Each is independently killable at runtime with no redeploy —
open devtools on the host page and set `engine.creativeFlags.sampling`,
`.novelty`, or `.transitionVariety` to `false` to fall back to that
mechanism's pre-existing deterministic behavior mid-party if something
feels off. The engine logs its RNG seed to the console on load so a
given night's pick sequence can be reconstructed post-mortem if needed.

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
