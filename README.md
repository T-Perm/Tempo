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

## Known scope simplifications (intentional, not oversights)

- **No true musical-key detection.** BPM + loudness-based "energy" is the sequencing signal for this pilot. A reliable, simple, browser-based key-detection library doesn't exist the way BPM detection does — adding one would be over-engineering for a 2-week validation pilot. If the pilot validates the core claim, revisit this for the venue product.
- **Guest QR page is functional but not yet design-reviewed.** `/plan-design-review` only ran a full pass on the host-approval screen; the guest request page (`public/guest/`) is built to match the same visual language but hasn't been through its own design pass. Tracked in `TODOS.md`.
- **No persistence across server restarts.** Pending requests and the analyzed library live in memory only — acceptable for a single-night pilot; restarting the server mid-party would lose in-flight state.

## Architecture

- `server/index.js` — Express static file server + WebSocket relay. Holds the single source of truth for pending guest requests, including the 90-second auto-timeout decided in `/plan-eng-review`.
- `public/host/engine.js` — the rules-based DJ engine: library pre-analysis, energy-curve track selection, crossfade playback, and the failure-mode handling for load errors, exhausted libraries, and bad metadata.
- `public/host/app.js` — host UI: WebSocket wiring, pending-request cards, ambient urgency color, toasts, connectivity dot — implements the `/plan-design-review` decisions.
- `public/guest/` — guest-facing request form.
