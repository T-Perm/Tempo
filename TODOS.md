# TODOS

## Product

### Venue product build (dashboard, licensing bundle, POS/ops integration)

**What:** The full venue-side product — recurring subscription dashboard, commercial music licensing bundle, POS/ops integration.

**Why:** This is the actual target business per the TEMPO design doc's sequencing decision (parties first for validation, venues for the real recurring-revenue business), but has zero design work done yet.

**Context:** Only start this after a party pilot (see design doc's "The Assignment") shows a real willingness-to-pay signal, not just enthusiasm. Captures the reasoning now, while fresh, so it isn't lost between now and when the pilot validates — but adds a backlog item for work that may never happen if the pilot fails to show willingness-to-pay. See `~/.gstack/projects/DJ-suite/email-unknown-design-20260801-222140.md` for full context.

**Effort:** XL
**Priority:** P3
**Depends on:** Successful party pilot outcome.

### Guest QR request page design pass

**What:** Wireframe and interaction states (loading, empty, error, success) for the guest-facing song-request page — scan QR, search a track, submit.

**Why:** `/plan-design-review` only covered the host-approval screen; this screen has zero design work done yet, and TEMPO's whole product claim depends on guests actually using it.

**Context:** Lower risk than the host screen (simpler search+submit flow), but still needs empty/error/success states specified so an engineer doesn't ship "No results found." as the whole empty state. Closes the remaining UI gap before build starts on that half of the pilot, at the cost of one more step before implementation can start on the guest-facing side. See `~/.gstack/projects/DJ-suite/email-unknown-design-20260801-222140.md` for the host-screen decisions this should stay consistent with (typeface, dark theme, connectivity indicator pattern).

**Effort:** M
**Priority:** P2
**Depends on:** None — can run anytime before that screen is built.

## Mixing Engine

### Chunk or offload beat-grid onset detection and structural segmentation

**What:** `_detectOnsets()` and (added round 6) `_analyzeStructure()` in `public/host/engine.js` both run fully synchronously per track during library pre-analysis — the latter adds a full-track mono/lowpass pass plus an up-to-300x300 similarity/novelty matrix — which can briefly freeze the setup-progress UI on long tracks.

**Why:** Flagged by adversarial review during the `mvp-mixing-console` ship (originally `_detectOnsets` only; `/ship`'s performance specialist flagged `_analyzeStructure` as the same class of issue on 2026-08-05). Only affects the one-time "choose your music folder" setup step before a party starts, not live mixing — already labeled "one-time setup cost" in the UI.

**Context:** Fixing properly means chunked yielding (`await` every N frames) or moving the computation to a Web Worker. Deferred as real scope beyond the mixing-engine ship; revisit if it's actually painful at real library sizes.

**Effort:** S
**Priority:** P3
**Depends on:** None.

### Headphone pre-listen for cue points

**What:** Add a second audio output route (`setSinkId`) so the host can preview a cue point on headphones before it goes live, instead of "cue" always being a live jump on the audience-facing output.

**Why:** Deferred at the `/autoplan` D4 gate — this architecture has one shared `AudioContext.destination`; adding true pre-listen means a second output-device picker and doubles the audio routing surface.

**Context:** Matches how real DJ hardware works, but meaningfully more build risk than most hosts (non-DJs) will use. Revisit if pilot feedback specifically calls out bad/wrong cue jumps as a problem.

**Effort:** M
**Priority:** P4
**Depends on:** None.

### Mixer console mobile layout

**What:** The mixer console (crossfader, EQ, cue, tempo) is currently laptop/tablet-width only — no dedicated mobile layout.

**Why:** Deliberately deferred at the `/autoplan` Design phase (decision 5d) — matches the existing architecture decision that the engine runs on the host's own laptop, not a phone.

**Context:** The guest-facing request page remains the mobile-optimized surface. Revisit only if a pilot host specifically wants to run the whole thing from a phone/tablet.

**Effort:** M
**Priority:** P4
**Depends on:** None.

### Read-the-room feedback (guest-request/skip bias on selection)

**What:** Use guest-request approvals and skip/timeout outcomes as a live signal to bias the energy-target curve or penalize a track's energy/BPM neighborhood — "the DJ reacts to the crowd," not just its own rule set.

**Why:** Cut from the `/autoplan` "DJ feel" pass at the D3 gate. User's stated reasoning: the DJ needs to be able to mix well (selection variety + transition expressiveness) before it can meaningfully read the room — this is a later-sequenced capability, not a pilot-blocking one.

**Context:** The eng review for this pass also flagged that guest requests currently have no rate limit, decay, or stable per-guest identity (`requester` is free text) — one motivated guest could already force next-picks via the existing priority queue, and this feature would let that guest also steer the ambient energy curve with no cap. Any future build of this needs that abuse-resistance work, not just the feedback wiring.

**Effort:** M
**Priority:** P3
**Depends on:** Mixing quality (selection sampling/novelty, transition variety) validated at a real pilot first.

### Semantic structure labeling, harmonic mixing, and tolerance-based beatmatching

**What:** Four extensions to the real structural signal built in `/autoplan` round 6 (`_analyzeStructure` — self-similarity + novelty segmentation), each explicitly scoped out of that pass: (1) semantic section labeling (verse/chorus/drop, not just "loudness went up/down here" — round 6's segmentation deliberately only detects loudness/density transitions, same honesty bar as the existing onset detector); (2) harmonic-key compatibility folded into track selection and the joint transition decision; (3) vocal-clash detection (avoid blending two vocal sections together); (4) tolerance-based beatmatching beyond the existing ±6% hard cap — treating "close enough" tempo as locked rather than only exact-or-bail equal-power fallback.

**Why:** Named explicitly by the CEO review at the round-6 gate as real, multi-week, ML-adjacent work — the product owner chose to build the real (non-ML) structural signal instead and accepted these as deferred, not silently dropped.

**Context:** True section labeling likely needs either a trained segmentation model (no viable in-browser path without a server) or much more sophisticated hand-rolled feature extraction (chroma/MFCC via an added FFT) than round 6's coarse 3-band energy features. Harmonic compatibility was already ruled out earlier in this project's history ("no simple, reliable browser library exists" — same call as the original key-detection scope cut). Vocal-clash detection needs a vocal-presence signal that doesn't exist yet.

**Effort:** L (each sub-item is independently sizeable)
**Priority:** P3
**Depends on:** Round 6's real structural signal validated by ear at a real listening pass first — no part of this has been heard live yet.

### Live multi-stem synced blending (vocal-hold-back, isolated-bass swap, drums drop-out)

**What:** Three transition techniques enabled by round 7's Demucs stem separation, but requiring live multi-stem playback DURING an active crossfade rather than round 7's single-stem, pre-crossfade loop-roll: (1) vocal-hold-back / instrumental-first blends (play the incoming track's instrumental before its vocals enter); (2) isolated-bass EQ-swap (replace the bass-swap EQ duck with the outgoing track's actual bass stem fading out against the incoming track's actual bass stem, instead of a shared-frequency-band cut); (3) drums drop-out on breakdowns (mute the drums stem during a detected low-energy segment, structure-aware).

**Why:** Explicitly considered and rejected in round 7's scope decision — real DJ moves, but real risk: separate `<audio>` elements per stem can drift out of sync with each other (HTMLMediaElement playback isn't sample-accurate across multiple elements), and decoding 4-8 stem files synchronously at crossfade time adds real latency and more moving parts than anything built in 6 prior rounds, with no live audio validation across any round so far.

**Context:** Round 7 shipped the lower-risk version instead — loop-roll's echo-stutter uses the isolated drums stem, but only before the crossfade starts, reusing the existing single-source beat-grid/timing/handoff logic unchanged. Revisit once that stem variant has actually been heard.

**Effort:** L
**Priority:** P3
**Depends on:** Loop-roll's stem variant validated at a real listening pass first.

## Completed
