# TODOS

## Venue product build (dashboard, licensing bundle, POS/ops integration)
- **What:** The full venue-side product — recurring subscription dashboard, commercial music licensing bundle, POS/ops integration.
- **Why:** This is the actual target business per the TEMPO design doc's sequencing decision (parties first for validation, venues for the real recurring-revenue business), but has zero design work done yet.
- **Pros:** Captures the reasoning now, while fresh, so it isn't lost between now and when the pilot validates.
- **Cons:** Adds a backlog item for work that may never happen if the pilot fails to show willingness-to-pay.
- **Context:** Only start this after a party pilot (see design doc's "The Assignment") shows a real willingness-to-pay signal, not just enthusiasm. See `~/.gstack/projects/DJ-suite/email-unknown-design-20260801-222140.md` for full context.
- **Depends on / blocked by:** Successful party pilot outcome.

## Guest QR request page design pass
- **What:** Wireframe and interaction states (loading, empty, error, success) for the guest-facing song-request page — scan QR, search a track, submit.
- **Why:** `/plan-design-review` only covered the host-approval screen; this screen has zero design work done yet, and TEMPO's whole product claim depends on guests actually using it.
- **Pros:** Closes the remaining UI gap before build starts on that half of the pilot.
- **Cons:** Adds a step before implementation can start on the guest-facing side.
- **Context:** Lower risk than the host screen (simpler search+submit flow), but still needs empty/error/success states specified so an engineer doesn't ship "No results found." as the whole empty state. See `~/.gstack/projects/DJ-suite/email-unknown-design-20260801-222140.md` for the host-screen decisions this should stay consistent with (typeface, dark theme, connectivity indicator pattern).
- **Depends on / blocked by:** None — can run anytime before that screen is built.
