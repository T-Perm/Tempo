import { MusicEngine } from './engine.js';
import { DeckVisualizer } from './visualizer.js';

const TIMEOUT_MS = 90 * 1000; // must match server/index.js REQUEST_TIMEOUT_MS

const statusDot = document.getElementById('status-dot');
const statusLabel = document.getElementById('status-label');
const nowPlayingText = document.getElementById('now-playing-text');
const setupPanel = document.getElementById('setup-panel');
const loadLibraryBtn = document.getElementById('load-library-btn');
const changeFolderLink = document.getElementById('change-folder-link');
const setupProgress = document.getElementById('setup-progress');
const setupError = document.getElementById('setup-error');
const pendingSection = document.getElementById('pending-section');
const toastSection = document.getElementById('toast-section');
const queueSection = document.getElementById('queue-section');
const queueList = document.getElementById('queue-list');
const emptyState = document.getElementById('empty-state');

const mixerToggleBtn = document.getElementById('mixer-toggle-btn');
const mixerPanel = document.getElementById('mixer-panel');
const mixerBanner = document.getElementById('mixer-banner');
const backToAutoBtn = document.getElementById('back-to-auto-btn');
const mixerOverlay = document.getElementById('mixer-autopilot-overlay');
const deckTrackEls = { A: document.getElementById('deck-a-track'), B: document.getElementById('deck-b-track') };
const visualController = document.getElementById('visual-controller');

/** @type {Map<string, {id: string, track: string, requester: string, submittedAt: number}>} */
const pendingRequests = new Map();

const engine = new MusicEngine({
  onNowPlaying: (track) => {
    nowPlayingText.innerHTML = `Now: <span class="track">${escapeHtml(track.name)}</span>`;
    deckTrackEls[engine.liveDeckId].textContent = track.name;
  },
  onLibraryProgress: ({ done, total }) => {
    setupProgress.textContent = `Analyzing library: ${done}/${total} tracks (BPM + energy, one-time setup cost)`;
  },
  onLibraryError: ({ name, error }) => {
    setupError.textContent = `Skipped "${name}": ${error}`;
  },
  onManualStateChange: (armed) => {
    mixerBanner.style.display = armed ? 'flex' : 'none';
    mixerOverlay.classList.toggle('hidden', armed);
    if (!armed) resetMixerSliders(); // engine resets audio state to neutral — UI must match
  },
  onAutoPilotResumed: () => {
    renderToast('Auto-pilot resumed', false);
  },
});

/** @type {FileSystemDirectoryHandle|null} Set by the remembered-folder check below; consumed on first click. */
let rememberedDirHandle = null;

// Offer the remembered folder instead of the OS picker, if one exists from a
// previous session — this is the "point to my music folder once" fix. Still
// needs one click either way (AudioContext.resume() requires a user gesture,
// see engine.js unlockAudio() comment), but skips the native folder dialog
// and (via engine.js's per-track cache) the full re-analysis pass.
engine.tryLoadRememberedDirectory().then((remembered) => {
  if (!remembered) return;
  rememberedDirHandle = remembered.handle;
  loadLibraryBtn.textContent = `Use remembered folder (${remembered.name})`;
  changeFolderLink.style.display = 'inline-block';
});

changeFolderLink.addEventListener('click', (e) => {
  e.preventDefault();
  rememberedDirHandle = null;
  loadLibraryBtn.textContent = 'Choose your music folder';
  changeFolderLink.style.display = 'none';
});

loadLibraryBtn.addEventListener('click', async () => {
  engine.unlockAudio(); // must run first, synchronously in the gesture — see engine.js
  setupError.textContent = '';
  loadLibraryBtn.disabled = true;
  changeFolderLink.style.display = 'none';
  loadLibraryBtn.textContent = 'Analyzing…';
  try {
    await engine.loadLibraryFromDirectory(rememberedDirHandle || undefined);
    setupProgress.textContent = `Library ready — ${engine.library.length} tracks analyzed. Starting set…`;
    setupPanel.style.display = 'none';
    mixerToggleBtn.style.display = 'inline-block';
    visualController.style.display = 'flex';
    // Always visible (not behind the mixer toggle) — this is the check-in
    // confidence glance, per the /autoplan D1 decision, so it starts as soon
    // as there's a deck to show.
    new DeckVisualizer(
      document.getElementById('vis-canvas-a'),
      engine.playerA,
      document.getElementById('vis-meter-a'),
      document.getElementById('vis-bpm-a')
    );
    new DeckVisualizer(
      document.getElementById('vis-canvas-b'),
      engine.playerB,
      document.getElementById('vis-meter-b'),
      document.getElementById('vis-bpm-b')
    );
    renderPending(); // shows the empty state now that setup is done
    await engine.start();
  } catch (err) {
    setupError.textContent = err.message || String(err);
    loadLibraryBtn.disabled = false;
    // Whatever failed (permission denied, no audio files, bad browser), a
    // remembered handle that just failed isn't worth re-offering — fall back
    // to the plain picker rather than repeating the same failure.
    rememberedDirHandle = null;
    loadLibraryBtn.textContent = 'Choose your music folder';
    changeFolderLink.style.display = 'none';
  }
});

// --- Mixer console: opt-in panel, hold-to-arm overlay, live controls ---
// Collapsed by default (autoplan Design decision #1); touching any control
// arms manual mode globally (decision #5c), so the overlay's job is only to
// gate the *first* touch behind a deliberate hold, not every subsequent drag.

mixerToggleBtn.addEventListener('click', () => {
  const willShow = mixerPanel.style.display === 'none';
  mixerPanel.style.display = willShow ? 'block' : 'none';
  mixerToggleBtn.classList.toggle('active', willShow);
});

backToAutoBtn.addEventListener('click', () => engine.backToAuto());

const ARM_HOLD_MS = 350;
let armHoldTimer = null;
mixerOverlay.addEventListener('pointerdown', () => {
  armHoldTimer = setTimeout(() => engine.armManual(), ARM_HOLD_MS);
});
['pointerup', 'pointerleave', 'pointercancel'].forEach((evt) => {
  mixerOverlay.addEventListener(evt, () => clearTimeout(armHoldTimer));
});
// A pointer-down interrupted by an OS-level focus switch (alt-tab, notification)
// may not fire pointercancel — clear on blur/visibility loss too.
window.addEventListener('blur', () => clearTimeout(armHoldTimer));
document.addEventListener('visibilitychange', () => {
  if (document.hidden) clearTimeout(armHoldTimer);
});
// Keyboard equivalent of the pointer hold-to-arm gesture — no accidental-touch
// risk for keyboard input, so Enter/Space arms immediately, no hold delay.
mixerOverlay.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    engine.armManual();
  }
});

document.querySelectorAll('.eq-slider').forEach((el) => {
  el.addEventListener('input', () => {
    engine.setEQ(el.dataset.deck, el.dataset.band, Number(el.value));
  });
});

document.querySelectorAll('.tempo-slider').forEach((el) => {
  el.addEventListener('input', () => {
    engine.setTempoNudge(el.dataset.deck, Number(el.value));
  });
});

document.getElementById('crossfader-slider').addEventListener('input', (event) => {
  engine.setCrossfader(Number(event.target.value));
});

document.querySelectorAll('.cue-btn').forEach((el) => {
  el.addEventListener('click', () => engine.setCue(el.dataset.deck));
});

function resetMixerSliders() {
  document.querySelectorAll('.eq-slider, .tempo-slider').forEach((el) => {
    el.value = 0;
  });
  document.getElementById('crossfader-slider').value = 0.5;
}

// --- WebSocket relay: pending requests, approvals, connectivity status ---

let socket;
function connect() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  socket = new WebSocket(`${proto}//${location.host}`);

  socket.addEventListener('open', () => {
    socket.send(JSON.stringify({ type: 'register', role: 'host' }));
    setOnline(true);
  });

  socket.addEventListener('close', () => {
    setOnline(false);
    setTimeout(connect, 2000); // auto-reconnect — a dropped host connection shouldn't need a manual refresh
  });

  socket.addEventListener('error', () => setOnline(false));

  socket.addEventListener('message', (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === 'pending_request') {
      pendingRequests.set(msg.id, msg);
      renderPending();
    } else if (msg.type === 'request_resolved') {
      pendingRequests.delete(msg.id);
      if (msg.outcome === 'auto-approved' || msg.outcome === 'approved') {
        engine.enqueueRequestedTrack(msg.track);
        renderToast(`Auto-added "${msg.track}"`, msg.outcome === 'approved');
      }
      renderPending();
    }
  });
}
connect();

function setOnline(isOnline) {
  statusDot.classList.toggle('offline', !isOnline);
  statusLabel.textContent = isOnline ? 'Online' : 'Reconnecting…';
}

function renderToast(text, wasManualApproval) {
  const message = wasManualApproval ? `Added "${text.match(/"(.*)"/)?.[1] || text}"` : text;
  const el = document.createElement('div');
  el.className = 'toast';
  el.innerHTML = `&#10003;&nbsp; ${escapeHtml(message)}`;
  toastSection.appendChild(el);
  setTimeout(() => el.remove(), 3000);
}

function renderPending() {
  pendingSection.innerHTML = '';

  if (pendingRequests.size === 0) {
    emptyState.style.display = pendingRequests.size === 0 && engine.library.length > 0 ? 'block' : 'none';
    return;
  }
  emptyState.style.display = 'none';

  const title = document.createElement('div');
  title.className = 'section-title';
  title.style.marginTop = '0';
  title.textContent = `Needs your OK — ${pendingRequests.size} pending`;
  pendingSection.appendChild(title);

  for (const req of pendingRequests.values()) {
    pendingSection.appendChild(renderPendingCard(req));
  }
}

function renderPendingCard(req) {
  const card = document.createElement('div');
  card.className = 'pending-card';
  card.id = `pending-${req.id}`;

  const elapsed = Date.now() - req.submittedAt;
  const fraction = Math.min(1, elapsed / TIMEOUT_MS);
  card.style.setProperty('--urgency', urgencyColor(fraction));

  card.innerHTML = `
    <div class="label">Requested by ${escapeHtml(req.requester)}</div>
    <div class="track">${escapeHtml(req.track)}</div>
    <div class="urgency-note">${fraction < 0.5 ? "Just arrived — plenty of time" : "Auto-inserting soon if you don't respond"}</div>
    <div class="actions">
      <button class="approve" data-id="${req.id}">Approve</button>
      <button class="reject" data-id="${req.id}">Skip</button>
    </div>
  `;

  card.querySelector('.approve').addEventListener('click', () => {
    socket.send(JSON.stringify({ type: 'approve', id: req.id }));
  });
  card.querySelector('.reject').addEventListener('click', () => {
    socket.send(JSON.stringify({ type: 'skip', id: req.id }));
  });

  return card;
}

/** Ambient urgency per /plan-design-review Pass 3 — a color shift, never a ticking number. */
function urgencyColor(fraction) {
  // calm purple (#6d5ef4) -> warm amber (#e0a63c) as the 90s window elapses
  const from = [0x6d, 0x5e, 0xf4];
  const to = [0xe0, 0xa6, 0x3c];
  const mix = from.map((c, i) => Math.round(c + (to[i] - c) * fraction));
  return `rgb(${mix.join(',')})`;
}

// Re-render every second so urgency color and copy stay live without a full re-fetch.
setInterval(() => {
  if (pendingRequests.size > 0) renderPending();
}, 1000);

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}
