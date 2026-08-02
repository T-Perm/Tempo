import { MusicEngine } from './engine.js';

const TIMEOUT_MS = 90 * 1000; // must match server/index.js REQUEST_TIMEOUT_MS

const statusDot = document.getElementById('status-dot');
const statusLabel = document.getElementById('status-label');
const nowPlayingText = document.getElementById('now-playing-text');
const setupPanel = document.getElementById('setup-panel');
const loadLibraryBtn = document.getElementById('load-library-btn');
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
  },
  onAutoPilotResumed: () => {
    renderToast('Auto-pilot resumed', false);
  },
});

loadLibraryBtn.addEventListener('click', async () => {
  engine.unlockAudio(); // must run first, synchronously in the gesture — see engine.js
  setupError.textContent = '';
  loadLibraryBtn.disabled = true;
  loadLibraryBtn.textContent = 'Analyzing…';
  try {
    await engine.loadLibraryFromDirectory();
    setupProgress.textContent = `Library ready — ${engine.library.length} tracks analyzed. Starting set…`;
    setupPanel.style.display = 'none';
    mixerToggleBtn.style.display = 'inline-block';
    renderPending(); // shows the empty state now that setup is done
    await engine.start();
  } catch (err) {
    setupError.textContent = err.message || String(err);
    loadLibraryBtn.disabled = false;
    loadLibraryBtn.textContent = 'Choose your music folder';
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
