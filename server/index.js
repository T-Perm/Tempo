import express from 'express';
import http from 'http';
import { WebSocketServer } from 'ws';
import path from 'path';
import { fileURLToPath } from 'url';
import { randomUUID } from 'crypto';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = process.env.PORT || 8787;
const REQUEST_TIMEOUT_MS = 90 * 1000; // per /plan-eng-review 3A decision

const app = express();
app.use(express.json());
app.use('/host', express.static(path.join(__dirname, '../public/host')));
app.use('/guest', express.static(path.join(__dirname, '../public/guest')));
app.get('/', (_req, res) => res.redirect('/host'));

const server = http.createServer(app);
const wss = new WebSocketServer({ server });

/** @type {Map<string, WebSocket>} role: 'host' | 'guest' — hosts get pending_request pushes */
const hostSockets = new Set();

/**
 * Pending requests are the single source of truth for the reconciliation flow
 * decided in /plan-eng-review: host approval OR a 90s auto-timeout, then
 * insertion at the next natural gap (not immediate, not a scored pool —
 * simplified after outside-voice review).
 * @type {Map<string, {id: string, track: string, requester: string, submittedAt: number, timer: NodeJS.Timeout}>}
 */
const pending = new Map();

function broadcastToHosts(message) {
  const payload = JSON.stringify(message);
  for (const socket of hostSockets) {
    if (socket.readyState === socket.OPEN) socket.send(payload);
  }
}

function resolveRequest(id, outcome) {
  const req = pending.get(id);
  if (!req) return; // already resolved (e.g., host approved right as timeout fired)
  clearTimeout(req.timer);
  pending.delete(id);
  broadcastToHosts({
    type: 'request_resolved',
    id,
    outcome, // 'approved' | 'skipped' | 'auto-approved'
    track: req.track,
    requester: req.requester,
  });
}

app.post('/api/request', (req, res) => {
  const { track, requester } = req.body || {};
  if (!track || typeof track !== 'string' || !track.trim()) {
    return res.status(400).json({ error: 'track is required' });
  }
  const id = randomUUID();
  const submittedAt = Date.now();
  const timer = setTimeout(() => resolveRequest(id, 'auto-approved'), REQUEST_TIMEOUT_MS);
  pending.set(id, { id, track: track.trim(), requester: (requester || 'a guest').trim(), submittedAt, timer });

  broadcastToHosts({
    type: 'pending_request',
    id,
    track: track.trim(),
    requester: (requester || 'a guest').trim(),
    submittedAt,
    timeoutMs: REQUEST_TIMEOUT_MS,
  });

  res.json({ id, status: 'pending', timeoutMs: REQUEST_TIMEOUT_MS });
});

wss.on('connection', (socket) => {
  socket.on('message', (raw) => {
    let msg;
    try {
      msg = JSON.parse(raw.toString());
    } catch {
      return; // ignore malformed frames — never crash the relay on bad input
    }

    if (msg.type === 'register' && msg.role === 'host') {
      hostSockets.add(socket);
      // Replay current pending state so a reconnecting/refreshed host doesn't lose requests.
      for (const req of pending.values()) {
        socket.send(JSON.stringify({
          type: 'pending_request',
          id: req.id,
          track: req.track,
          requester: req.requester,
          submittedAt: req.submittedAt,
          timeoutMs: REQUEST_TIMEOUT_MS,
        }));
      }
      return;
    }

    if (msg.type === 'approve' && msg.id) {
      resolveRequest(msg.id, 'approved');
      return;
    }

    if (msg.type === 'skip' && msg.id) {
      resolveRequest(msg.id, 'skipped');
      return;
    }
  });

  socket.on('close', () => hostSockets.delete(socket));
});

server.listen(PORT, () => {
  console.log(`TEMPO pilot server running:`);
  console.log(`  Host screen:  http://localhost:${PORT}/host`);
  console.log(`  Guest screen: http://<this-machine's-local-IP>:${PORT}/guest  (share this on the local wifi/hotspot)`);
});
