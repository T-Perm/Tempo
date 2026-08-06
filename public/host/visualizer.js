// Visual DJ controller — /autoplan 2026-08-02. Read-only waveform/beat-grid/
// meter view for the host's check-in confidence glance. Never drives audio
// decisions — engine.js owns all of that; this only reads deck state.
//
// Performance: the waveform + beat-grid render is cached to an offscreen
// canvas once per track load (O(peaks) only on track change), and every
// animation frame only blits that cache plus a thin playhead line and the
// meter bar (O(1) per frame) — see autoplan Eng review, "redraw cost".

const BEAT_TICK_DENSITY_PX = 2; // minimum on-screen spacing between drawn beat-grid ticks

export class DeckVisualizer {
  /** @param {HTMLCanvasElement} canvas @param {{audio: HTMLAudioElement, analyser: AnalyserNode, track?: object}} deck @param {HTMLElement} meterFillEl @param {HTMLElement} bpmEl */
  constructor(canvas, deck, meterFillEl, bpmEl) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.deck = deck;
    this.meterFillEl = meterFillEl;
    this.bpmEl = bpmEl;
    this._staticBitmap = null;
    this._lastRenderedTrackName = null;
    // getByteTimeDomainData copies min(array.length, fftSize) samples — sizing
    // to frequencyBinCount (fftSize/2) silently truncates the window it reads.
    this._meterData = new Uint8Array(deck.analyser.fftSize);
    this._stopped = false;
    this._tick();
  }

  stop() {
    this._stopped = true;
  }

  _renderStatic(track) {
    const { width, height } = this.canvas;
    const off = document.createElement('canvas');
    off.width = width;
    off.height = height;
    const octx = off.getContext('2d');

    if (!track.beatGrid) {
      // Explicit "no beat grid detected" state — a silently-empty overlay
      // would look indistinguishable from a broken UI (autoplan Eng review).
      octx.fillStyle = 'rgba(248, 113, 113, 0.08)';
      octx.fillRect(0, 0, width, height);
    } else {
      octx.strokeStyle = 'rgba(224, 166, 60, 0.4)';
      octx.lineWidth = 1;
      let lastX = -Infinity;
      for (const t of track.beatGrid) {
        const x = (t / track.duration) * width;
        if (x - lastX < BEAT_TICK_DENSITY_PX) continue; // downsample to pixel density
        lastX = x;
        octx.beginPath();
        octx.moveTo(x, 0);
        octx.lineTo(x, height);
        octx.stroke();
      }
    }

    if (track.peaks) {
      octx.strokeStyle = '#6d5ef4';
      octx.lineWidth = 1;
      const numPeaks = track.peaks.length / 2;
      const mid = height / 2;
      for (let i = 0; i < numPeaks; i++) {
        const x = (i / numPeaks) * width;
        const min = track.peaks[i * 2];
        const max = track.peaks[i * 2 + 1];
        octx.beginPath();
        octx.moveTo(x, mid + min * mid);
        octx.lineTo(x, mid + max * mid);
        octx.stroke();
      }
    }

    this._staticBitmap = off;
  }

  _tick = () => {
    if (this._stopped) return;
    const track = this.deck.track;

    if (track && track.name !== this._lastRenderedTrackName) {
      this._renderStatic(track);
      this._lastRenderedTrackName = track.name;
      if (this.bpmEl) {
        const bpm = Math.round(track.bpm);
        this.bpmEl.textContent = track.bpmFallback ? `~${bpm} (est.)` : `${bpm}`;
      }
    } else if (!track && this._lastRenderedTrackName !== null) {
      this._staticBitmap = null;
      this._lastRenderedTrackName = null;
      if (this.bpmEl) this.bpmEl.textContent = '—';
    }

    const { width, height } = this.canvas;
    this.ctx.clearRect(0, 0, width, height);
    if (this._staticBitmap) this.ctx.drawImage(this._staticBitmap, 0, 0);

    // Playhead: re-read currentTime raw every frame, no interpolation — a
    // cue-jump or tempo-synced seek must show as a real snap, not a slide
    // (autoplan Eng review).
    if (track && this.deck.audio.duration) {
      const x = (this.deck.audio.currentTime / this.deck.audio.duration) * width;
      this.ctx.strokeStyle = '#f2f2f5';
      this.ctx.lineWidth = 2;
      this.ctx.beginPath();
      this.ctx.moveTo(x, 0);
      this.ctx.lineTo(x, height);
      this.ctx.stroke();
    }

    if (this.meterFillEl) {
      this.deck.analyser.getByteTimeDomainData(this._meterData);
      let sumSquares = 0;
      for (let i = 0; i < this._meterData.length; i++) {
        const v = (this._meterData[i] - 128) / 128;
        sumSquares += v * v;
      }
      const level = Math.sqrt(sumSquares / this._meterData.length); // 0..1 rms, post-gain — reflects what's actually audible
      this.meterFillEl.style.transform = `scaleX(${Math.min(1, level * 3)})`; // *3: RMS of typical music sits well under 1.0
    }

    requestAnimationFrame(this._tick);
  };
}
