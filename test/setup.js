// Minimal Web Audio API + <audio> mocks so MusicEngine (public/host/engine.js)
// can be constructed and driven under jsdom, which implements neither. Real
// AudioContext/HTMLMediaElement behavior (actual playback, real decoding) is
// NOT covered here — these tests target the engine's pure decision logic
// (scoring, beat-grid math, state-machine transitions), not audio output.
import { vi } from 'vitest';

vi.mock('https://esm.sh/web-audio-beat-detector@8', () => ({
  analyze: vi.fn(async () => 120),
}));

// Real AudioParam automation is scheduled and time-based; these tests don't
// drive a real audio clock, so each scheduling method applies its target
// value to `.value` immediately — close enough to let assertions on gain/EQ
// state reflect what the engine actually told the param to do.
class MockAudioParam {
  constructor(value = 0) {
    this.value = value;
  }
  setValueAtTime(value) {
    this.value = value;
    return this;
  }
  linearRampToValueAtTime(value) {
    this.value = value;
    return this;
  }
  exponentialRampToValueAtTime(value) {
    this.value = value;
    return this;
  }
  setValueCurveAtTime(curve) {
    this.value = curve[curve.length - 1];
    return this;
  }
  setTargetAtTime(value) {
    this.value = value;
    return this;
  }
  cancelScheduledValues() {
    return this;
  }
}

class MockAudioNode {
  connect(target) {
    return target ?? this;
  }
  disconnect() {}
}

class MockGainNode extends MockAudioNode {
  constructor() {
    super();
    this.gain = new MockAudioParam(1);
  }
}

class MockBiquadFilterNode extends MockAudioNode {
  constructor() {
    super();
    this.type = 'lowpass';
    this.frequency = new MockAudioParam(0);
    this.Q = new MockAudioParam(1);
    this.gain = new MockAudioParam(0);
  }
}

class MockDelayNode extends MockAudioNode {
  constructor() {
    super();
    this.delayTime = new MockAudioParam(0);
  }
}

class MockAnalyserNode extends MockAudioNode {
  constructor() {
    super();
    this.fftSize = 2048;
  }
  getByteTimeDomainData(arr) {
    arr.fill(128);
  }
}

class MockAudioBufferSourceNode extends MockAudioNode {
  constructor() {
    super();
    this.buffer = null;
    this.loop = false;
    this.loopStart = 0;
    this.loopEnd = 0;
    this.onended = null;
  }
  start() {}
  stop() {
    if (this.onended) this.onended();
  }
}

class MockAudioContext {
  constructor() {
    this._currentTime = 0;
    this.destination = new MockAudioNode();
  }
  get currentTime() {
    return this._currentTime;
  }
  createGain() {
    return new MockGainNode();
  }
  createBiquadFilter() {
    return new MockBiquadFilterNode();
  }
  createDelay() {
    return new MockDelayNode();
  }
  createAnalyser() {
    return new MockAnalyserNode();
  }
  createDynamicsCompressor() {
    return new MockAudioNode();
  }
  createMediaElementSource() {
    return new MockAudioNode();
  }
  createBufferSource() {
    return new MockAudioBufferSourceNode();
  }
  decodeAudioData() {
    return Promise.resolve({});
  }
}

class MockAudioElement extends EventTarget {
  constructor() {
    super();
    this.src = '';
    this.currentTime = 0;
    this.duration = NaN;
    this.playbackRate = 1;
    this.paused = true;
    this.crossOrigin = null;
    this.preservesPitch = true;
    this.mozPreservesPitch = true;
    this.webkitPreservesPitch = true;
    this.oncanplaythrough = null;
    this.onerror = null;
  }
  play() {
    this.paused = false;
    return Promise.resolve();
  }
  pause() {
    this.paused = true;
  }
}

globalThis.AudioContext = MockAudioContext;
globalThis.window.AudioContext = MockAudioContext;
globalThis.Audio = MockAudioElement;
globalThis.window.Audio = MockAudioElement;
