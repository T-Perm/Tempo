// Purely cosmetic: drives the rotary-knob pointer angle and jog-wheel spin
// from live slider values. Polls on rAF instead of 'input' listeners because
// resetMixerSliders() in app.js sets .value directly, which doesn't fire
// 'input' — polling stays correct regardless of how the value changed.

const knobs = Array.from(document.querySelectorAll('.knob'));
const jogwheels = Array.from(document.querySelectorAll('.jogwheel'));

const KNOB_MIN_DEG = -135;
const KNOB_MAX_DEG = 135;

function syncKnobs() {
  for (const knob of knobs) {
    const input = knob.querySelector('.knob-input');
    const pointer = knob.querySelector('.knob-pointer');
    if (!input || !pointer) continue;
    const min = Number(input.min);
    const max = Number(input.max);
    const frac = max > min ? (Number(input.value) - min) / (max - min) : 0.5;
    const deg = KNOB_MIN_DEG + frac * (KNOB_MAX_DEG - KNOB_MIN_DEG);
    pointer.style.transform = `translateX(-50%) rotate(${deg}deg)`;
    knob.style.setProperty('--knob-frac', String(frac));
  }
  requestAnimationFrame(syncKnobs);
}

let spinDeg = 0;
function spinJogwheels() {
  spinDeg = (spinDeg + 0.35) % 360;
  for (const wheel of jogwheels) {
    const ring = wheel.querySelector('.jogwheel-ring');
    if (ring) ring.style.transform = `rotate(${spinDeg}deg)`;
  }
  requestAnimationFrame(spinJogwheels);
}

requestAnimationFrame(syncKnobs);
requestAnimationFrame(spinJogwheels);
