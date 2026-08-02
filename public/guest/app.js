const form = document.getElementById('request-form');
const submitBtn = document.getElementById('submit-btn');
const result = document.getElementById('result');

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const track = document.getElementById('track').value.trim();
  const requester = document.getElementById('requester').value.trim();
  if (!track) return;

  submitBtn.disabled = true;
  submitBtn.textContent = 'Sending…';
  result.textContent = '';
  result.className = '';

  try {
    const res = await fetch('/api/request', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ track, requester }),
    });
    if (!res.ok) throw new Error('Request failed — try again');

    result.textContent = `Sent! The host will approve "${track}" soon, or it plays automatically in about 90 seconds.`;
    result.className = 'success';
    form.reset();
  } catch (err) {
    result.textContent = err.message || 'Something went wrong — check your connection and try again.';
    result.className = 'error';
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = 'Send request';
  }
});
