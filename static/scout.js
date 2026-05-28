// Small UI handlers for the Scout home page.
// Extracted into a file (not inline) so the strict Content-Security-Policy
// can keep script-src locked to 'self' + unpkg, with no 'unsafe-inline'.

// Prompt chips in the hero pre-fill the question input.
document.addEventListener('click', function (e) {
  var chip = e.target.closest('.prompt-chip');
  if (!chip) return;
  var input = document.querySelector('input[name="question"]');
  if (!input) return;
  input.value = chip.dataset.q;
  input.focus();
});

// Clicking the "Scout" brand returns to the empty/welcome state.
var brand = document.getElementById('brand-link');
if (brand) {
  brand.addEventListener('click', function () {
    var result = document.getElementById('result');
    if (result) result.innerHTML = '';
    var input = document.querySelector('input[name="question"]');
    if (input) { input.value = ''; input.focus(); }
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
}

// ---------------------------------------------------------------------------
// Streaming answer body
//
// POST /ask returns an answer shell with a `.answer-body[data-stream-url]`
// element. This script:
//   1. Streams the Claude response from /ask/stream/{id} into the element as
//      plain text (textContent, so any markup in the model's output is inert
//      during streaming — the final render goes through linkify_citations
//      server-side which has already escaped its input).
//   2. On stream end, GETs /ask/finalize/{id} which returns the validated +
//      linkified HTML, the rendered sources sidebar, and a list of warnings
//      from the citation validator. We swap those into place.
// ---------------------------------------------------------------------------

function _renderWarnings(warnings) {
  if (!warnings || !warnings.length) return null;
  var box = document.createElement('div');
  box.className = 'answer-warnings';
  for (var i = 0; i < warnings.length; i++) {
    var item = document.createElement('div');
    item.className = 'answer-warning';
    item.textContent = warnings[i];
    box.appendChild(item);
  }
  return box;
}

async function _streamAnswer(bodyEl) {
  if (bodyEl.dataset.streamingStarted === '1') return;
  bodyEl.dataset.streamingStarted = '1';

  var streamUrl = bodyEl.dataset.streamUrl;
  var finalizeUrl = bodyEl.dataset.finalizeUrl;
  if (!streamUrl || !finalizeUrl) return;

  var block = bodyEl.closest('.answer-block');

  try {
    var resp = await fetch(streamUrl, { credentials: 'same-origin' });
    if (!resp.ok || !resp.body) {
      throw new Error('stream request failed: ' + resp.status);
    }
    var reader = resp.body.getReader();
    var decoder = new TextDecoder('utf-8');
    bodyEl.textContent = '';
    while (true) {
      var step = await reader.read();
      if (step.done) break;
      bodyEl.textContent += decoder.decode(step.value, { stream: true });
    }
    // flush any remaining bytes in the decoder
    bodyEl.textContent += decoder.decode();

    var finResp = await fetch(finalizeUrl, { credentials: 'same-origin' });
    if (!finResp.ok) throw new Error('finalize request failed: ' + finResp.status);
    var data = await finResp.json();

    bodyEl.classList.remove('streaming');
    // Server-rendered HTML — already escaped + linkified by linkify_citations.
    bodyEl.innerHTML = data.answer_html;

    if (block) {
      var slot = block.querySelector('.sources-slot');
      if (slot && data.sources_html) {
        var tmp = document.createElement('div');
        tmp.innerHTML = data.sources_html.trim();
        var rendered = tmp.firstElementChild;
        if (rendered) slot.replaceWith(rendered);
      }
      var warnSlot = block.querySelector('.answer-warnings-slot');
      var warnEl = _renderWarnings(data.warnings);
      if (warnSlot && warnEl) {
        warnSlot.replaceWith(warnEl);
      }
    }
  } catch (e) {
    bodyEl.classList.remove('streaming');
    bodyEl.textContent = 'Error: ' + e.message;
  }
}

// HTMX swaps the answer shell into #result on POST /ask. We pick up any
// streaming bodies in the new content and connect them.
document.body.addEventListener('htmx:afterSwap', function (e) {
  var target = e.target;
  if (!target || !target.querySelectorAll) return;
  var bodies = target.querySelectorAll('.answer-body[data-stream-url]');
  for (var i = 0; i < bodies.length; i++) {
    _streamAnswer(bodies[i]);
  }
});
