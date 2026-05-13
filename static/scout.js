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
