(function () {
  const R = window.RESULTS, O = R.optics, $ = (id) => document.getElementById(id);
  const go = (id) => { const b = document.querySelector('#tabs button[data-id="' + id + '"]'); if (b) b.click(); };

  /* ---- Problem cards ---- */
  const P = [
    ['👁️', 'Eyes need checking', 'Diabetes and other conditions can damage the retina long before a person notices. A photo of the back of the eye (a fundus image) lets a clinician spot it early.', 'The catch: the camera that takes that photo is the bottleneck.'],
    ['⚖️', 'Glass makes it heavy and costly', 'A good retinal camera lens stacks several precision glass elements. Each one must be ground and polished. That adds weight, price and a longer supply chain.', 'Our glass reference design has 4 elements. Our alternative has 2 molded plastic ones.'],
    ['🏘️', 'Screening must reach far', 'Clinics, camps and village health workers need something light, cheap to make in volume and easy to carry.', 'If the lens gets simpler, the camera can get simpler too.'],
    ['🤔', 'The hard question', 'Plastic lenses are blurrier at the edges and change with temperature. Can software win that sharpness back without inventing or hiding disease signs?', 'This is exactly what the prototype tests, including where it fails.']];
  P.forEach((p, i) => { const b = document.createElement('button'); b.className = 'pcard'; b.innerHTML = '<span class="ic">' + p[0] + '</span><b>' + p[1] + '</b><span>Tap to read</span>'; b.onclick = () => { [...$('pcards').children].forEach((c) => c.classList.remove('on')); b.classList.add('on'); $('pdetail').innerHTML = '<h3>' + p[1] + '</h3><p>' + p[2] + '</p><p><b>' + p[3] + '</b></p>'; }; $('pcards').appendChild(b); if (i === 0) b.click(); });

  /* ---- Swap the glass ---- */
  const G = O.glass, Pm = O.polymer, ratio = (O.cost_sensitivity && O.cost_sensitivity.ratio_median) || 0.14, S0 = R.samples[0];
  const stack = (n, col) => { let s = ''; const xs = n === 4 ? [60, 100, 140, 180] : [90, 150]; xs.forEach((x) => { s += '<ellipse cx="' + x + '" cy="75" rx="' + (n === 4 ? 9 : 12) + '" ry="48" fill="' + col + '" fill-opacity=".35" stroke="' + col + '" stroke-width="2.5"/>'; }); for (let i = 0; i < 5; i++) s += '<polyline fill="none" stroke="' + col + '" stroke-width="1.5" opacity=".7" points="5,' + (45 + i * 15) + ' ' + (n === 4 ? 60 : 90) + ',' + (45 + i * 15) + ' 240,75"/>'; return s + '<line x1="240" y1="35" x2="240" y2="115" stroke="#0A1B3F" stroke-width="3"/>' + (n === 4 ? '<circle cx="240" cy="75" r="3" fill="' + col + '"/>' : '<ellipse cx="240" cy="75" rx="6" ry="12" fill="' + col + '" opacity=".5"/>'); };
  const cnt = (el, to, dec, suf) => { const from = parseFloat(el.dataset.v || 0), t0 = performance.now(); el.dataset.v = to; (function f(t) { const k = Math.min(1, (t - t0) / 600); el.textContent = (from + (to - from) * k).toFixed(dec) + (suf || ''); if (k < 1) requestAnimationFrame(f); })(t0); };
  function setSwap(m) {
    const poly = m === 'poly'; [...$('swapSeg').children].forEach((b) => b.classList.toggle('on', b.dataset.m === m));
    $('swapSvg').innerHTML = stack(poly ? 2 : 4, poly ? '#E0407B' : '#0072EF');
    cnt($('swEl'), poly ? Pm.n_el : G.n_el, 0); cnt($('swMass'), poly ? Pm.mass_g : G.mass_g, 2); cnt($('swCost'), poly ? ratio * 100 : 100, 0, '%');
    $('swNote').textContent = poly ? 'Lighter and cheaper on paper, but blurrier at the edges. Costs use placeholder inputs, so treat them as a direction, not a quote.' : 'The reference: sharp across the field, but heavier and costlier to make.';
    if (S0) { $('swImg').src = S0.files[poly ? 'D' : 'B']; $('swCap').textContent = poly ? 'Through the plastic lens, then restored by the AI' : 'Through the glass lens'; }
  }
  [...$('swapSeg').children].forEach((b) => { b.onclick = () => setSwap(b.dataset.m); }); setSwap('glass');


  /* ---- What we found ---- */
  try {
    const C = R.conditions, NI = R.noninferiority, f3 = (x) => x.toFixed(3), IC = R.independent_classifier || {};
    const g = C['B/glass'], c = C['C/nominal'], d = C['D/nominal'], ni = NI['D/nominal vs B/glass'], nc = NI['C/nominal vs B/glass'];
    const indep = IC['D/nominal vs B/glass (independent classifier)'], indepC = IC['C/nominal vs B/glass (independent classifier)'];
    $('foundIntro').textContent = (R.meta.synthetic ? 'Synthetic smoke-test data, not evidence. ' : 'Real ' + R.meta.dataset + ' photos, ' + R.meta.n_test + ' held-out test images. ') + 'The lens blur is simulated. We report what happened, including where the idea falls short.';
    const T = [
      ['Glass lens', f3(g.auc), 'AUC, the reference to beat', '#0072EF'],
      ['Plastic lens, no AI', f3(c.auc), 'AUC · ' + f3(nc.auc_diff) + ' vs glass (' + f3(nc.auc_lo) + ' to ' + f3(nc.auc_hi) + '): ' + (nc.verdict.auc_noninferior ? 'within our 0.05 margin' : 'worse than our 0.05 margin allows'), '#E0407B'],
      ['Plastic + AI restorer', f3(d.auc), 'AUC · ' + f3(ni.auc_diff) + ' vs glass: AUC ' + (ni.verdict.auc_noninferior ? 'within' : 'outside') + ' the margin, but false positives ' + (ni.verdict.fpr_ok ? 'ok' : 'too high'), '#F2A100'],
      ['Independent judge', indep ? f3(indep.auc_diff - (indepC ? indepC.auc_diff : 0)) : '–', 'AUC change from adding the AI, seen by a classifier the AI never trained against. About zero means no proven gain.', '#7A5AF8']];
    $('found').innerHTML = T.map((t) => '<div class="ftile" style="border-top:4px solid ' + t[3] + '"><span>' + t[0] + '</span><b>' + t[1] + '</b><small>' + t[2] + '</small></div>').join('') +
      '<div class="ftile wide"><span>Plain reading</span><small>The simpler lens loses some accuracy. The AI recovers most of it on the classifier it was tuned with, but an independent classifier does not confirm a real gain, and we cannot check lesions without masks. Optics: the plastic design misses our image-quality rules at the edge of the field, while weighing about ' + Math.round(100 * (O.polymer.mass_g / O.glass.mass_g)) + '% of the glass lens.</small></div>';
  } catch (e) { console.error('found', e); }

  /* ---- You be the judge ---- */
  const J = [
    ['Does it still find disease?', 'Open the Evidence tab. It compares glass, plastic and plastic + AI on a held-out test set, with confidence intervals. Small test sets give wide intervals, and we show them.', 'p-evidence'],
    ['Does the AI invent lesions?', 'Open Restorer Study. It counts lesions that were kept, removed, distorted or invented, and uses a second AI judge the restorer never saw.', 'p-study'],
    ['Can plastic lenses really be made?', 'Open Engineering. It checks thickness and slope limits, tolerance yield over 800 simulated builds and five plastic materials.', 'p-eng'],
    ['Is it really cheaper and lighter?', 'Open Cost & Deploy. It shows the whole camera module, not just the lens, so the saving is not overstated.', 'p-cost'],
    ['What if the photo is bad?', 'Open Capture Check and upload a photo. It tells the operator what to fix before any result is trusted.', 'p-capture']];
  J.forEach((j, i) => { const b = document.createElement('button'); b.textContent = j[0]; b.onclick = () => { [...$('judgeSeg').children].forEach((c) => c.classList.toggle('on', c === b)); $('judgeOut').innerHTML = '<p>' + j[1] + '</p><button class="btn">Take me there</button>'; $('judgeOut').querySelector('button').onclick = () => go(j[2]); }; $('judgeSeg').appendChild(b); if (i === 0) b.click(); });

  /* ---- Lenny the guide ---- */
  const TIPS = {
    'p-start': ['Hi, I am Lenny, a very small lens with big opinions. Tap a problem card, flip Glass to Plastic, then click me for the next stop.', 'p-lab'],
    'p-lab': ['Drag the big slider. Watch the picture go from crisp glass to blurry plastic, then back to sharp. Magic? No. Maths.', 'p-optics'],
    'p-optics': ['Move the field slider and watch the red blur patch grow at the edge. Plastic lenses are dramatic like that.', 'p-ai'],
    'p-ai': ['This is the tiny AI. Compare the before and after tiles. Do not trust it blindly, I will not either.', 'p-eng'],
    'p-eng': ['Charts, rules and a pass/fail table. Hover and squint like an engineer, then see how the lens is actually made.', 'p-study'],
    'p-study': ['Here the AI faces the judge. Look for lesions that were kept, lost or invented. Honesty is the best policy.', 'p-evidence'],
    'p-evidence': ['Click the names above the ROC curve to hide lines. Wider bars mean less certainty. Science is humble.', 'p-rain'],
    'p-rain': ['Drag the white lines and watch pink drops get caught. Then slide the disease share in the clinic simulator. Fun, but serious.', 'p-capture'],
    'p-capture': ['Pick a sample photo or upload your own. I will tell you if it is too blurry, dark or off-centre. Rude but helpful.', 'p-cost'],
    'p-cost': ['Money and carbon. Notice the whole camera saves less than the lens alone. I like numbers that tell the truth.', 'p-impact'],
    'p-impact': ['Nearly there. Adjust the sliders and see when plastic breaks even. Then head to the last tab.', 'p-run'],
    'p-run': ['Copy the commands to rerun everything on real data. You made it. Reward yourself with tea.', null]};
  const L = document.createElement('div'); L.id = 'lenny'; document.body.appendChild(L);
  let muted = false, cur = 'p-start';
  const face = '<svg width="38" height="38" viewBox="0 0 40 40"><circle cx="20" cy="20" r="15" fill="#E8F1FF" stroke="#0072EF" stroke-width="3"/><circle cx="14" cy="17" r="3" fill="#0A1B3F"/><circle cx="26" cy="17" r="3" fill="#0A1B3F"/><path d="M12 25 Q20 32 28 25" stroke="#0A1B3F" stroke-width="2.5" fill="none" stroke-linecap="round"/><circle cx="31" cy="10" r="4" fill="#F2A100"/></svg>';
  function draw() {
    const t = TIPS[cur]; L.innerHTML = '';
    if (!muted && t) {
      const b = document.createElement('div'); b.className = 'bub'; b.setAttribute('role', 'status'); b.innerHTML = '<b>Lenny:</b> ' + t[0] + '<div class="acts"></div>'; const a = b.querySelector('.acts');
      if (t[1]) { const n = document.createElement('button'); n.textContent = 'Next stop →'; n.onclick = () => go(t[1]); a.appendChild(n); }
      const h = document.createElement('button'); h.className = 'g'; h.textContent = 'Hide tips'; h.onclick = () => { muted = true; draw(); }; a.appendChild(h); L.appendChild(b);
    }
    const f = document.createElement('button'); f.className = 'face'; f.setAttribute('aria-label', muted ? 'Show tips' : 'Lenny the guide'); f.innerHTML = face; f.onclick = () => { muted = !muted; draw(); }; L.appendChild(f);
  }
  const mo = new MutationObserver(() => { const v = [...document.querySelectorAll('.pane')].find((p) => !p.hidden); if (v && v.id !== cur) { cur = v.id; draw(); } });
  document.querySelectorAll('.pane').forEach((p) => mo.observe(p, { attributes: true, attributeFilter: ['hidden'] }));
  const v0 = [...document.querySelectorAll('.pane')].find((p) => !p.hidden); if (v0) cur = v0.id; draw();
})();
