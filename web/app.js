(function () {
  const R = window.RESULTS, O = R.optics, C = R.conditions;
  const $ = (id) => document.getElementById(id);
  const f3 = (x) => x.toFixed(3), pct = (x) => (100 * x).toFixed(1) + '%';
  const COL = { A: '#7A5AF8', B: '#0072EF', C: '#E0407B', D: '#F2A100' };
  const SHORT = { A: 'Original', B: 'Glass', C: 'Polymer', D: 'Polymer + AI' };
  const KEY = { A: 'A/ref', B: 'B/glass', C: 'C/nominal', D: 'D/nominal' };
  const thr = R.meta.threshold_glass;
  { const b = $('banner'); b.hidden = false;
    if (R.meta.synthetic) b.textContent = 'Smoke-test data: SYNTHETIC images prove the pipeline runs. They are not scientific evidence. Run on RFMiD / IDRiD for real numbers.';
    else { b.classList.add('real'); b.textContent = 'Real data: ' + R.meta.dataset + ' fundus photos. ' + R.meta.n_test + ' held-out test images (' + R.meta.n_pos_test + ' abnormal, ' + (R.meta.n_test - R.meta.n_pos_test) + ' normal). Lens blur is simulated, not from a physical lens. Images: RFMiD, Pachade et al., 2021.'; } }

  /* ---------- Tabs ---------- */
  const panes = [...document.querySelectorAll('.pane')], tabs = $('tabs');
  function show(id) {
    panes.forEach((p) => { p.hidden = p.id !== id; });
    [...tabs.children].forEach((b) => b.classList.toggle('on', b.dataset.id === id));
    try { history.replaceState(null, '', '#' + id); } catch (e) {}
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
  panes.forEach((p) => { const b = document.createElement('button'); b.textContent = p.dataset.title; b.dataset.id = p.id; b.setAttribute('role', 'tab'); b.onclick = () => show(p.id); tabs.appendChild(b); });
  document.querySelectorAll('[data-go]').forEach((b) => { b.onclick = () => show(b.dataset.go); });
  const startId = location.hash.slice(1);
  show(panes.some((p) => p.id === startId) ? startId : 'p-start');

  /* ---------- Hero + how it works ---------- */
  $('heroSvg').innerHTML = '<defs><marker id="ar" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8z" fill="#7FB6FF"/></marker></defs>' +
    '<text x="14" y="26" fill="#9CC4FF" font-size="13" font-weight="700">GLASS · 4 ELEMENTS</text>' +
    [0, 1, 2, 3, 4].map((i) => '<polyline class="flowanim" fill="none" stroke="#5AA5FF" stroke-width="2" points="20,' + (50 + i * 14) + ' 150,' + (50 + i * 14) + ' 250,' + (70 + i * 6) + ' 340,84"/>').join('') +
    [120, 150, 180, 210].map((x) => '<ellipse cx="' + x + '" cy="82" rx="7" ry="34" fill="#0072EF" fill-opacity=".35" stroke="#7FB6FF" stroke-width="2"/>').join('') +
    '<line x1="340" y1="48" x2="340" y2="120" stroke="#fff" stroke-width="3"/><circle cx="340" cy="84" r="4" fill="#5AA5FF"/><text x="352" y="88" fill="#C7DEFF" font-size="12">sharp</text>' +
    '<text x="14" y="176" fill="#FFB3CF" font-size="13" font-weight="700">POLYMER · 2 ELEMENTS + AI</text>' +
    [0, 1, 2, 3, 4].map((i) => '<polyline class="flowanim" fill="none" stroke="#FF7AA8" stroke-width="2" points="20,' + (200 + i * 14) + ' 150,' + (200 + i * 14) + ' 250,' + (222 + i * 6) + ' 340,236"/>').join('') +
    [140, 200].map((x) => '<ellipse cx="' + x + '" cy="232" rx="9" ry="34" fill="#E0407B" fill-opacity=".35" stroke="#FF9AC0" stroke-width="2"/>').join('') +
    '<ellipse cx="340" cy="236" rx="8" ry="16" fill="#FF7AA8" fill-opacity=".5"/><line x1="340" y1="196" x2="340" y2="274" stroke="#fff" stroke-width="3"/><text x="352" y="240" fill="#FFD0E2" font-size="12">soft</text>' +
    '<rect x="400" y="212" width="100" height="48" rx="6" fill="#F2A100"/><text x="450" y="241" fill="#00143C" font-size="15" font-weight="800" text-anchor="middle">AI restorer</text><line x1="360" y1="236" x2="398" y2="236" stroke="#fff" stroke-width="2" marker-end="url(#ar)"/>';
  const STEPS = [
    ['1', 'Needs', 'what must be seen', 'We start from the task, not the optics. A frozen classifier trained on glass images shows, via Grad-CAM, which parts of the image matter. That sets field-position weights for the lens design.'],
    ['2', 'Design', 'lens by optimiser', 'An open-source differentiable ray tracer (Optiland) optimises a 2-element aspheric polymer lens for the weighted blur. A material sweep, tolerance Monte Carlo and element X-ray check it can really be molded.'],
    ['3', 'Render', 'simulate the camera', 'Images are blurred by the simulated spot size, with random tolerance, temperature and noise. A separate renderer with a different blur shape is kept only for testing.'],
    ['4', 'Restore', 'small AI on the phone', 'A 119k-parameter U-Net sharpens the polymer image. Its loss also asks the frozen classifier to still see lesions, so it protects what matters rather than prettiness.'],
    ['5', 'Verify', 'one honest test', 'One held-out evaluation with a threshold fixed on glass. We report confidence intervals, non-inferiority, baselines (retrain on polymer, Wiener) and lesion-level checks, including failures.']
  ];
  const fl = $('flow'), fd = $('flowDetail');
  STEPS.forEach((s, i) => { const b = document.createElement('button'); b.innerHTML = '<b>' + s[0] + '</b><span>' + s[1] + '</span><small>' + s[2] + '</small>'; b.onclick = () => pick(i); fl.appendChild(b); });
  function pick(i) { [...fl.children].forEach((b, k) => b.classList.toggle('on', k === i)); fd.innerHTML = '<h3>Step ' + STEPS[i][0] + ': ' + STEPS[i][1] + '</h3><p class="muted" style="margin:0">' + STEPS[i][3] + '</p>'; }
  pick(0);

  /* ---------- Lens Lab ---------- */
  let cur = 0, sample = 0, front = 'imgA', ood = false;
  const conds = ['A', 'B', 'C', 'D'], seg = $('seg'), th = $('thumbs');
  const DESC = { A: 'The untouched image: the upper reference.', B: 'As a 4-element glass lens would deliver it (simulated). Sets the bar and the decision line.', C: 'Through the 2-element polymer lens with no correction.', D: 'Polymer image after the small restorer.' };
  conds.forEach((c, i) => { const b = document.createElement('button'); b.textContent = SHORT[c]; b.onclick = () => { $('slider').value = i; setCond(i); }; seg.appendChild(b); });
  R.samples.forEach((s, i) => { const b = document.createElement('button'); const im = document.createElement('img'); im.src = s.files.A; im.alt = s.kind; b.title = s.kind; b.appendChild(im); b.onclick = () => { sample = i; setCond(cur); blurExp(); }; th.appendChild(b); });
  $('kinds').textContent = '4 random + 2 worst failures of the restorer (not cherry-picked)';
  $('slider').oninput = (e) => setCond(+e.target.value);
  $('oodToggle').onchange = (e) => { ood = e.target.checked; setCond(cur); };
  function setCond(i) {
    cur = i; const c = conds[i], s = R.samples[sample];
    const src = (ood && (c === 'C' || c === 'D') && s.files.X && c === 'C') ? s.files.X : s.files[c];
    const showEl = $(front === 'imgA' ? 'imgB' : 'imgA'), hideEl = $(front);
    showEl.src = src; showEl.style.opacity = 1; hideEl.style.opacity = 0; front = showEl.id;
    $('stamp').textContent = SHORT[c] + (ood && c === 'C' ? ' · independent renderer' : '') + ' · ' + s.kind + ' sample';
    [...seg.children].forEach((b, k) => b.classList.toggle('on', k === i));
    [...th.children].forEach((b, k) => b.classList.toggle('on', k === sample));
    const l = $('condLabel'); l.textContent = c === 'A' ? 'Measured' : 'Simulated'; l.className = 'lab ' + (c === 'A' ? 'm' : 's');
    $('condName').textContent = SHORT[c]; $('condDesc').textContent = DESC[c];
    $('thr').style.left = Math.max(1, 100 * thr) + '%';
    const sc = s.scores[c === 'A' ? 'B' : c];
    if (c === 'A' || sc === undefined) { $('mark').style.opacity = 0; $('verdict').textContent = 'Reference image'; $('verdict').style.color = '#5A6478'; $('truth').textContent = 'Truth: ' + (s.truth ? 'abnormal' : 'normal') + '.'; return; }
    $('mark').style.opacity = 1; $('mark').style.left = (100 * sc) + '%';
    const fl2 = sc > thr;
    $('verdict').textContent = fl2 ? 'Flagged for review' : 'Not flagged'; $('verdict').style.color = fl2 ? '#C62828' : '#00875A';
    $('truth').textContent = 'Score ' + sc.toFixed(2) + ' vs line ' + thr.toFixed(4) + (thr < 0.02 ? ' (the smoke-test classifier is saturated, so the line sits near 0)' : '') + '. Truth: ' + (s.truth ? 'abnormal' : 'normal') + ((fl2 === (s.truth === 1)) ? ', correct.' : ', WRONG.');
  }
  setCond(0);
  const interp = (arr, t) => { const x = t * (arr.length - 1), i = Math.min(arr.length - 2, Math.floor(x)); return arr[i] + (arr[i + 1] - arr[i]) * (x - i); };
  function blurExp() {
    const t = +$('fieldSlider').value / 100, s = R.samples[sample], pxum = 10, disp = $('bGlass').clientWidth || 200, k = disp / R.meta.size;
    const g = interp(O.glass.spot_um, t), p = interp(O.polymer.spot_um, t);
    $('bGlass').src = $('bPoly').src = s.files.A;
    $('bGlass').style.filter = 'blur(' + (g / pxum * k).toFixed(2) + 'px)'; $('bPoly').style.filter = 'blur(' + (p / pxum * k).toFixed(2) + 'px)';
    $('bGlassCap').textContent = 'Glass · spot ' + g.toFixed(1) + ' µm'; $('bPolyCap').textContent = 'Polymer · spot ' + p.toFixed(1) + ' µm';
  }
  $('fieldSlider').oninput = blurExp; blurExp();
  new MutationObserver(blurExp).observe($('p-lab'), { attributes: true, attributeFilter: ['hidden'] });

  /* ---------- Optics: rays ---------- */
  let field = 0; const FN = O.field_labels, maxSpot = Math.max(...O.polymer.spot_um, ...O.glass.spot_um);
  function rays(id, lensX, spot, color) {
    const fy = 110 + field * 48, off = [-36, -18, 0, 18, 36];
    let g = '<rect width="460" height="220" fill="#00143C"/><line x1="20" y1="110" x2="440" y2="110" stroke="#1B3A7A" stroke-dasharray="3 5"/>';
    const blur = 1 + spot / maxSpot * 16;
    off.forEach((o) => { const yin = 110 + o + field * 12, ylens = yin - field * 6; g += '<polyline class="flowanim" fill="none" stroke="' + color + '" stroke-width="2" points="20,' + yin + ' ' + lensX[0] + ',' + ylens + ' ' + lensX[lensX.length - 1] + ',' + (ylens + (fy - ylens) * .35) + ' 400,' + (fy + o / 36 * blur) + '"/>'; });
    lensX.forEach((x) => { g += '<ellipse cx="' + x + '" cy="' + (110 + field * 6) + '" rx="8" ry="52" fill="#0072EF" fill-opacity=".3" stroke="#7FB6FF" stroke-width="2"/>'; });
    g += '<line x1="400" y1="40" x2="400" y2="190" stroke="#fff" stroke-width="3"/><ellipse cx="400" cy="' + fy + '" rx="' + Math.max(2, blur / 3) + '" ry="' + blur + '" fill="#FF4D6D" fill-opacity=".75"/>';
    g += '<text x="20" y="206" fill="#9CC4FF" font-size="12">spot (RMS radius): ' + spot.toFixed(1) + ' µm · schematic, spot drawn to relative scale</text>';
    $(id).innerHTML = g;
  }
  function drawRays() { rays('rayGlass', [150, 190, 230, 270], O.glass.spot_um[field], '#5AA5FF'); rays('rayPoly', [190, 250], O.polymer.spot_um[field], '#FF7AA8'); [...$('fieldSeg').children].forEach((b, k) => b.classList.toggle('on', k === field)); }
  FN.forEach((n, i) => { const b = document.createElement('button'); b.textContent = n; b.onclick = () => { field = i; drawRays(); }; $('fieldSeg').appendChild(b); });
  drawRays();
  $('spotBars').innerHTML = FN.map((n, i) => '<div class="row bars"><div class="t"><span>' + n + '</span></div><div class="b cg" style="width:' + Math.max(10, 100 * O.glass.spot_um[i] / maxSpot) + '%">Glass ' + O.glass.spot_um[i].toFixed(1) + ' µm</div><div class="b cp" style="width:' + Math.max(10, 100 * O.polymer.spot_um[i] / maxSpot) + '%">Polymer ' + O.polymer.spot_um[i].toFixed(1) + ' µm</div></div>').join('') + '<span class="lab s">Simulated · ' + (O.polymer.aspheric ? 'aspheric' : 'spherical') + ' polymer</span>';

  const Y = O.yield_mc && O.yield_mc.polymer_molded, fm = O.floor_multiples || [];
  if (Y) {
    $('yield').innerHTML = '<p class="small">Pre-declared floor: 2× glass blur. 90% yield is reached only at <b>' + O.required_floor_multiple_for_polymer + '×</b>. We did not move the floor.</p>' + fm.map((m) => { const gq = Y[m] || Y[String(m)]; if (!gq) return ''; const v = gq['1'] !== undefined ? gq['1'] : Object.values(gq)[0]; return '<div class="row bars" title="' + Math.round(v * 100) + '% of simulated lenses pass"><div class="t"><span>Floor ' + m + '× glass</span></div><div class="b ' + (v >= .9 ? 'cm' : 'cp') + '" style="width:' + Math.max(9, v * 100) + '%">' + Math.round(v * 100) + '%</div></div>'; }).join('') + '<span class="lab s">Simulated · tolerance grades ASSUMED</span>';
  }
  const X = O.xray;
  if (X) { const mxg = Math.max(...X.map((e) => e.spot_increase_um || 0)) || 1; $('xray').innerHTML = '<div class="xr">' + X.map((e) => '<div><i class="' + (/low/.test(e.flag || '') ? 'low' : '') + '" style="height:' + Math.max(4, 120 * (e.spot_increase_um || 0) / mxg).toFixed(0) + 'px" title="' + (e.flag || '') + '"></i>El ' + e.element + '<br>' + e.material + '<br><b>' + (e.spot_increase_um || 0).toFixed(0) + ' µm</b></div>').join('') + '</div><p class="small muted">Pink = low value for its cost: a candidate to drop or merge.</p><span class="lab s">Simulated</span>'; }
  const sw = R.meta.saliency_weights;
  $('sal').innerHTML = sw ? '<p class="small">Design weights centre / half-field / edge: <b>' + sw.map((v) => v.toFixed(2)).join(' / ') + '</b>.<br>Higher weight makes the optimiser work harder there.</p><span class="lab m">Measured</span>' : '';

  /* ---------- Restorer ---------- */
  $('params').textContent = R.meta.restorer_params.toLocaleString();
  const s0 = R.samples[0];
  $('cmp').innerHTML = [['A', 'Original'], ['B', 'Glass'], ['C', 'Polymer'], ['D', 'Restored']].map(([c, n]) => '<figure><img src="' + s0.files[c] + '" alt="' + n + '"><figcaption>' + n + '</figcaption></figure>').join('');
  $('cmpNote').textContent = 'Sample #1 (' + s0.kind + '). Pick others in the Lens Lab.';
  const L = (k) => (R.fidelity[k] || {}).lesion;
  const lk = ['raw/nominal', 'restored/nominal', 'raw/ood', 'restored/ood'].filter(L);
  $('lesion').innerHTML = lk.length ? lk.map((k) => { const l = L(k), good = k.startsWith('restored'); return '<div class="row"><div class="t"><span><b>' + (good ? 'Restored' : 'Raw polymer') + '</b> · ' + (k.endsWith('ood') ? 'independent renderer' : 'nominal') + '</span></div><div class="bars"><div class="b ' + (good ? 'cm' : 'cx') + '" style="width:' + Math.max(8, l.lesion_recall * 100) + '%">lesions found ' + pct(l.lesion_recall) + '</div></div><div class="small muted">contrast kept ' + f3(l.contrast_retention) + ' · invented blobs per image ' + (+l.hallucinated_blobs).toFixed(1) + '</div></div>'; }).join('') + '<span class="lab m">Measured (synthetic masks)</span>' : '<p class="muted">No lesion masks in this dataset. IDRiD provides them.</p>';
  $('latency').innerHTML = 'Restorer takes <b>' + R.meta.restorer_ms_per_image + ' ms</b> per ' + R.meta.size + '×' + R.meta.size + ' image on the build machine (' + R.meta.restorer_ms_label + ').';

  /* ---------- Evidence ---------- */
  const sb = $('score');
  [['A', 'Original, frozen classifier', 'Measured'], ['B', 'Glass lens', 'Simulated'], ['C', 'Polymer, raw', 'Simulated'], ['D', 'Polymer + restorer', 'Simulated']].forEach(([c, nm, lab]) => {
    const r = C[KEY[c]]; if (!r) return; const lo = (r.auc_ci[0] - .5) / .5 * 100, hi = (r.auc_ci[1] - .5) / .5 * 100, fid = R.fidelity[{ B: 'glass/nominal', C: 'raw/nominal', D: 'restored/nominal' }[c]];
    const d = document.createElement('div'); d.className = 'sc';
    d.innerHTML = '<span class="lab ' + (lab === 'Measured' ? 'm' : 's') + '">' + lab + '</span><div class="n2" style="color:' + COL[c] + '">' + f3(r.auc) + '</div><div class="nm">AUC · ' + nm + '</div><div class="ci"><i style="left:' + Math.max(0, lo) + '%;width:' + Math.max(2, Math.min(100, hi) - Math.max(0, lo)) + '%;background:' + COL[c] + '"></i></div><div class="small muted">95% CI ' + f3(r.auc_ci[0]) + ' to ' + f3(r.auc_ci[1]) + '</div><dl><dt>Sensitivity</dt><dd>' + pct(r.sens_at_fixed_thr) + '</dd><dt>False positives</dt><dd>' + pct(r.fpr_on_normal) + '</dd>' + (fid ? '<dt>PSNR</dt><dd>' + fid.psnr.toFixed(1) + ' dB</dd>' : '') + '</dl>';
    sb.appendChild(d);
  });
  $('verdictText').textContent = 'Test set: ' + R.meta.n_test + ' images (' + R.meta.n_pos_test + ' abnormal). Wide intervals mean the test is too small to decide. That is shown, not hidden.';
  const NAMES = { C: 'Raw polymer', D: 'Polymer + restorer (ours)', E: 'Baseline: retrain classifier', W: 'Baseline: Wiener', Dfid: 'Ablation: fidelity-only' };
  let rows = '<table class="tb"><tr><th>Pipeline</th><th>Renderer</th><th>AUC diff vs glass (lower bound)</th><th>Verdict</th></tr>';
  Object.keys(R.noninferiority).forEach((k) => { const v = R.noninferiority[k], p = k.split('/')[0], cnd = k.split('/')[1].split(' ')[0]; rows += '<tr><td>' + (NAMES[p] || p) + '</td><td>' + (cnd === 'ood' ? 'independent' : 'nominal') + '</td><td>' + (v.auc_diff >= 0 ? '+' : '') + f3(v.auc_diff) + ' (' + f3(v.auc_lo) + ')</td><td style="color:' + (v.verdict.pass_ ? '#00875A' : '#B45F06') + ';font-weight:700">' + (v.verdict.pass_ ? 'non-inferior' : 'not shown') + '</td></tr>'; });
  $('niTable').innerHTML = rows + '</table>';
  $('niNote').textContent = R.rule + '.';
  const vb = R.restorer_vs_best_baseline;
  $('vsBase').textContent = Object.keys(vb).map((k) => (k === 'ood' ? 'Independent renderer' : 'Nominal renderer') + ': restorer vs best baseline (' + NAMES[vb[k].best_baseline] + ') AUC diff ' + (vb[k].auc_diff >= 0 ? '+' : '') + f3(vb[k].auc_diff) + ', 95% CI ' + f3(vb[k].lo) + ' to ' + f3(vb[k].hi) + (vb[k].restorer_significantly_better ? ', significantly better.' : vb[k].restorer_significantly_worse ? ', significantly worse.' : ', no significant difference.')).join(' ');
  const on = { A: true, B: true, C: true, D: true };
  function drawRoc() {
    const W = 440, P = 46, S = W - P - 16, x = (v) => P + v * S, y = (v) => P - 30 + (1 - v) * S; let g = '';
    [0, .25, .5, .75, 1].forEach((t) => { g += '<line x1="' + x(0) + '" x2="' + x(1) + '" y1="' + y(t) + '" y2="' + y(t) + '" stroke="#E3E7EF"/><line y1="' + y(0) + '" y2="' + y(1) + '" x1="' + x(t) + '" x2="' + x(t) + '" stroke="#E3E7EF"/><text x="' + (P - 8) + '" y="' + (y(t) + 4) + '" fill="#5A6478" font-size="12" text-anchor="end">' + t + '</text><text x="' + x(t) + '" y="' + (y(0) + 18) + '" fill="#5A6478" font-size="12" text-anchor="middle">' + t + '</text>'; });
    g += '<line x1="' + x(0) + '" y1="' + y(0) + '" x2="' + x(1) + '" y2="' + y(1) + '" stroke="#9AA3B5" stroke-dasharray="5 5"/>';
    conds.forEach((c) => { const r = R.roc[KEY[c]]; if (!on[c] || !r) return; g += '<polyline fill="none" stroke="' + COL[c] + '" stroke-width="3" stroke-linejoin="round" points="' + r.fpr.map((f, i) => x(f) + ',' + y(r.tpr[i])).join(' ') + '"/>'; });
    $('roc').innerHTML = g + '<text x="' + x(.5) + '" y="' + (y(0) + 38) + '" fill="#5A6478" font-size="13" text-anchor="middle">False-positive rate (x) vs sensitivity (y)</text>';
  }
  conds.forEach((c) => { const b = document.createElement('button'); b.textContent = SHORT[c]; b.style.borderColor = COL[c]; b.style.color = COL[c]; b.onclick = () => { on[c] = !on[c]; b.style.opacity = on[c] ? 1 : .35; drawRoc(); }; $('rocLegend').appendChild(b); });
  drawRoc();

  /* ---------- Impact ---------- */
  const bar = (label, val, max, cls, txt) => '<div class="row bars"><div class="t"><span>' + label + '</span></div><div class="b ' + cls + '" style="width:' + Math.max(8, 100 * val / max) + '%">' + txt + '</div></div>';
  $('trade').innerHTML = bar('Optical elements, glass', O.glass.n_el, 4, 'cg', O.glass.n_el) + bar('Optical elements, polymer', O.polymer.n_el, 4, 'cp', O.polymer.n_el) + bar('Mass, glass = 100', 100, 100, 'cg', '100') + bar('Mass, polymer', O.mass_rel, 100, 'cp', O.mass_rel.toFixed(1)) + bar('Mass, polymer with mount', O.mass_rel_with_mount, 100, 'cp', O.mass_rel_with_mount.toFixed(1)) + '<span class="lab s">Simulated</span>';
  const cs = O.cost_sensitivity || {}, be = cs.poly_cost_multiplier_to_break_even || {}, ratio = cs.ratio_median || 0.14;
  $('cost').innerHTML = '<p class="small">At our placeholder inputs polymer costs about <b>' + (ratio * 100).toFixed(0) + '%</b> of glass. It stops being cheaper if its cost is <b>' + (be.conservative ? be.conservative.toFixed(1) : '?') + '×</b> (conservative) to <b>' + (be.median ? be.median.toFixed(1) : '?') + '×</b> (median) the placeholder.</p><span class="lab e">Estimated · placeholder inputs</span>';
  $('mul').oninput = (e) => { const m = +e.target.value; $('mulv').textContent = m.toFixed(1) + '×'; const r = ratio * m; $('mulout').textContent = 'Polymer ≈ ' + (r * 100).toFixed(0) + '% of glass cost: ' + (r < 1 ? 'still cheaper' : 'NOT cheaper'); $('mulout').style.color = r < 1 ? '#00875A' : '#C62828'; };
  $('mul').oninput({ target: $('mul') });
})();
