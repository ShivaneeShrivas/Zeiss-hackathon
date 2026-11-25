(function () {
  const R = window.RESULTS, CS = R.case_scores; if (!CS) return;
  const $ = (id) => document.getElementById(id), NS = 'http://www.w3.org/2000/svg';
  const ROWS = [['B/glass', 'Glass lens', '#7FB6FF'], ['C/nominal', 'Plastic lens, raw', '#FF7AA8'], ['D/nominal', 'Plastic + AI restorer', '#F2A100'], ['E/nominal', 'Plastic, classifier retrained', '#7BE0B0']].filter((r) => CS.scores[r[0]]);
  const y = CS.truth, N = y.length, pos = y.reduce((a, b) => a + b, 0), neg = N - pos;
  const X0 = 190, X1 = 880, RH = 110, TOP = 20, LO = -9.2, HI = 9.2;
  const lg = (p) => { p = Math.min(1 - 1e-4, Math.max(1e-4, p)); return Math.log(p / (1 - p)); };
  const px = (p) => X0 + ((lg(p) - LO) / (HI - LO)) * (X1 - X0), inv = (x) => { const l = LO + ((x - X0) / (X1 - X0)) * (HI - LO); return 1 / (1 + Math.exp(-l)); };
  const thr = {}; ROWS.forEach((r) => { thr[r[0]] = CS.thresholds[r[0]] != null ? CS.thresholds[r[0]] : 0.5; });
  const jit = (i) => ((Math.sin(i * 12.9898) * 43758.5453) % 1 + 1) % 1;
  const svg = $('rainSvg'), dots = {}, lines = {}, texts = {};
  const mk = (t, a) => { const e = document.createElementNS(NS, t); Object.keys(a).forEach((k) => e.setAttribute(k, a[k])); return e; };
  ROWS.forEach((r, ri) => {
    const top = TOP + ri * RH;
    svg.appendChild(mk('rect', { x: 0, y: top, width: 900, height: RH - 6, fill: ri % 2 ? '#0E2250' : '#112758', opacity: 0.6 }));
    const lb = mk('text', { x: 12, y: top + 34, fill: r[2], 'font-size': 15, 'font-weight': 700 }); lb.textContent = r[1]; svg.appendChild(lb);
    texts[r[0]] = mk('text', { x: 12, y: top + 56, fill: '#C7DEFF', 'font-size': 12 }); svg.appendChild(texts[r[0]]);
    const ax = mk('text', { x: X0, y: top + RH - 12, fill: '#7C93C0', 'font-size': 10 }); ax.textContent = 'scored normal'; svg.appendChild(ax);
    const ay = mk('text', { x: X1, y: top + RH - 12, fill: '#7C93C0', 'font-size': 10, 'text-anchor': 'end' }); ay.textContent = 'scored abnormal'; svg.appendChild(ay);
    dots[r[0]] = y.map((t, i) => { const c = mk('circle', { cx: px(CS.scores[r[0]][i]), cy: top - 10, r: 5, fill: t ? '#FF5C93' : '#4DA3FF', 'fill-opacity': 0.85, stroke: '#fff', 'stroke-width': 0.6 }); c.dataset.top = top + (t ? 26 : 62) + (jit(i) - 0.5) * 22; const tt = mk('title', {}); tt.textContent = 'Case ' + (i + 1) + ' · truth: ' + (t ? 'abnormal' : 'normal') + ' · score ' + CS.scores[r[0]][i].toFixed(3); c.appendChild(tt); svg.appendChild(c); return c; });
    const g = mk('g', { style: 'cursor:ew-resize' }); const ln = mk('line', { y1: top + 4, y2: top + RH - 22, stroke: '#fff', 'stroke-width': 3 }), hd = mk('circle', { r: 9, cy: top + 6, fill: '#fff', stroke: r[2], 'stroke-width': 3 }); const hit = mk('rect', { y: top, height: RH - 6, width: 26, fill: 'transparent', 'x': -13 });
    g.appendChild(hit); g.appendChild(ln); g.appendChild(hd); svg.appendChild(g); lines[r[0]] = { g, ln, hd, hit };
    let drag = false; const mv = (e) => { const b = svg.getBoundingClientRect(), x = (e.clientX - b.left) / b.width * 900; thr[r[0]] = inv(Math.max(X0, Math.min(X1, x))); upd(); };
    g.addEventListener('pointerdown', (e) => { drag = true; g.setPointerCapture(e.pointerId); mv(e); }); g.addEventListener('pointermove', (e) => { if (drag) mv(e); }); g.addEventListener('pointerup', () => { drag = false; });
    // clicking anywhere in the row also moves the line
    svg.appendChild(mk('rect', { x: X0, y: top + 70, width: X1 - X0, height: 30, fill: 'transparent', style: 'cursor:pointer' })).addEventListener('click', (e) => { const b = svg.getBoundingClientRect(); thr[r[0]] = inv(Math.max(X0, Math.min(X1, (e.clientX - b.left) / b.width * 900))); upd(); });
    svg.appendChild(g);
  });
  svg.setAttribute('viewBox', '0 0 900 ' + (TOP + ROWS.length * RH + 6));
  const st = {};
  function stat(k) { const s = CS.scores[k], t = thr[k]; let tp = 0, fp = 0; s.forEach((v, i) => { if (v >= t) { if (y[i]) tp++; else fp++; } }); return { tp, fp, sens: pos ? tp / pos : 0, fpr: neg ? fp / neg : 0 }; }
  function upd() {
    ROWS.forEach((r) => { const k = r[0], x = px(thr[k]), a = lines[k]; a.ln.setAttribute('x1', x); a.ln.setAttribute('x2', x); a.hd.setAttribute('cx', x); a.hit.setAttribute('x', x - 13); st[k] = stat(k); texts[k].textContent = 'catches ' + st[k].tp + '/' + pos + ' · false alarms ' + st[k].fp + '/' + neg;
      dots[k].forEach((c, i) => { const hit = CS.scores[k][i] >= thr[k]; c.setAttribute('r', hit ? 6.5 : 4.5); c.setAttribute('fill-opacity', hit ? 1 : 0.55); }); });
    $('rainStats').innerHTML = ROWS.map((r) => '<div class="rs"><span style="color:' + r[2].replace('#7FB6FF', '#0072EF') + ';font-weight:700">' + r[1] + '</span><br>Sensitivity <b>' + (100 * st[r[0]].sens).toFixed(0) + '%</b> · False-alarm rate <b>' + (100 * st[r[0]].fpr).toFixed(0) + '%</b></div>').join('');
    clinic();
  }
  function rain() { ROWS.forEach((r) => dots[r[0]].forEach((c, i) => { c.style.transition = 'none'; c.setAttribute('cy', c.dataset.top - 120); })); void svg.getBoundingClientRect(); ROWS.forEach((r) => dots[r[0]].forEach((c, i) => { setTimeout(() => { c.style.transition = ''; c.setAttribute('cy', c.dataset.top); }, 15 * (i % 40)); })); }
  function clinic() {
    const n = +$('nPat').value, pv = +$('prev').value / 100; $('nPatV').textContent = n; $('prevV').textContent = Math.round(pv * 100) + '%';
    const sick = n * pv, well = n - sick, mx = Math.max(1, ...ROWS.map((r) => Math.max(sick * (1 - st[r[0]].sens), well * st[r[0]].fpr)));
    $('clinic').innerHTML = ROWS.map((r) => { const m = sick * (1 - st[r[0]].sens), f = well * st[r[0]].fpr; return '<div class="clrow"><b>' + r[1] + '</b><div class="clbars"><div style="width:' + Math.max(4, 100 * m / mx) + '%;background:#E0407B">' + Math.round(m) + ' missed</div><div style="width:' + Math.max(4, 100 * f / mx) + '%;background:#B45F06">' + Math.round(f) + ' false alarms</div></div></div>'; }).join('');
    $('clinicNote').textContent = 'Out of ' + Math.round(sick) + ' people with disease and ' + Math.round(well) + ' healthy people. Missed cases are the serious error; false alarms cost a follow-up visit. Based on ' + N + ' test images' + (R.meta.synthetic ? ' (synthetic smoke-test data)' : '') + ', so small differences are noise.';
  }
  $('rainLab').textContent = R.meta.synthetic ? 'Synthetic data' : 'Measured';
  $('rainGo').onclick = rain; $('rainReset').onclick = () => { ROWS.forEach((r) => { thr[r[0]] = CS.thresholds[r[0]] != null ? CS.thresholds[r[0]] : 0.5; }); upd(); };
  $('rainMatch').onclick = () => { const target = st['B/glass'].sens; ROWS.forEach((r) => { const k = r[0], s = CS.scores[k], pp = s.filter((_, i) => y[i]).sort((a, b) => a - b); const idx = Math.max(0, Math.floor((1 - target) * pp.length - 1e-9)); thr[k] = pp[idx] != null ? pp[idx] : 0.5; }); upd(); };
  $('nPat').oninput = clinic; $('prev').oninput = clinic;
  upd(); rain();
  new MutationObserver(() => { if (!$('p-rain').hidden) rain(); }).observe($('p-rain'), { attributes: true, attributeFilter: ['hidden'] });
})();
