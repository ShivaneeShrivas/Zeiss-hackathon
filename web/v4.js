(function () {
  const R = window.RESULTS, O = R.optics, X = O.optics_extended, $ = (id) => document.getElementById(id);
  const f1 = (x) => (x == null ? '–' : x.toFixed(1)), f2 = (x) => (x == null ? '–' : x.toFixed(2)), pc = (x) => (x == null ? '–' : (100 * x).toFixed(0) + '%');
  const tick = (b) => '<span class="' + (b ? 'ok">pass' : 'bad">fail') + '</span>';
  const tbl = (head, rows) => '<table class="tbl"><tr>' + head.map((h) => '<th>' + h + '</th>').join('') + '</tr>' + rows.map((r) => '<tr>' + r.map((c) => '<td class="mono">' + c + '</td>').join('') + '</tr>').join('') + '</table>';
  const ci = (o) => pc(o.yield_) + ' <span class="small muted">(' + pc(o.lo) + '–' + pc(o.hi) + ')</span>';

  /* ---- Engineering ---- */
  try {
    const G = X.psf_mtf.metrics.glass.by_field, P = X.psf_mtf.metrics.polymer.by_field, W = 440, H = 280, L = 44, B = 36;
    const x = (v) => L + v * (W - L - 12), y = (v) => 14 + (1 - v) * (H - B - 14);
    let s = '';
    [0, .25, .5, .75, 1].forEach((t) => { s += '<line x1="' + L + '" x2="' + (W - 12) + '" y1="' + y(t) + '" y2="' + y(t) + '" stroke="#E3E7EF"/><text x="' + (L - 6) + '" y="' + (y(t) + 4) + '" font-size="11" fill="#5A6478" text-anchor="end">' + t + '</text><text x="' + x(t) + '" y="' + (H - 18) + '" font-size="11" fill="#5A6478" text-anchor="middle">' + t + '</text>'; });
    const line = (arr, col, dash) => '<polyline fill="none" stroke="' + col + '" stroke-width="3" ' + (dash ? 'stroke-dasharray="6 5"' : '') + ' points="' + arr.map((f) => x(f.field) + ',' + y(f.m)).join(' ') + '"/>';
    const gm = G.map((f) => ({ field: f.field, m: (f.mtf_tan[1] + f.mtf_sag[1]) / 2 })), pm = P.map((f) => ({ field: f.field, m: (f.mtf_tan[1] + f.mtf_sag[1]) / 2 }));
    s += line(gm, '#0072EF') + line(pm, '#E0407B') + line(gm.map((f) => ({ field: f.field, m: f.m * 0.7 })), '#9AA3B5', true);
    s += '<text x="' + (L + 6) + '" y="28" font-size="12" fill="#0072EF" font-weight="700">Glass</text><text x="' + (L + 60) + '" y="28" font-size="12" fill="#E0407B" font-weight="700">Polymer</text><text x="' + (L + 130) + '" y="28" font-size="12" fill="#5A6478">dashed: required (70% of glass)</text>';
    s += '<text x="' + x(.5) + '" y="' + (H - 3) + '" font-size="12" fill="#5A6478" text-anchor="middle">Field position, centre (0) to edge (1)  ·  MTF at 25 lp/mm, y</text>';
    $('mtfChart').innerHTML = s;
    const v = X.psf_mtf.verdict, th = X.psf_mtf.thresholds;
    $('mtfVerdict').innerHTML = '<h3>Verdict against the rules we set first</h3>' + tbl(['Check', 'Rule', 'Result'], [
      ['MTF vs glass', '≥ ' + pc(th.mtf_min_ratio_to_glass) + ' at all fields', tick(v.mtf_ratio_ok) + ' <span class="small muted">worst ' + pc(v.mtf_ratio_min) + '</span>'],
      ['Absolute MTF', '≥ ' + th.mtf_abs_min, tick(v.mtf_abs_ok)],
      ['80% energy radius', '≤ ' + th.ee80_radius_max_um + ' µm', tick(v.ee80_ok)],
      ['Distortion', '≤ ' + th.distortion_max_pct + '%', tick(v.distortion_ok)],
      ['Edge brightness', '≥ ' + pc(th.rel_illum_min), tick(v.rel_illum_ok) + ' <span class="small muted">' + pc(v.rel_illum_edge) + '</span>']]) +
      '<p class="' + (v.overall_pass ? 'ok' : 'bad') + '">Overall: ' + (v.overall_pass ? 'passes' : 'the polymer lens does NOT pass') + '.</p><p class="small muted">This is a real negative result. It is why the restorer, and a wider quality floor, matter.</p>';
    const M = X.manufacturing, rows = [];
    ['glass', 'polymer'].forEach((k) => { const m = M[k]; if (!m) return; m.elements.forEach((e) => rows.push([k + ' ' + e.element, e.material, f2(e.min_thickness_mm), f1(e.max_slope_deg) + '°', f2(e.thickness_ratio), e.violations.length ? '<span class="bad">' + e.violations.join(', ') + '</span>' : '<span class="ok">ok</span>'])); });
    $('mfg').innerHTML = tbl(['Element', 'Material', 'Min thick (mm)', 'Max slope', 'Thick ratio', 'Gates'], rows) + '<p class="small muted">Gates are checked after optimisation, not enforced inside it.</p>';
    const T = X.tolerance, runs = T.runs;
    $('tolNote').textContent = T.n_draws + ' random builds per grade. Ranges are 95% Wilson intervals. Tilt and asphere tolerances are assumed; radius and thickness follow Apollo Optical grades.';
    $('tol').innerHTML = tbl(['Grade', 'Within 125% of own design', 'Meets 4× glass blur'], Object.keys(runs).map((k) => [k.replace(/_/g, ' '), ci(runs[k].robustness_yield_125pct_of_own_nominal), ci(runs[k].quality_floor_yield['4x_glass'])]));
    const mr = X.materials.rows.slice().sort((a, b) => b.composite - a.composite).slice(0, 8);
    $('mats').innerHTML = tbl(['Pair', 'Mean blur µm', 'Mass g', 'Score', 'Screens'], mr.map((r) => [r.pair.replace(/_/g, ' '), f1(r.mean_spot_um), f2(r.mass_g), f2(r.composite), r.hard_fail.length ? '<span class="bad">' + r.hard_fail.join(', ') + '</span>' : '<span class="ok">ok</span>']));
    $('fov').innerHTML = tbl(['Full FOV', 'Glass edge µm', 'Polymer edge µm', 'Converged'], X.fov.rows.map((r) => [r.full_fov_deg + '°', f1(r.glass.spot_after_um[2]), f1(r.polymer.spot_after_um[2]), r.polymer.converged_to_better ? '<span class="ok">yes</span>' : '<span class="bad">no</span>']));
  } catch (e) { console.error('eng', e); }

  /* ---- Restorer study ---- */
  try {
    const S = O.restorer_study, D = { fid_s0: 'Fidelity only (L1 + edges)', lesion_s0: 'Lesion-weighted', lesion_task_s0: 'Lesion-weighted + classifier loss', lesion_glass_s0: 'Lesion-weighted, glass target' };
    $('stDesc').textContent = 'All numbers are from the validation split (the test set is opened once at the end). Selected: ' + (D[S.selected] || S.selected) + '. ' + S.selection_rule + '. Raw polymer: ' + f1(S.raw_polymer.psnr) + ' dB' + (S.raw_polymer.lesion ? ', lesion recall ' + pc(S.raw_polymer.lesion.lesion_recall) : '. This dataset has no lesion masks, so lesion checks are not assessed') + '.';
    const note = { fid_s0: 'Highest image fidelity (PSNR).', lesion_s0: 'Gives lesion-like regions more weight. Needs lesion masks to judge properly.', lesion_task_s0: 'Adds the classifier into its training loss. Scores highest on the validation classifier it trained against, so treat that gain with caution.', lesion_glass_s0: 'Trained to look like the glass image. Smallest fidelity gain.' };
    $('stTable').innerHTML = tbl(['Variant', 'PSNR gain', 'Val AUC', 'Lesion recall', 'Contrast kept', 'Invented / img', 'What we learned'], Object.keys(S.variants).map((k) => { const v = S.variants[k]; return [(D[k] || k) + (k === S.selected ? ' ★' : ''), '+' + f1(v.psnr_gain_db) + ' dB', f2(v.val_auc_frozen_clf), pc((v.lesion || {}).lesion_recall), pc((v.lesion || {}).contrast_retention), f1((v.lesion || {}).hallucinated_blobs), '<span class="small">' + (note[k] || '') + '</span>']; }));
    const t = R.lesion_taxonomy, name = { 'raw/nominal': 'Raw polymer', 'restored/nominal': 'Restored', 'glass/nominal': 'Glass' };
    const taxRows = Object.keys(name).filter((k) => t[k] && t[k].lesion);
    if (!taxRows.length) $('tax').innerHTML = '<p class="mid">Not assessed. This dataset has no lesion masks. Lesion-level checks need a dataset such as IDRiD.</p>'; else $('tax').innerHTML = tbl(['Image', 'Kept', 'Removed', 'Distorted', 'Invented/img', 'Dice'], Object.keys(name).filter((k) => t[k] && t[k].lesion).map((k) => { const l = t[k].lesion; return [name[k], pc(l.kept), pc(l.removed), pc(l.distorted), f1(l.invented_per_image), f2(l.dice)]; }));
    const cl = R.restoration_claims.nominal, lab = { psnr_better: 'Better image fidelity', lesion_recall_not_worse: 'Lesion recall not worse', invented_not_worse: 'No more invented lesions than raw', independent_auc_gain_significant: 'Gain confirmed by independent judge' };
    $('claims').innerHTML = Object.keys(lab).map((k) => '<div class="chk-item">' + (cl[k] == null ? '<span class="mid">n/a</span>' : tick(cl[k])) + '<span>' + lab[k] + (cl[k] == null ? ' <span class="small muted">(needs lesion masks)</span>' : '') + '</span></div>').join('') + '<p class="' + (cl.restoration_claimed ? 'ok' : 'mid') + '">' + (cl.restoration_claimed ? 'Restoration claim supported.' : 'Restoration claim NOT made. ' + (cl.note ? 'Without lesion masks the lesion checks cannot be run, so we never claim it.' : 'At least one check failed.')) + '</p>';
    const ic = R.independent_classifier;
    $('indep').innerHTML = tbl(['Comparison', 'AUC', 'Diff vs glass (95% CI)', 'Non-inferior'], Object.keys(ic).map((k) => { const v = ic[k]; return [k.replace(' (independent classifier)', ''), f2(v.auc), (v.auc_diff >= 0 ? '+' : '') + v.auc_diff.toFixed(3) + ' (' + v.auc_lo.toFixed(3) + ', ' + v.auc_hi.toFixed(3) + ')', tick(v.pass_)]; }));
  } catch (e) { console.error('study', e); }

  /* ---- Capture check ---- */
  try {
    const cv = $('capCanvas'), cx = cv.getContext('2d', { willReadFrequently: true });
    function analyse(img) {
      const N = 256; cx.clearRect(0, 0, N, N); const s = Math.min(N / img.width, N / img.height), w = img.width * s, h = img.height * s;
      cx.fillStyle = '#000'; cx.fillRect(0, 0, N, N); cx.drawImage(img, (N - w) / 2, (N - h) / 2, w, h);
      const d = cx.getImageData(0, 0, N, N).data, g = new Float32Array(N * N); let sum = 0, vis = 0;
      for (let i = 0; i < N * N; i++) { const v = (0.299 * d[4 * i] + 0.587 * d[4 * i + 1] + 0.114 * d[4 * i + 2]) / 255; g[i] = v; if (v > 0.06) { sum += v; vis++; } }
      const mean = vis ? sum / vis : 0; let vr = 0; for (let i = 0; i < N * N; i++) if (g[i] > 0.06) vr += (g[i] - mean) ** 2; const sd = vis ? Math.sqrt(vr / vis) : 0;
      let lap = 0, ln = 0, nz = 0, nn = 0;
      for (let r = 2; r < N - 2; r++) for (let c = 2; c < N - 2; c++) { const k = r * N + c; if (g[k] < 0.06 || g[k - 1] < 0.06 || g[k + 1] < 0.06 || g[k - N] < 0.06 || g[k + N] < 0.06 || g[k - 2] < 0.06 || g[k + 2] < 0.06) continue; const L = 4 * g[k] - g[k - 1] - g[k + 1] - g[k - N] - g[k + N]; lap += L * L; ln++; const dd = g[k] - (g[k - 1] + g[k + 1] + g[k - N] + g[k + N]) / 4; if (Math.abs(dd) < 0.02) { nz += dd * dd; nn++; } }
      const sharp = ln ? Math.sqrt(lap / ln) / Math.max(mean, 0.05) : 0, cover = vis / (N * N * 0.78);
      const items = [
        ['Sharpness', sharp > 0.03, sharp > 0.03 ? 'Detail visible.' : 'Image looks blurred. Refocus, hold steady or clean the lens.', f2(sharp)],
        ['Exposure', mean > 0.18 && mean < 0.62, mean <= 0.18 ? 'Too dark. Raise illumination or reduce eyelid shadow.' : mean >= 0.62 ? 'Too bright. Lower illumination.' : 'Brightness fine.', f2(mean)],
        ['Contrast', sd > 0.07, sd > 0.07 ? 'Enough contrast.' : 'Flat image. Check alignment and focus.', f2(sd)],
        ['Visible area', cover > 0.6, cover > 0.6 ? 'Most of the retina is in view.' : 'Less than ~60% visible. Re-centre on the pupil.', pc(Math.min(cover, 1))]];
      return items;
    }
    function render(items) {
      const ok = items.every((i) => i[1]);
      $('capOut').innerHTML = '<h3 class="' + (ok ? 'ok' : 'bad') + '">' + (ok ? 'Quality check: OK to analyse' : 'Quality check: please recapture') + '</h3>' + items.map((i) => '<div class="chk-item">' + tick(i[1]) + '<span><b>' + i[0] + '</b> ' + i[2] + ' <span class="small muted">(' + i[3] + ')</span></span></div>').join('') +
        '<p class="small muted">Disease-risk result: shown separately, and only after the quality check passes. Not a diagnosis.</p>';
    }
    function load(src) { const im = new Image(); im.onload = () => render(analyse(im)); im.src = src; }
    $('capFile').onchange = (e) => { const f = e.target.files[0]; if (f) load(URL.createObjectURL(f)); };
    R.samples.forEach((s) => { [['C', 'Polymer'], ['D', 'Restored']].forEach(([k, nm]) => { if (!s.files[k]) return; const b = document.createElement('button'); b.className = 'btn ghost'; b.style.padding = '4px 10px'; b.style.width = 'auto'; b.textContent = (s.kind === 'failure' ? 'Fail ' : '') + s.id + ' ' + nm; b.onclick = () => load(s.files[k]); $('capThumbs').appendChild(b); }); });
    const gb = R.gate.by_condition, nm = { nominal: 'Nominal polymer', ood: 'Independent renderer', psf: 'Ray-traced PSF' };
    $('gateTbl').innerHTML = tbl(['Condition', 'Accepted', 'Rejected: quality', 'Rejected: uncertainty'], Object.keys(gb).map((k) => [nm[k] || k, pc(gb[k].accepted), pc(gb[k].quality_rejected), pc(gb[k].uncertainty_rejected)]));
    $('gateNote').textContent = 'Limits were fitted on the nominal images, so the independent and PSF sets are rejected almost entirely. That is a known weakness: the gate is too strict for conditions it has not seen, and needs real-device calibration.';
  } catch (e) { console.error('capture', e); }

  /* ---- Cost & deploy ---- */
  try {
    const cm = O.cost_model, mm = cm.module_mass_g;
    $('modMass').innerHTML = tbl(['Mass (g, median)', 'Glass', 'Polymer'], [['Common parts', f1(mm.common.median), f1(mm.common.median)], ['Whole module', f1(mm.glass_total.median), f1(mm.polymer_total.median)]]) +
      '<p><b>Optics alone: ' + f1(mm.optics_only_saving_pct) + '% lighter.</b> <b>Whole module: ' + f1(mm.module_saving_pct.median) + '% lighter</b> <span class="small muted">(' + f1(mm.module_saving_pct.p5) + '–' + f1(mm.module_saving_pct.p95) + '%)</span>. The big optics number does not carry through to the device.</p>';
    const rows = []; Object.keys(cm.scenarios).forEach((vol) => Object.keys(cm.scenarios[vol]).forEach((g) => { const s = cm.scenarios[vol][g]; rows.push([vol, g.replace('polymer_', ''), '$' + s.glass_camera_module.median.toFixed(0), '$' + s.polymer_camera_module.median.toFixed(0), pc(s.p_polymer_cheaper_camera_module)]); }));
    $('costTbl').innerHTML = tbl(['Units', 'Grade', 'Glass', 'Polymer', 'P(polymer cheaper)'], rows) + '<p class="small muted">' + cm.break_even.note + '</p>';
    const lc = cm.lca; $('lca').innerHTML = tbl(['Units', 'Glass kg CO₂e', 'Polymer kg CO₂e', 'P(polymer lower)'], Object.keys(lc).map((v) => [v, f1(lc[v].glass.median), f1(lc[v].polymer_mixed.median), pc(lc[v].p_polymer_lower)])) + '<p class="small muted">' + cm.lca_boundary + '</p>';
    const D = O.deployment, nmx = { torch_fp32: 'PyTorch fp32', torch_pruned30: 'Pruned 30%', onnx_fp32: 'ONNX fp32', onnx_int8_dynamic: 'ONNX int8' };
    $('deploy').innerHTML = tbl(['Variant', 'ms / image', 'Size MB', 'PSNR dB'], Object.keys(D.rows).map((k) => { const r = D.rows[k]; return [nmx[k] || k, f1(r.median_ms), r.size_mb ? r.size_mb.toFixed(2) : '–', f1(r.psnr_db)]; })) +
      '<p class="small muted">Raspberry Pi 4 estimate ' + f1(D.board_estimate.raspberry_pi4_ms[0]) + '–' + f1(D.board_estimate.raspberry_pi4_ms[1]) + ' ms (' + D.board_estimate.label + '). Power: ' + D.power_w + '.</p>';
  } catch (e) { console.error('cost', e); }

})();
