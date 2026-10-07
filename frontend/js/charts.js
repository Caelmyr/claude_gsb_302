/* Lightweight SVG charts (bar, horizontal bar / tornado, line). No external deps. */

function _svg(container, width, height) {
  container.innerHTML = '';
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('width', width);
  svg.setAttribute('height', height);
  svg.style.fontFamily = 'system-ui, sans-serif';
  container.appendChild(svg);
  return svg;
}

function _axis(svg, x, y, label, opts = {}) {
  const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
  t.setAttribute('x', x); t.setAttribute('y', y);
  t.setAttribute('font-size', opts.size || 10);
  t.setAttribute('fill', '#6b7280');
  t.setAttribute(opts.anchor || 'text-anchor', opts.anchor || 'middle');
  t.textContent = label;
  svg.appendChild(t);
}

/* Vertical bar chart: labels along x, numeric values along y. */
function barChart(container, { labels, values, height = 240, color = '#4e79a7', valueFmt = v => fmt(v) }) {
  const W = 560, H = height, padL = 46, padB = 44, padT = 16;
  const max = Math.max(1, ...values.map(v => Math.abs(v)));
  const innerW = W - padL - 12, innerH = H - padT - padB;
  const svg = _svg(container, W, H);

  const bw = Math.min(52, innerW / labels.length * 0.6);
  const stepX = innerW / labels.length;

  // y axis
  for (let i = 0; i <= 4; i++) {
    const val = max * i / 4;
    const y = padT + innerH - innerH * i / 4;
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', padL); line.setAttribute('y1', y);
    line.setAttribute('x2', W - 12); line.setAttribute('y2', y);
    line.setAttribute('stroke', '#eef0f3');
    svg.appendChild(line);
    _axis(svg, padL - 6, y + 3, valueFmt(val), { anchor: 'end' });
  }

  labels.forEach((lb, i) => {
    const v = values[i];
    const x = padL + i * stepX + (stepX - bw) / 2;
    const barH = innerH * Math.abs(v) / max;
    const y = padT + innerH - barH;
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('x', x); rect.setAttribute('y', y);
    rect.setAttribute('width', bw); rect.setAttribute('height', Math.max(0, barH));
    rect.setAttribute('rx', 3); rect.setAttribute('fill', color);
    svg.appendChild(rect);
    _axis(svg, x + bw / 2, y - 5, valueFmt(v));
    _axis(svg, x + bw / 2, padT + innerH + 16, lb, { size: 10 });
  });
  return svg;
}

/* Horizontal bar chart (tornado for +/- deltas): zero line in the middle. */
function hBarChart(container, { labels, values, height = 260, valueFmt = v => fmt(v) }) {
  const W = 640, H = height, padL = 130, padR = 60, padT = 16, padB = 12;
  const max = Math.max(1, ...values.map(v => Math.abs(v)));
  const innerW = W - padL - padR, innerH = H - padT - padB;
  const svg = _svg(container, W, H);
  const centerX = padL + innerW / 2;

  // zero line
  const zl = document.createElementNS('http://www.w3.org/2000/svg', 'line');
  zl.setAttribute('x1', centerX); zl.setAttribute('y1', padT);
  zl.setAttribute('x2', centerX); zl.setAttribute('y2', padT + innerH);
  zl.setAttribute('stroke', '#9ca3af'); zl.setAttribute('stroke-width', 1);
  svg.appendChild(zl);

  const rowH = Math.min(46, innerH / labels.length);
  labels.forEach((lb, i) => {
    const v = values[i];
    const y = padT + i * rowH;
    const barW = innerW / 2 * Math.abs(v) / max;
    const x = v >= 0 ? centerX : centerX - barW;
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('x', x); rect.setAttribute('y', y + 4);
    rect.setAttribute('width', Math.max(1, barW)); rect.setAttribute('height', Math.max(8, rowH - 10));
    rect.setAttribute('rx', 3);
    rect.setAttribute('fill', v >= 0 ? '#e15759' : '#4e79a7');
    svg.appendChild(rect);
    _axis(svg, padL - 8, y + rowH / 2 + 4, lb, { anchor: 'end', size: 11 });
    const valTxt = valueFmt(v);
    _axis(svg, v >= 0 ? x + barW + 5 : x - 5, y + rowH / 2 + 4, valTxt, { anchor: v >= 0 ? 'start' : 'end', size: 11 });
  });
  return svg;
}

/* Simple line chart with dots. */
function lineChart(container, { labels, values, height = 240, color = '#2563eb', valueFmt = v => fmt(v) }) {
  const W = 560, H = height, padL = 46, padB = 40, padT = 16;
  const min = Math.min(...values), max = Math.max(...values);
  const span = (max - min) || 1;
  const innerW = W - padL - 12, innerH = H - padT - padB;
  const svg = _svg(container, W, H);
  const x = i => padL + (labels.length === 1 ? innerW / 2 : innerW * i / (labels.length - 1));
  const y = v => padT + innerH - innerH * (v - min) / span;

  for (let i = 0; i <= 4; i++) {
    const val = min + span * i / 4;
    const yy = y(val);
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', padL); line.setAttribute('y1', yy);
    line.setAttribute('x2', W - 12); line.setAttribute('y2', yy);
    line.setAttribute('stroke', '#eef0f3');
    svg.appendChild(line);
    _axis(svg, padL - 6, yy + 3, valueFmt(val), { anchor: 'end' });
  }

  let d = '';
  values.forEach((v, i) => { d += (i ? 'L' : 'M') + x(i) + ' ' + y(v) + ' '; });
  const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  path.setAttribute('d', d);
  path.setAttribute('fill', 'none'); path.setAttribute('stroke', color);
  path.setAttribute('stroke-width', 2);
  svg.appendChild(path);

  values.forEach((v, i) => {
    const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    c.setAttribute('cx', x(i)); c.setAttribute('cy', y(v)); c.setAttribute('r', 3.5);
    c.setAttribute('fill', color);
    svg.appendChild(c);
    _axis(svg, x(i), y(v) - 7, valueFmt(v));
    _axis(svg, x(i), padT + innerH + 16, labels[i], { size: 10 });
  });
  return svg;
}

/* Best-incumbent convergence chart: points are (time seconds, best objective)
   improvement events; drawn as a monotone non-increasing step curve (classic
   solver log look).  An optional lower-bound series is overlaid for exact
   solvers.  Value labels are only shown at first/last to avoid clutter when
   there are hundreds of improvements. */
function convergenceChart(container, { history, lowerBounds = [], height = 220,
  valueFmt = v => fmt(v) }) {
  const W = 560, H = height, padL = 52, padR = 14, padB = 34, padT = 18;
  container.innerHTML = '';
  if (!history || history.length === 0) {
    container.innerHTML = '<p class="small muted" style="padding:24px;text-align:center">等待第一个可行解…</p>';
    return null;
  }
  // Extend the curve to "now" so the final incumbent holds to the right edge.
  const now = Math.max(history[history.length - 1].t,
    lowerBounds.length ? lowerBounds[lowerBounds.length - 1].t : 0);
  const pts = history.map(p => [p.t, p.best]);
  pts.push([now, history[history.length - 1].best]);
  const lbPts = lowerBounds.map(p => [p.t, p.lb]);

  const maxT = Math.max(1, now);
  let lo = Math.min(...pts.map(p => p[1]));
  let hi = Math.max(...pts.map(p => p[1]));
  if (lbPts.length) {
    lo = Math.min(lo, ...lbPts.map(p => p[1]));
    hi = Math.max(hi, ...lbPts.map(p => p[1]));
  }
  if (!isFinite(lo) || !isFinite(hi) || hi - lo < 1e-9) { lo = (lo || 0) - 1; hi = (hi || 0) + 1; }
  const span = hi - lo;
  lo -= span * 0.05; hi += span * 0.05;

  const svg = _svg(container, W, H);
  const innerW = W - padL - padR, innerH = H - padT - padB;
  const x = t => padL + innerW * Math.min(1, t / maxT);
  const y = v => padT + innerH - innerH * (v - lo) / (hi - lo);

  for (let i = 0; i <= 4; i++) {
    const val = lo + (hi - lo) * i / 4;
    const yy = y(val);
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', padL); line.setAttribute('y1', yy);
    line.setAttribute('x2', W - padR); line.setAttribute('y2', yy);
    line.setAttribute('stroke', '#eef0f3');
    svg.appendChild(line);
    _axis(svg, padL - 6, yy + 3, valueFmt(val), { anchor: 'end' });
  }
  for (let i = 0; i <= 4; i++) {
    _axis(svg, padL + innerW * i / 4, padT + innerH + 16,
      (maxT * i / 4).toFixed(1) + 's', { size: 10 });
  }

  // lower bound (dashed)
  if (lbPts.length > 1) {
    let dlb = '';
    lbPts.forEach((p, i) => { dlb += (i ? 'L' : 'M') + x(p[0]) + ' ' + y(p[1]) + ' '; });
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', dlb); path.setAttribute('fill', 'none');
    path.setAttribute('stroke', '#f28e2b'); path.setAttribute('stroke-width', 1.5);
    path.setAttribute('stroke-dasharray', '5 4');
    svg.appendChild(path);
  }

  // incumbent step curve
  let d = '';
  pts.forEach((p, i) => {
    if (i === 0) { d += `M${x(p[0])} ${y(p[1])} `; }
    else {
      const prev = pts[i - 1];
      d += `L${x(p[0])} ${y(prev[1])} L${x(p[0])} ${y(p[1])} `;
    }
  });
  const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  path.setAttribute('d', d); path.setAttribute('fill', 'none');
  path.setAttribute('stroke', '#2563eb'); path.setAttribute('stroke-width', 2);
  svg.appendChild(path);

  // markers + labels at first improvement and current best
  const first = history[0], last = history[history.length - 1];
  [[first.t, first.best, 'start'], [last.t, last.best, 'end']].forEach(([t, v, k]) => {
    const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    c.setAttribute('cx', x(t)); c.setAttribute('cy', y(v)); c.setAttribute('r', 3.5);
    c.setAttribute('fill', '#2563eb');
    svg.appendChild(c);
    _axis(svg, x(t) + (k === 'end' ? -6 : 6), y(v) - 7, valueFmt(v),
      { anchor: k === 'end' ? 'end' : 'start', size: 10 });
  });

  // legend
  _axis(svg, padL + 8, padT + 4, '— 目前最好目标值', { anchor: 'start', size: 10 });
  const leg = svg.lastChild; leg.setAttribute('fill', '#2563eb');
  if (lbPts.length) {
    _axis(svg, padL + 150, padT + 4, '- - 下界', { anchor: 'start', size: 10 });
    svg.lastChild.setAttribute('fill', '#f28e2b');
  }
  return svg;
}
