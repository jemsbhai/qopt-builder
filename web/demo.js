/* Guided demo: plays the whole pipeline as one animation on a fixed
 * stage, for watching or for recording.
 *
 * Nothing here computes a result of its own. The recommendation, the
 * exact solve, the Ising form, the circuit, the scan, the refinement and
 * the success probability all come from qubo.js, qaoa.js and circuit.js,
 * the same functions the parity test checks against the Python package.
 * This file only decides when and how to draw them.
 *
 * Address options: ?preset=NAME  ?autoplay=1  ?speed=0.5|1|2
 *                  ?budget=N  ?controls=0  ?captions=0
 * Keys: space play or pause, right arrow next stage, R restart,
 *       H hide the controls, C hide the captions. */
(function () {
  "use strict";
  const Q = window.QOPT;
  const SVGNS = "http://www.w3.org/2000/svg";
  const LAYER_COLORS = ["#22d3ee", "#f59e0b", "#a78bfa", "#34d399", "#f472b6", "#facc15"];
  const NEUTRAL = "#e2e8f0";
  const SCAN_SAMPLES = 200, SCAN_SEED = 20261110;
  const STAGES = ["problem", "recommend", "certify", "build", "run"];
  const $ = (id) => document.getElementById(id);

  const stage = $("stage"), scaled = $("scaled"), caption = $("caption"), card = $("card"), overlay = $("overlay");
  const query = new URLSearchParams(window.location.search);
  const reducedMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ------------------------------------------------------------ helpers

  function svg(tag, attrs, parent) {
    const node = document.createElementNS(SVGNS, tag);
    if (attrs) for (const key of Object.keys(attrs)) node.setAttribute(key, attrs[key]);
    if (parent) parent.appendChild(node);
    return node;
  }
  function html(tag, className, text, parent) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    if (parent) parent.appendChild(node);
    return node;
  }
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const lerp = (a, b, t) => a + (b - a) * t;
  const easeInOut = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
  const easeOut = (t) => 1 - Math.pow(1 - t, 3);
  const linear = (t) => t;
  const num = (v, digits) => Q.fixed(v, digits === undefined ? 3 : digits); // same text as the Python output
  const count = (v) => Math.round(v).toLocaleString("en-US");

  // -------------------------------------------- the stage scales to fit

  let scale = 1;
  function fit() {
    scale = Math.min(window.innerWidth / 1600, window.innerHeight / 900);
    stage.style.transform = "scale(" + scale + ")";
    scaled.style.width = 1600 * scale + "px";
    scaled.style.height = 900 * scale + "px";
  }
  window.addEventListener("resize", fit);
  fit();

  /* A point inside one of the stage's SVGs, in stage coordinates. */
  function toStage(svgEl, x, y) {
    const box = svgEl.getBoundingClientRect(), origin = stage.getBoundingClientRect();
    const view = svgEl.viewBox.baseVal;
    return [
      (box.left + (x / view.width) * box.width - origin.left) / scale,
      (box.top + (y / view.height) * box.height - origin.top) / scale,
    ];
  }

  // ------------------------------------- a clock that can pause and skip

  const CANCEL = { cancelled: true };
  const clock = { speed: 1, paused: true, skip: false, run: 0 };

  /* Call onFrame(eased, raw) every frame until `duration` ms of playing
   * time have passed. Pausing stops the time; skipping jumps to the end;
   * a restart cancels it. */
  function tween(duration, onFrame, ease) {
    const run = clock.run, easing = ease || easeInOut;
    return new Promise((resolve, reject) => {
      let elapsed = 0, last = null;
      function frame(now) {
        if (run !== clock.run) { reject(CANCEL); return; }
        if (last === null) last = now;
        if (!clock.paused) elapsed += (now - last) * clock.speed;
        last = now;
        const t = clock.skip || duration <= 0 ? 1 : Math.min(1, elapsed / duration);
        if (onFrame) onFrame(easing(t), t);
        if (t >= 1) resolve(); else window.requestAnimationFrame(frame);
      }
      window.requestAnimationFrame(frame);
    });
  }
  const sleep = (ms) => tween(ms, null);

  /* Run a computation a slice per frame: step() returns true when it is
   * finished. Keeps the page responsive during the larger solves. */
  function slices(step, budgetMs) {
    const run = clock.run;
    return new Promise((resolve, reject) => {
      function frame() {
        if (run !== clock.run) { reject(CANCEL); return; }
        const started = window.performance.now();
        let finished = false;
        do { finished = step(); } while (!finished && window.performance.now() - started < (budgetMs || 10));
        if (finished) resolve(); else window.requestAnimationFrame(frame);
      }
      window.requestAnimationFrame(frame);
    });
  }

  function say(text) {
    const run = clock.run;
    caption.classList.add("fade");
    window.setTimeout(() => {
      if (run !== clock.run) return;
      caption.textContent = text;
      // a caption too long for its two lines is set smaller, never cut off
      caption.style.fontSize = "";
      for (let size = 16; size >= 12 && caption.scrollHeight > caption.clientHeight + 1; size--) {
        caption.style.fontSize = size + "px";
      }
      caption.classList.remove("fade");
    }, 170);
  }

  function setStage(name) {
    const at = STAGES.indexOf(name);
    document.querySelectorAll("#stages li").forEach((li, k) => {
      li.classList.toggle("done", at < 0 ? name === "end" : k < at);
      li.classList.toggle("now", k === at);
    });
  }

  function openPanel(id) {
    document.querySelectorAll(".panel.now").forEach((p) => p.classList.remove("now"));
    const panel = $(id);
    panel.classList.remove("idle");
    panel.classList.add("now");
  }

  /* The way back to the main page: the brand in the header and a link in
   * the controls (demo.html), and a link on the title and end cards,
   * which cover both while they are up. */
  const HOME = "index.html";

  function anchor(parent, label, href) {
    const link = html("a", null, label, parent);
    link.href = href;
    return link;
  }

  function showCard(build) {
    card.textContent = "";
    build(card);
    card.classList.add("show");
  }
  function hideCard() { card.classList.remove("show"); }

  // ------------------------------------------------- the problem as a graph

  /* The couplings of a problem as edges. A MaxCut preset keeps its own
   * edge weights; a QUBO shows Q[i][j] + Q[j][i] for every coupled pair. */
  function couplingEdges(preset, problem) {
    if (preset.kind === "maxcut") return preset.edges.map(([i, j, w]) => ({ i, j, w }));
    const edges = [], M = problem.matrix;
    for (let i = 0; i < problem.n; i++) {
      for (let j = i + 1; j < problem.n; j++) {
        const w = M[i][j] + M[j][i];
        if (Math.abs(w) > 1e-12) edges.push({ i, j, w });
      }
    }
    return edges;
  }
  const edgeKey = (i, j) => Math.min(i, j) + "-" + Math.max(i, j);

  /* Level-by-level positions when the graph is a tree, otherwise null. */
  function treeLayout(n, edges, width, height, pad) {
    if (edges.length !== n - 1 || n < 3) return null;
    const next = Array.from({ length: n }, () => []);
    for (const e of edges) { next[e.i].push(e.j); next[e.j].push(e.i); }
    const depth = new Array(n).fill(-1), children = Array.from({ length: n }, () => []);
    const queue = [0];
    depth[0] = 0;
    let reached = 0;
    while (queue.length) {
      const u = queue.shift();
      reached++;
      for (const v of next[u]) {
        if (depth[v] < 0) { depth[v] = depth[u] + 1; children[u].push(v); queue.push(v); }
      }
    }
    if (reached !== n) return null;
    const slot = new Array(n).fill(0);
    let leaves = 0;
    (function place(u) {
      if (!children[u].length) { slot[u] = leaves++; return; }
      children[u].forEach(place);
      slot[u] = (slot[children[u][0]] + slot[children[u][children[u].length - 1]]) / 2;
    })(0);
    const deepest = Math.max.apply(null, depth);
    return slot.map((s, i) => [
      pad + (leaves > 1 ? s / (leaves - 1) : 0.5) * (width - 2 * pad),
      pad + (deepest ? depth[i] / deepest : 0.5) * (height - 2 * pad),
    ]);
  }

  function ringLayout(n, width, height, pad) {
    return Array.from({ length: n }, (_, i) => {
      const a = (2 * Math.PI * i) / n - Math.PI / 2;
      return [width / 2 + (width / 2 - pad) * Math.cos(a), height / 2 + (height / 2 - pad) * Math.sin(a)];
    });
  }

  /* Where each edge's weight is written: beside its own edge, 8 units off
   * the line, at whichever of a few points along the edge leaves the
   * label most clear of the other edges, the nodes, the labels placed
   * before it and the rim of the drawing. With room to spare, the
   * midpoint and the upper side win. */
  function weightSpots(edges, edgeEls, lengths, pos, r, W, H) {
    const OFF = 8, ROOM = 10;
    const outlines = edges.map((e) => {
      const key = edgeKey(e.i, e.j), points = [];
      for (let s = 0; s <= 16; s++) {
        const point = edgeEls[key].getPointAtLength((lengths[key] * s) / 16);
        points.push([point.x, point.y]);
      }
      return points;
    });
    const toSegment = (x, y, a, b) => {
      const dx = b[0] - a[0], dy = b[1] - a[1];
      const t = clamp(((x - a[0]) * dx + (y - a[1]) * dy) / (dx * dx + dy * dy || 1), 0, 1);
      return Math.hypot(x - a[0] - t * dx, y - a[1] - t * dy);
    };
    const spots = [];
    edges.forEach((e, k) => {
      const key = edgeKey(e.i, e.j), line = edgeEls[key], length = lengths[key];
      let best = null;
      for (const t of [0.5, 0.38, 0.62, 0.26, 0.74]) {
        const at = line.getPointAtLength(length * t);
        const a = line.getPointAtLength(length * t - 1), b = line.getPointAtLength(length * t + 1);
        const size = Math.hypot(b.x - a.x, b.y - a.y) || 1;
        const nx = -(b.y - a.y) / size, ny = (b.x - a.x) / size;
        for (const side of [1, -1]) {
          const x = at.x + side * OFF * nx, y = at.y + side * OFF * ny;
          let room = Math.min(x - 5, W - 5 - x, y - 6, H - 6 - y);
          outlines.forEach((points, other) => {
            if (other === k) return;
            for (let s = 0; s + 1 < points.length; s++) room = Math.min(room, toSegment(x, y, points[s], points[s + 1]));
          });
          pos.forEach(([cx, cy]) => { room = Math.min(room, Math.hypot(x - cx, y - cy) - r - 3); });
          spots.forEach(([sx, sy]) => { room = Math.min(room, Math.hypot(x - sx, y - sy) - 5); });
          const score = Math.min(room, ROOM) - Math.abs(t - 0.5) * 16 - (side * ny > 0.3 ? 0.5 : 0);
          if (best === null || score > best.score) best = { score, x, y };
        }
      }
      spots.push([best.x, best.y]);
    });
    return spots;
  }

  function drawGraph(S) {
    const g = $("graph"), W = 392, H = 200, n = S.n;
    g.textContent = "";
    const r = n <= 10 ? 13 : n <= 16 ? 10 : 7;
    const pos = treeLayout(n, S.edges, W, H, r + 5) || ringLayout(n, W, H, r + 5);
    const showWeights = n <= 10 && S.edges.some((e) => e.w !== S.edges[0].w);
    const edgeEls = {}, lengths = {}, mids = {};
    for (const e of S.edges) {
      const [x1, y1] = pos[e.i], [x2, y2] = pos[e.j];
      // A straight edge that would run through another node is bowed
      // towards the middle of the drawing instead, so it reads as its own edge.
      const dx = x2 - x1, dy = y2 - y1, span = Math.hypot(dx, dy) || 1;
      let blocked = false;
      for (let k = 0; k < n && !blocked; k++) {
        if (k === e.i || k === e.j) continue;
        const along = ((pos[k][0] - x1) * dx + (pos[k][1] - y1) * dy) / (span * span);
        if (along <= 0 || along >= 1) continue;
        blocked = Math.hypot(pos[k][0] - (x1 + along * dx), pos[k][1] - (y1 + along * dy)) < r + 3;
      }
      let d = "M" + x1 + " " + y1 + "L" + x2 + " " + y2;
      if (blocked) {
        const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
        const tx = W / 2 - mx, ty = H / 2 - my, pull = Math.hypot(tx, ty) || 1, bow = 2 * (r + 9);
        d = "M" + x1 + " " + y1 + "Q" + (mx + (tx / pull) * bow) + " " + (my + (ty / pull) * bow) + " " + x2 + " " + y2;
      }
      const line = svg("path", { d, class: "edge", fill: "none" }, g);
      const key = edgeKey(e.i, e.j), length = line.getTotalLength(), mid = line.getPointAtLength(length / 2);
      line.style.strokeDasharray = length;
      line.style.strokeDashoffset = length;
      edgeEls[key] = line;
      lengths[key] = length;
      mids[key] = [mid.x, mid.y];
    }
    const weightEls = [];
    if (showWeights) {
      const spots = weightSpots(S.edges, edgeEls, lengths, pos, r, W, H);
      S.edges.forEach((e, k) => {
        const label = svg("text", { x: spots[k][0], y: spots[k][1], class: "weight", opacity: 0 }, g);
        label.textContent = String(e.w);
        weightEls.push(label);
      });
    }
    const nodeEls = pos.map(([x, y], i) => {
      const node = svg("g", { class: "node", opacity: 0 }, g);
      svg("circle", { cx: x, cy: y, r }, node);
      const label = svg("text", { x, y: y + 0.5 }, node);
      label.textContent = String(i);
      if (r < 13) label.style.fontSize = r < 10 ? "8px" : "10px";
      return node;
    });
    S.graph = { pos, edgeEls, lengths, mids, weightEls, nodeEls, r };
  }

  /* Colour the graph by an assignment: the two sides, and for MaxCut the
   * edges the assignment cuts. */
  function paintAssignment(S, x) {
    S.graph.nodeEls.forEach((node, i) => {
      node.classList.toggle("side-0", !x[i]);
      node.classList.toggle("side-1", !!x[i]);
    });
    if (S.preset.kind !== "maxcut") return;
    for (const e of S.edges) {
      const cut = x[e.i] !== x[e.j];
      const line = S.graph.edgeEls[edgeKey(e.i, e.j)];
      line.classList.toggle("cut", cut);
      line.classList.toggle("uncut", !cut);
    }
  }

  // ------------------------------------------------------------ scenes

  async function sceneTitle(S, auto) {
    setStage("");
    showCard((box) => {
      const headline = html("h1", null, null, box);
      ["Quantum Optimisation", "Circuit Builder"].forEach((line) => html("div", null, line, headline));
      html("p", null, "Describe an optimisation problem. Get the circuit, the reasoning behind it, " +
        "and a certified answer to check the quantum result against.", box);
      if (!auto) {
        const actions = html("div", "actions", null, box);
        const button = html("button", null, "Play the guided demo", actions);
        button.type = "button";
        button.addEventListener("click", () => setPaused(false));
        anchor(actions, "Back to the main page", HOME);
      }
    });
    say("");
    await sleep(2600);
    hideCard();
    await sleep(500);
  }

  async function sceneProblem(S) {
    setStage("problem");
    openPanel("p-problem");
    $("problem-title").textContent = S.preset.title;
    drawGraph(S);
    const linearTerms = S.problem.matrix.filter((row, i) => Math.abs(row[i]) > 1e-12).length;
    const facts = $("problem-facts");
    facts.textContent = "";
    facts.append(html("b", null, String(S.n)), " variables, ", html("b", null, String(S.edges.length)), " couplings");
    if (S.preset.kind === "qubo") {
      facts.append(", linear terms on ", html("b", null, String(linearTerms)));
      if (S.problem.offset) facts.append(", offset ", html("b", null, String(S.problem.offset)));
    }

    const sentences = S.preset.about.match(/[^.]+\.(\s|$)/g) || [S.preset.about];
    say(sentences[0].trim());
    const G = S.graph, n = S.n, m = S.edges.length;
    await tween(900, (e, t) => {
      G.nodeEls.forEach((node, i) => { node.setAttribute("opacity", clamp(t * (n + 4) / 4 - i / 4, 0, 1)); });
    }, linear);
    await tween(clamp(260 * m, 900, 2200), (e, t) => {
      S.edges.forEach((edge, k) => {
        const key = edgeKey(edge.i, edge.j);
        const local = clamp(t * (m + 3) / 3 - k / 3, 0, 1);
        G.edgeEls[key].style.strokeDashoffset = G.lengths[key] * (1 - easeOut(local));
      });
      G.weightEls.forEach((label) => label.setAttribute("opacity", clamp(t * 2 - 1, 0, 1)));
    }, linear);
    for (const line of S.edges) G.edgeEls[edgeKey(line.i, line.j)].style.strokeDasharray = "none";
    await sleep(1200);
    // the rest of the description, one caption at a time: sentences share
    // a caption only while it stays short enough to take in at once
    const captions = [];
    for (const sentence of sentences.slice(1)) {
      const last = captions.length - 1;
      if (last >= 0 && (captions[last] + sentence).trim().length <= 120) captions[last] += sentence;
      else captions.push(sentence);
    }
    for (const line of captions) {
      say(line.trim());
      await sleep(clamp(1400 + 22 * line.trim().length, 2400, 4600)); // longer text stays up longer
    }
  }

  function setMarker(gaugeId, fraction) {
    const marker = $(gaugeId).querySelector(".marker");
    marker.style.opacity = 1;
    marker.style.left = clamp(fraction, 0, 1) * 100 + "%";
  }

  async function sceneRecommend(S) {
    setStage("recommend");
    openPanel("p-rec");
    const rec = S.rec, n = S.n, limit = Q.EXACT_CLASSICAL_LIMIT;

    say("The recommender reads the problem's features: its size, how densely its variables are coupled, " +
      "and a hardware qubit budget if one is given.");
    await sleep(2600);

    // size against the exact-solve limit
    const sizeMax = Math.max(30, n + 4);
    const sizeGauge = $("g-size");
    sizeGauge.querySelector('[data-tick="limit"]').style.left = (limit / sizeMax) * 100 + "%";
    const sizeValue = sizeGauge.querySelector(".value");
    await tween(900, (e) => {
      setMarker("g-size", (n * e) / sizeMax);
      sizeValue.textContent = "n = " + Math.round(n * e);
    });
    say(n <= limit
      ? "With n = " + n + " every assignment can be checked, so the classical optimum can be certified exactly."
      : "With n = " + n + " there are too many assignments to check them all, so no exact answer is promised.");
    await sleep(2400);

    // coupling density against the two thresholds
    const density = S.density, pairs = (n * (n - 1)) / 2;
    const densityValue = $("g-density").querySelector(".value");
    await tween(1300, (e) => {
      setMarker("g-density", density * e);
      densityValue.textContent = Q.fixed(density * e, 2);
    });
    if (rec.algorithm !== "none") {
      const p = rec.params.p; // the engine's choice; the band it falls in is lit, not recomputed here
      $("g-density").querySelector('[data-band="' + p + '"]').classList.add("lit");
      say(S.edges.length + " of " + pairs + " variable pairs are coupled, a density of " + Q.fixed(density, 2) +
        ". That is the " + ["sparse", "moderate", "high"][p - 1] + " band: QAOA depth p = " + p + ".");
      await sleep(2800);
    }

    // the qubit budget, when one is set
    if (S.budget !== null) {
      const gauge = $("g-budget");
      gauge.classList.remove("off");
      const top = Math.max(n, S.budget, 1) * 1.25;
      const tick = gauge.querySelector('[data-tick="budget"]');
      tick.style.left = (S.budget / top) * 100 + "%";
      tick.querySelector("span").textContent = "budget " + S.budget;
      const value = gauge.querySelector(".value");
      value.textContent = S.budget + " qubits";
      await tween(900, (e) => setMarker("g-budget", (n * e) / top));
      if (n > S.budget) value.classList.add("bad");
      say(n > S.budget
        ? "The problem needs " + n + " qubits and the budget is " + S.budget + ". No full-width circuit fits."
        : "The problem needs " + n + " qubits, which fits the budget of " + S.budget + ".");
      await sleep(2400);
    }

    // the decision, with the engine's own reasons
    const decision = $("decision"), algo = decision.querySelector(".algo"), reasons = decision.querySelector("ul");
    reasons.textContent = "";
    for (const line of rec.rationale) html("li", null, line, reasons);
    if (rec.algorithm === "none") {
      algo.textContent = "Refused: no circuit built";
      algo.classList.add("refused");
    } else {
      algo.textContent = (rec.algorithm === "exact+qaoa" ? "Exact solve + QAOA" : "QAOA") + ", depth p = " + rec.params.p;
    }
    decision.classList.add("show");
    if (rec.algorithm === "none") {
      say("The builder refuses and says why. It does not hand back a truncated circuit.");
      await sleep(4200);
    } else {
      say("Each recommendation carries its reasons, so any choice can be overridden with the reason in view.");
      await sleep(3000);
    }
  }

  /* y scale and static parts shared by both forms of the cost chart. */
  function landscapeFrame(lo, hi) {
    const chart = $("landscape");
    chart.textContent = "";
    const box = { left: 46, right: 536, top: 8, bottom: 126 };
    const span = hi - lo || 1;
    const y = (v) => box.bottom - ((v - lo) / span) * (box.bottom - box.top);
    svg("line", { x1: box.left, y1: box.bottom + 0.5, x2: box.right, y2: box.bottom + 0.5, class: "wire" }, chart);
    const top = svg("text", { x: box.left - 6, y: box.top + 8, class: "axis", "text-anchor": "end" }, chart);
    top.textContent = num(hi, 1);
    const bottom = svg("text", { x: box.left - 6, y: box.bottom, class: "axis", "text-anchor": "end" }, chart);
    bottom.textContent = num(lo, 1);
    const name = svg("text", { x: 12, y: (box.top + box.bottom) / 2, class: "axis", "text-anchor": "middle",
      transform: "rotate(-90 12 " + (box.top + box.bottom) / 2 + ")" }, chart);
    name.textContent = "cost";
    return { chart, box, y };
  }

  async function sceneCertify(S) {
    setStage("certify");
    openPanel("p-exact");
    const n = S.n, total = Math.pow(2, n);
    const lead = $("exact-lead"), note = $("exact-note");

    if (S.rec.algorithm !== "exact+qaoa") {
      lead.textContent = "Not run: n = " + n + " is beyond the exact-solve limit of " + Q.EXACT_CLASSICAL_LIMIT + ".";
      say("There is no certified optimum at this size, and the result below says so.");
      await sleep(3000);
      return;
    }

    say("Now the exact classical solve. Every one of the " + count(total) + " assignments is scored, " +
      "and the lowest cost is the certified optimum.");
    lead.textContent = "scored 0 of " + count(total) + " assignments";

    if (n <= Q.SIMULATION_LIMIT) {
      // small enough to hold every cost: draw the whole landscape by assignment number
      const costs = Q.basisCosts(S.ising, n);
      let lo = Infinity, hi = -Infinity;
      for (let k = 0; k < total; k++) { if (costs[k] < lo) lo = costs[k]; if (costs[k] > hi) hi = costs[k]; }
      const { chart, box, y } = landscapeFrame(lo, hi);
      const width = box.right - box.left;
      const clip = svg("clipPath", { id: "sweep" }, chart);
      const clipRect = svg("rect", { x: box.left, y: 0, width: 0, height: 150 }, clip);
      const shown = svg("g", { "clip-path": "url(#sweep)" }, chart);
      let barEls = null;
      if (total <= 128) {
        // one stem per assignment, hanging from the highest cost down to its own
        const step = width / total, r = clamp(step * 0.32, 1.6, 3.4);
        barEls = [];
        for (let k = 0; k < total; k++) {
          const cx = box.left + (k + 0.5) * step;
          const stem = svg("g", { class: "bar" }, shown);
          svg("line", { x1: cx, x2: cx, y1: box.top, y2: y(costs[k]) }, stem);
          svg("circle", { cx, cy: y(costs[k]), r }, stem);
          barEls.push(stem);
        }
      } else {
        const buckets = Math.floor(width / 2), per = total / buckets;
        let d = "";
        for (let b = 0; b < buckets; b++) {
          let bl = Infinity, bh = -Infinity;
          for (let k = Math.floor(b * per); k < Math.floor((b + 1) * per); k++) {
            if (costs[k] < bl) bl = costs[k];
            if (costs[k] > bh) bh = costs[k];
          }
          d += "M" + (box.left + b * 2 + 1) + " " + y(bh).toFixed(1) + "V" + (y(bl) + 1).toFixed(1);
        }
        svg("path", { d, class: "envelope", stroke: "#46526a", "stroke-width": 1.4, fill: "none" }, shown);
      }
      // the running minimum, as a step line
      let running = Infinity, path = "";
      for (let k = 0; k < total; k++) {
        if (costs[k] < running - Q.TIE_TOLERANCE) {
          const px = box.left + (k / total) * width;
          path += (path ? "H" + px.toFixed(1) : "M" + px.toFixed(1) + " " + y(costs[k]).toFixed(1)) + "V" + y(costs[k]).toFixed(1);
          running = costs[k];
        }
      }
      path += "H" + box.right;
      svg("path", { d: path, class: "runmin" }, shown);

      await tween(clamp(700 + 40 * Math.sqrt(total), 2600, 4200), (e, t) => {
        clipRect.setAttribute("width", width * t);
        lead.textContent = "scored " + count(total * t) + " of " + count(total) + " assignments";
      }, linear);

      S.exact = Q.bruteForce(S.problem);
      const optimal = [];
      for (let k = 0; k < total; k++) if (costs[k] <= S.exact.value + Q.TIE_TOLERANCE) optimal.push(k);
      S.optimalCount = optimal.length;
      const reported = S.exact.x.reduce((acc, bit, i) => acc + (bit << i), 0);
      if (barEls) optimal.forEach((k) => barEls[k].classList.add("best"));
      svg("line", { x1: box.left, x2: box.right, y1: y(S.exact.value), y2: y(S.exact.value), class: "optline" }, chart);
      const px = box.left + ((reported + 0.5) / total) * width;
      svg("path", { d: "M" + px + " " + (box.bottom + 4) + "l-5 8h10z", class: "pointer" }, chart);
      const tag = svg("text", { x: clamp(px, box.left + 40, box.right - 40), y: box.bottom + 21, class: "pointer-label",
        "text-anchor": "middle" }, chart);
      tag.textContent = "assignment " + count(reported);
      note.textContent = optimal.length === 1
        ? "One assignment reaches the optimum."
        : optimal.length + " assignments reach the optimum; the lowest-numbered one is reported.";
    } else {
      // too many to hold: run the solver a slice per frame and draw what each slice saw
      const solver = Q.exactSolver(S.problem);
      const buckets = 245, per = Math.ceil(total / buckets);
      const seen = [];
      let lo = Infinity, hi = -Infinity;
      const redraw = (finished) => {
        const { chart, box, y } = landscapeFrame(lo, hi);
        let d = "", path = "", running = Infinity;
        seen.forEach((part, b) => {
          const px = box.left + b * 2 + 1;
          d += "M" + px + " " + y(part.hi).toFixed(1) + "V" + (y(part.lo) + 1).toFixed(1);
          if (part.lo < running - Q.TIE_TOLERANCE) {
            path += (path ? "H" + px : "M" + px + " " + y(part.lo).toFixed(1)) + "V" + y(part.lo).toFixed(1);
            running = part.lo;
          }
        });
        svg("path", { d, stroke: "#46526a", "stroke-width": 1.4, fill: "none" }, chart);
        svg("path", { d: path + "H" + (finished ? box.right : box.left + seen.length * 2), class: "runmin" }, chart);
        if (finished) svg("line", { x1: box.left, x2: box.right, y1: y(S.exact.value), y2: y(S.exact.value), class: "optline" }, chart);
      };
      while (!solver.done) {
        // one slice of the enumeration, then a frame to draw it
        await slices(() => {
          const part = solver.step(per);
          seen.push(part);
          if (part.lo < lo) lo = part.lo;
          if (part.hi > hi) hi = part.hi;
          return true;
        });
        lead.textContent = "scored " + count(solver.visited) + " of " + count(total) + " assignments";
        redraw(false);
      }
      S.exact = solver.result();
      redraw(true);
      note.textContent = "Each stroke spans the costs met in one slice of the enumeration, in the order visited.";
    }

    lead.textContent = "scored all " + count(total) + " assignments";
    const readout = $("exact-readout");
    readout.querySelector(".big").textContent = "Certified optimum " + num(S.exact.value);
    readout.querySelector(".x").textContent = "x = " + S.exact.x.join(n <= 10 ? " " : "");
    readout.classList.add("show");
    paintAssignment(S, S.exact.x);
    if (S.preset.kind === "maxcut") {
      say("Certified: no assignment costs less than " + num(S.exact.value) + ". For MaxCut that is a cut of weight " +
        num(-S.exact.value) + ", the edges now highlighted.");
    } else {
      say("Certified: no assignment costs less than " + num(S.exact.value) + ". The variables set to 1 are filled in.");
    }
    await sleep(4200);
  }

  /* Geometry of the circuit for this problem: every gate has the column
   * circuit.js schedules it in, which is how its depth is counted. */
  function circuitGeometry(S) {
    const n = S.n, depth = S.stats.depth, W = 1540, H = 204, labelW = 40;
    const colW = clamp((W - labelW - 10) / depth, 10, 78);
    const rowH = clamp((H - 8) / n, 7, 32);
    const top = (H - rowH * n) / 2;
    return {
      n, depth, colW, rowH, labelW,
      x: (column) => labelW + colW * (column + 0.5),
      y: (qubit) => top + rowH * (qubit + 0.5),
      boxW: rowH >= 17 && colW >= 26 ? Math.min(colW - 6, 38) : Math.min(colW - 4, Math.max(4, (rowH - 4) * 1.5)),
      boxH: rowH >= 17 ? Math.min(rowH - 5, 24) : Math.max(3, rowH - 4),
      labelled: rowH >= 17 && colW >= 26,
      right: labelW + colW * depth,
    };
  }

  function drawGate(op, geo, parent) {
    const color = op.layer < 0 ? NEUTRAL : LAYER_COLORS[op.layer % LAYER_COLORS.length];
    const group = svg("g", { class: "gate" }, parent);
    const cx = geo.x(op.column);
    if (op.type === "rzz") {
      const [a, b] = op.qubits;
      svg("line", { x1: cx, x2: cx, y1: geo.y(a), y2: geo.y(b), stroke: color }, group);
      const r = clamp(geo.rowH * 0.2, 2, 5);
      svg("circle", { cx, cy: geo.y(a), r, fill: color }, group);
      svg("circle", { cx, cy: geo.y(b), r, fill: color }, group);
    } else {
      const cy = geo.y(op.qubits[0]);
      const box = svg("rect", { x: cx - geo.boxW / 2, y: cy - geo.boxH / 2, width: geo.boxW, height: geo.boxH,
        rx: geo.labelled ? 3 : 1.5, stroke: color }, group);
      if (geo.labelled) {
        const label = svg("text", { x: cx, y: cy + 0.5, "font-size": geo.boxH > 18 ? 11 : 9 }, group);
        label.style.fill = color;
        label.textContent = op.label;
      } else {
        box.style.fill = color;
      }
    }
    return group;
  }

  /* A spark that travels from the problem graph to a gate's place. */
  function fly(from, to, duration) {
    const dot = svg("circle", { r: 4, class: "flight" }, overlay);
    const trail = svg("path", { class: "trail" }, overlay);
    const cx = lerp(from[0], to[0], 0.7), cy = lerp(from[1], to[1], 0.15);
    const at = (t) => [
      (1 - t) * (1 - t) * from[0] + 2 * (1 - t) * t * cx + t * t * to[0],
      (1 - t) * (1 - t) * from[1] + 2 * (1 - t) * t * cy + t * t * to[1],
    ];
    return tween(duration, (e) => {
      const [x, y] = at(e), [tx, ty] = at(Math.max(0, e - 0.12));
      dot.setAttribute("cx", x);
      dot.setAttribute("cy", y);
      trail.setAttribute("d", "M" + tx + " " + ty + "L" + x + " " + y);
    }).then(() => { dot.remove(); trail.remove(); }, (reason) => { dot.remove(); trail.remove(); throw reason; });
  }

  function flash(element) {
    element.classList.add("hot");
    window.setTimeout(() => element.classList.remove("hot"), 420);
  }

  async function sceneBuild(S) {
    setStage("build");
    openPanel("p-circuit");
    const n = S.n, p = S.rec.params.p, geo = circuitGeometry(S), board = $("circuit");
    board.textContent = "";
    const counters = { gates: 0, depth: 0, params: 0 };
    const show = () => {
      $("c-gates").textContent = counters.gates;
      $("c-depth").textContent = counters.depth;
      $("c-params").textContent = counters.params;
    };
    const place = (op, el) => {
      el.classList.add("in");
      counters.gates++;
      counters.depth = Math.max(counters.depth, op.column + 1);
      show();
    };
    const chip = (text, layer) => {
      const el = html("span", "chip", text, $("angles"));
      el.style.color = LAYER_COLORS[layer % LAYER_COLORS.length];
      counters.params++;
      show();
    };
    $("angles").textContent = "";
    $("c-qubits").textContent = n;
    show();

    // wires
    say("Now the circuit. One wire per variable: " + n + " qubits.");
    const wires = [];
    for (let q = 0; q < n; q++) {
      if (geo.rowH >= 11) {
        const label = svg("text", { x: 2, y: geo.y(q) + 3.5, class: "qlabel" }, board);
        label.textContent = "q" + q;
      }
      wires.push(svg("line", { x1: geo.labelW - 8, x2: geo.labelW - 8, y1: geo.y(q), y2: geo.y(q), class: "wire" }, board));
    }
    await tween(900, (e) => wires.forEach((w) => w.setAttribute("x2", lerp(geo.labelW - 8, geo.right + 6, e))));

    const els = S.ops.map((op) => drawGate(op, geo, board));
    const byLayer = (layer, types) => S.ops.map((op, k) => k).filter((k) => S.ops[k].layer === layer && types.includes(S.ops[k].type));

    // Hadamards
    say("A Hadamard on every qubit puts all " + count(Math.pow(2, n)) + " assignments in play at once, equally weighted.");
    const hadamards = byLayer(-1, ["h"]);
    await tween(clamp(90 * n, 500, 1100), (e, t) => {
      hadamards.forEach((k, i) => { if (t * hadamards.length >= i && !els[k].classList.contains("in")) place(S.ops[k], els[k]); });
    }, linear);
    await sleep(1900);

    for (let layer = 0; layer < p; layer++) {
      const phase = byLayer(layer, ["rz", "rzz"]), mixer = byLayer(layer, ["rx"]);
      const first = layer === 0;
      if (phase.length) {
        const linearTerms = phase.some((k) => S.ops[k].type === "rz");
        if (first) {
          say("Layer 1, the phase separator: each coupling in the problem becomes one RZZ gate on its two qubits" +
            (linearTerms ? ", and each linear term one RZ gate." : "."));
        } else if (layer === 1) {
          say("The next layer repeats the same pattern with its own pair of angles.");
        }
        chip("\u03b3" + (layer + 1), layer); // gamma
        const spacing = first ? clamp(3600 / phase.length, 28, 400) : clamp(1300 / phase.length, 10, 110);
        const flightTime = first && !reducedMotion ? clamp(spacing * 2.2, 260, 620) : 0;
        const pending = [];
        for (const k of phase) {
          const op = S.ops[k];
          const land = () => {
            place(op, els[k]);
            if (op.type === "rzz") flash(S.graph.edgeEls[edgeKey(op.qubits[0], op.qubits[1])]);
            else flash(S.graph.nodeEls[op.qubits[0]]);
          };
          if (flightTime && !clock.skip) {
            const G = S.graph;
            const source = op.type === "rzz" ? G.mids[edgeKey(op.qubits[0], op.qubits[1])] : G.pos[op.qubits[0]];
            const target = op.type === "rzz"
              ? [geo.x(op.column), (geo.y(op.qubits[0]) + geo.y(op.qubits[1])) / 2]
              : [geo.x(op.column), geo.y(op.qubits[0])];
            if (op.type === "rzz") flash(G.edgeEls[edgeKey(op.qubits[0], op.qubits[1])]);
            pending.push(
              fly(toStage($("graph"), source[0], source[1]), toStage(board, target[0], target[1]), flightTime)
                .then(land, (reason) => { if (reason !== CANCEL) throw reason; })
            );
          } else {
            land();
          }
          await sleep(spacing);
        }
        await Promise.all(pending);
        if (first) await sleep(900);
      }
      if (first) say("Then the mixer: an RX on every qubit, which lets weight move between assignments.");
      chip("\u03b2" + (layer + 1), layer); // beta
      await tween(first ? clamp(80 * n, 450, 900) : clamp(30 * n, 150, 400), (e, t) => {
        mixer.forEach((k, i) => { if (t * mixer.length >= i && !els[k].classList.contains("in")) place(S.ops[k], els[k]); });
      }, linear);
      await sleep(first ? 1700 : 500);
    }

    // settle on the tested numbers
    counters.gates = S.stats.total;
    counters.depth = S.stats.depth;
    counters.params = S.stats.params;
    show();
    $("legend").textContent = geo.labelled
      ? "Linked dots are RZZ gates, one per coupling. Boxes are single-qubit gates. Colour marks the layer."
      : "At this size gates are drawn as marks: squares for H, RZ and RX, linked dots for RZZ. Colour marks the layer.";
    say("The finished circuit: " + n + " qubits, " + S.stats.total + " gates, depth " + S.stats.depth + ", and " +
      S.stats.params + " free angles to tune. It is the same circuit the Python package builds with Qiskit.");
    await sleep(4200);
  }

  async function sceneRun(S) {
    setStage("run");
    const n = S.n, p = S.rec.params.p, optimum = S.exact ? S.exact.value : null;

    if (n > Q.SIMULATION_LIMIT) {
      openPanel("p-scan");
      const body = $("p-scan").querySelector(".body");
      body.querySelector("svg").style.display = "none";
      html("p", "skipped", "Not simulated: n = " + n + " is above the statevector limit of " + Q.SIMULATION_LIMIT +
        ". The recommendation, the certified optimum and the circuit are complete. In the Python package the " +
        "same circuit is a Qiskit QuantumCircuit, ready to be run on a simulator or on hardware.", body);
      say("At " + n + " qubits a statevector no longer fits comfortably in a browser tab, so the run is not simulated. " +
        "The builder says so instead of showing a number it did not compute.");
      await sleep(5200);
      return;
    }

    openPanel("p-scan");
    say("Now the run. " + SCAN_SAMPLES + " random settings of the angles are scored on a statevector simulator.");
    const chart = $("scan");
    chart.textContent = "";
    const box = { left: 8, right: 574, top: 12, bottom: 116 }, split = 0.6;
    const scan = Q.angleScan(S.ising, n, p, SCAN_SAMPLES, SCAN_SEED);
    const tracePath = svg("path", { class: "trace" }, chart), bestPath = svg("path", { class: "bestline" }, chart);
    const scanW = (box.right - box.left) * split;
    let y = null, lo = optimum, hi = null;
    const xs = (k) => box.left + (k / (SCAN_SAMPLES - 1)) * scanW;
    const redraw = () => {
      const trace = scan.trace;
      if (!trace.length) return;
      hi = Math.max.apply(null, trace);
      lo = optimum === null ? Math.min.apply(null, trace) : Math.min(optimum, Math.min.apply(null, trace));
      const pad = (hi - lo) * 0.08 || 1;
      y = (v) => box.bottom - ((v - (lo - pad)) / (hi - lo + 2 * pad)) * (box.bottom - box.top);
      let d = "", b = "", best = Infinity;
      trace.forEach((v, k) => {
        d += (k ? "L" : "M") + xs(k).toFixed(1) + " " + y(v).toFixed(1);
        if (v < best) { best = v; b += (b ? "H" + xs(k).toFixed(1) : "M" + xs(k).toFixed(1) + " " + y(v).toFixed(1)) + "V" + y(v).toFixed(1); }
      });
      tracePath.setAttribute("d", d);
      bestPath.setAttribute("d", b + "H" + xs(trace.length - 1).toFixed(1));
    };
    await tween(3400, (e, t) => {
      const want = Math.ceil(t * SCAN_SAMPLES);
      if (scan.trace.length < want) scan.step(Math.min(want - scan.trace.length, 6));
      redraw();
    }, linear);
    while (!scan.done) scan.step(SCAN_SAMPLES);
    redraw();
    S.scan = scan.result();
    svg("line", { x1: box.left + scanW + 8, x2: box.left + scanW + 8, y1: box.top - 6, y2: box.bottom, class: "divider" }, chart);
    const scanTag = svg("text", { x: box.left + 2, y: box.top - 2, class: "tag accent" }, chart);
    scanTag.textContent = "scan: best " + num(S.scan.best.value);
    $("r-scan").textContent = num(S.scan.best.value);
    $("r-scan").classList.add("show");
    await sleep(900);

    // refinement: computed a slice per frame, then drawn with its true length
    say("The best setting found is then refined, step by step, with a Nelder-Mead search.");
    const refiner = Q.angleRefiner(S.ising, n, p, S.scan.best, S.scan.costs), history = [S.scan.best.value];
    await slices(() => { refiner.step(); history.push(refiner.value); return refiner.done; }, 9);
    S.refined = refiner.result();
    const x0 = box.left + scanW + 16, refineW = box.right - x0 - 4;
    const rx = (k) => x0 + (k / Math.max(1, history.length - 1)) * refineW;
    const refinePath = svg("path", { class: "refine" }, chart);
    const dot = svg("circle", { r: 3.5, class: "dot", cx: rx(0), cy: y(history[0]) }, chart);
    await tween(2600, (e, t) => {
      const upto = Math.max(1, Math.round(t * (history.length - 1)));
      let d = "";
      for (let k = 0; k <= upto; k++) d += (k ? "L" : "M") + rx(k).toFixed(1) + " " + y(history[k]).toFixed(1);
      refinePath.setAttribute("d", d);
      dot.setAttribute("cx", rx(upto));
      dot.setAttribute("cy", y(history[upto]));
    }, linear);
    const refineTag = svg("text", { x: box.right, y: box.top - 2, class: "tag warn", "text-anchor": "end" }, chart);
    refineTag.textContent = "refined: " + num(S.refined.value) + " after " + refiner.iteration + " steps";
    if (optimum !== null) {
      svg("line", { x1: box.left, x2: box.right, y1: y(optimum), y2: y(optimum), class: "optline" }, chart);
      const tag = svg("text", { x: box.left + 178, y: box.top - 2, class: "tag good" }, chart);
      tag.textContent = "dashed: certified optimum " + num(optimum);
    }
    $("r-refined").textContent = num(S.refined.value);
    $("r-refined").classList.add("show");
    await sleep(1400);

    // what a measurement of the refined circuit returns
    openPanel("p-dist");
    const rows = Q.topBitstrings(S.ising, n, p, S.refined.gammas, S.refined.betas, 12);
    const isBest = (row) => optimum !== null && row.cost <= optimum + Q.TIE_TOLERANCE;
    const plot = $("dist");
    plot.textContent = "";
    const area = { left: 6, right: 526, top: 14, bottom: n <= 8 ? 98 : 116 };
    const step = (area.right - area.left) / rows.length, peak = rows[0].prob || 1;
    const bars = rows.map((row, k) => {
      const bar = svg("rect", { x: area.left + k * step + 4, width: step - 8, y: area.bottom, height: 0,
        class: "pbar" + (isBest(row) ? " best" : "") }, plot);
      if (n <= 8) {
        const label = svg("text", { x: area.left + (k + 0.5) * step, y: area.bottom + 13,
          class: "plabel" + (isBest(row) ? " best" : "") }, plot);
        label.textContent = row.bits.join("");
      }
      return bar;
    });
    svg("line", { x1: area.left, x2: area.right, y1: area.bottom + 0.5, y2: area.bottom + 0.5, class: "wire" }, plot);
    S.success = optimum === null ? null : Q.successProbability(S.ising, n, p, S.refined.gammas, S.refined.betas, optimum);
    say(S.success === null
      ? "Measuring the refined circuit returns these assignments most often."
      : "Measuring the refined circuit: the green outcomes are optimal. Together, optimal outcomes come up " +
        num(S.success * 100, 1) + "% of the time.");
    await tween(1500, (e) => {
      rows.forEach((row, k) => {
        const h = (row.prob / peak) * (area.bottom - area.top) * e;
        bars[k].setAttribute("y", area.bottom - h);
        bars[k].setAttribute("height", h);
      });
    }, easeOut);
    const peakTag = svg("text", { x: area.right, y: area.top - 3, class: "tag", "text-anchor": "end" }, plot);
    peakTag.textContent = "most likely: " + num(rows[0].prob * 100, 1) + "%" + (n <= 8 ? "" : ", top 12 of " + count(Math.pow(2, n)) + " outcomes");
    await sleep(3200);
  }

  async function sceneResult(S) {
    openPanel("p-result");
    const set = (id, text, tone) => {
      const el = $(id);
      el.textContent = text;
      el.classList.add("show");
      if (tone) el.classList.add(tone);
    };
    const simulated = S.n <= Q.SIMULATION_LIMIT;
    set("r-opt", S.exact ? num(S.exact.value) : "not available", "good");
    if (!simulated) {
      ["r-scan", "r-refined", "r-gap", "r-success"].forEach((id) => set(id, "not simulated"));
      say("The result: a certified optimum of " + num(S.exact.value) + " and a " + S.stats.depth +
        "-deep circuit, with the run left honestly blank.");
      await sleep(4400);
      return;
    }
    set("r-scan", num(S.scan.best.value));
    set("r-refined", num(S.refined.value), "warn");
    if (S.exact) {
      await sleep(350);
      set("r-gap", num(S.refined.value - S.exact.value));
      await sleep(350);
      set("r-success", num(S.success * 100, 1) + "%");
      say("The quantum result is never reported on its own. It is scored against the certified optimum: " +
        num(S.refined.value) + " against " + num(S.exact.value) + ".");
    } else {
      set("r-gap", "no certificate");
      set("r-success", "no certificate");
      say("Beyond the exact-solve limit the result is reported without a certificate, and labelled that way.");
    }
    await sleep(5200);
  }

  async function sceneEnd(S, refused) {
    setStage("end");
    document.querySelectorAll(".panel.now").forEach((panel) => panel.classList.remove("now"));
    await sleep(600);
    showCard((box) => {
      const headline = html("h1", null, null, box);
      const lines = refused
        ? ["When the problem does not fit,", "the builder says no."]
        : ["Recommended with reasons.", "Certified classically.", "Built gate by gate."];
      lines.forEach((line) => html("div", null, line, headline));
      const links = html("div", "links", null, box);
      anchor(html("div", null, null, links), "qopt-builder.vercel.app", HOME);
      anchor(html("div", null, null, links), "github.com/jemsbhai/qopt-builder", "https://github.com/jemsbhai/qopt-builder");
      const actions = html("div", "actions", null, box);
      const again = html("button", null, "Play again", actions);
      again.type = "button";
      again.addEventListener("click", () => start(true));
      anchor(actions, "Back to the main page", HOME);
    });
    say("");
  }

  // -------------------------------------------------------- the director

  function resetStage() {
    hideCard();
    overlay.textContent = "";
    caption.textContent = "";
    document.querySelectorAll(".panel").forEach((panel) => { panel.classList.add("idle"); panel.classList.remove("now"); });
    document.querySelectorAll("#stages li").forEach((li) => li.classList.remove("done", "now"));
    ["graph", "landscape", "circuit", "scan", "dist"].forEach((id) => { $(id).textContent = ""; $(id).style.display = ""; });
    document.querySelectorAll(".skipped").forEach((el) => el.remove());
    document.querySelectorAll(".marker").forEach((marker) => { marker.style.opacity = 0; marker.style.left = "0%"; });
    document.querySelectorAll(".band").forEach((band) => band.classList.remove("lit"));
    document.querySelectorAll(".gauge .value").forEach((value) => { value.textContent = ""; value.classList.remove("bad"); });
    const budgetGauge = $("g-budget");
    budgetGauge.classList.add("off");
    budgetGauge.querySelector(".value").textContent = "none set";
    budgetGauge.querySelector('[data-tick="budget"] span').textContent = "";
    $("decision").classList.remove("show");
    $("decision").querySelector(".algo").classList.remove("refused");
    $("exact-readout").classList.remove("show");
    ["problem-title", "problem-facts", "exact-lead", "exact-note", "legend", "angles"].forEach((id) => { $(id).textContent = ""; });
    ["c-qubits", "c-gates", "c-depth", "c-params"].forEach((id) => { $(id).textContent = "0"; });
    document.querySelectorAll("#result-list dd").forEach((dd) => { dd.textContent = ""; dd.className = ""; });
  }

  function prepare() {
    const preset = Q.PRESETS.find((entry) => entry.name === $("preset").value) || Q.PRESETS[0];
    const problem = Q.buildPreset(preset);
    let budget = null;
    try { budget = Q.parseBudget($("budget").value); } catch (err) { $("budget").value = ""; } // not a whole number: no budget
    const rec = Q.recommend(problem, budget);
    const ising = Q.isingTerms(problem);
    const S = {
      preset, problem, budget, rec, ising,
      n: problem.n,
      density: Q.density(problem),
      edges: couplingEdges(preset, problem),
      exact: null, scan: null, refined: null, success: null,
    };
    if (rec.algorithm !== "none") {
      S.stats = Q.circuitStats(problem, ising, rec.params.p);
      S.ops = S.stats.ops; // each already carries the column it is scheduled in
    }
    return S;
  }

  let finished = false;

  async function start(play) {
    clock.run++;
    clock.skip = false;
    finished = false;
    resetStage();
    const S = prepare();
    setPaused(!play);
    try {
      await stageOf(() => sceneTitle(S, play));
      await stageOf(() => sceneProblem(S));
      await stageOf(() => sceneRecommend(S));
      if (S.rec.algorithm === "none") {
        await sceneEnd(S, true);
      } else {
        await stageOf(() => sceneCertify(S));
        await stageOf(() => sceneBuild(S));
        await stageOf(async () => { await sceneRun(S); await sceneResult(S); });
        await sceneEnd(S, false);
      }
      setPaused(true);
      finished = true;
    } catch (reason) {
      if (reason !== CANCEL) {
        caption.textContent = "Something went wrong: " + (reason && reason.message ? reason.message : reason);
        throw reason;
      }
    }
  }

  /* "Next stage" skips to the end of the stage that is playing. */
  async function stageOf(scene) {
    try { await scene(); } finally { clock.skip = false; }
  }

  // ------------------------------------------------------------ controls

  function setPaused(paused) {
    clock.paused = paused;
    $("play").textContent = paused ? "Play" : "Pause";
  }

  for (const preset of Q.PRESETS) {
    const option = html("option", null, preset.title, $("preset"));
    option.value = preset.name;
  }
  if (Q.PRESETS.some((preset) => preset.name === query.get("preset"))) $("preset").value = query.get("preset");
  if (query.get("budget") !== null && query.get("budget") !== "") $("budget").value = query.get("budget");
  if (["0.5", "1", "2"].includes(query.get("speed"))) $("speed").value = query.get("speed");
  clock.speed = parseFloat($("speed").value);
  if (query.get("controls") === "0") document.body.classList.add("clean");
  if (query.get("captions") === "0") document.body.classList.add("no-captions");

  $("play").addEventListener("click", () => { if (finished) start(true); else setPaused(!clock.paused); });
  $("next").addEventListener("click", () => { clock.skip = true; });
  $("restart").addEventListener("click", () => start(true));
  $("speed").addEventListener("change", () => { clock.speed = parseFloat($("speed").value); });
  $("preset").addEventListener("change", () => start(true));
  $("budget").addEventListener("change", () => start(true));
  document.addEventListener("keydown", (event) => {
    if (event.target.closest && event.target.closest("button, input, select")) return;
    if (event.key === " ") { event.preventDefault(); if (finished) start(true); else setPaused(!clock.paused); }
    else if (event.key === "ArrowRight") clock.skip = true;
    else if (event.key === "r" || event.key === "R") start(true);
    else if (event.key === "h" || event.key === "H") document.body.classList.toggle("clean");
    else if (event.key === "c" || event.key === "C") document.body.classList.toggle("no-captions");
  });

  start(query.get("autoplay") === "1");
})();
