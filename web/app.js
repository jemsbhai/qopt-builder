/* UI wiring for the Quantum Optimisation Circuit Builder demo.
 * All computation comes from qubo.js, qaoa.js and circuit.js, which
 * are ports of the Python package in this repository. The preset
 * problems come from presets.js. */
(function () {
  "use strict";
  const Q = window.QOPT;

  const $ = (id) => document.getElementById(id);
  const presetSel = $("preset"), edgesInput = $("edges-input"), quboInput = $("qubo-input");
  const offsetInput = $("offset-input"), presetAbout = $("preset-about");
  const edgesBox = $("edges-box"), quboBox = $("qubo-box");

  const CUSTOM = {
    "custom-edges": {
      title: "Custom MaxCut edge list",
      about: "Edit the edge list below. The number of nodes is one more than the largest index used.",
    },
    "custom-qubo": {
      title: "Custom QUBO matrix",
      about: "Edit the matrix below. It can be symmetric or upper-triangular.",
    },
  };

  function addOption(value, label) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    presetSel.appendChild(option);
  }
  for (const preset of Q.PRESETS) addOption(preset.name, preset.title);
  for (const key of Object.keys(CUSTOM)) addOption(key, CUSTOM[key].title);

  function presetByName(name) { return Q.PRESETS.find((p) => p.name === name) || null; }
  function edgesText(edges) { return edges.map((e) => e.join(", ")).join("\n"); }
  function matrixText(m) { return m.map((r) => r.join(" ")).join("\n"); }

  /* A preset fills the input boxes, which stay editable. The custom
   * entries keep whatever is in the boxes. */
  function applyPreset(key) {
    const preset = presetByName(key);
    if (preset && preset.kind === "maxcut") {
      edgesInput.value = edgesText(preset.edges);
    } else if (preset) {
      quboInput.value = matrixText(preset.matrix);
      offsetInput.value = preset.offset ? String(preset.offset) : "";
    }
    presetAbout.textContent = preset ? preset.about : CUSTOM[key].about;
    syncMode();
  }

  function currentMode() {
    const preset = presetByName(presetSel.value);
    if (preset) return preset.kind === "maxcut" ? "edges" : "qubo";
    return presetSel.value === "custom-qubo" ? "qubo" : "edges";
  }

  function syncMode() {
    const mode = currentMode();
    edgesBox.hidden = mode !== "edges";
    quboBox.hidden = mode !== "qubo";
  }

  presetSel.addEventListener("change", () => applyPreset(presetSel.value));

  function parseEdges(text) {
    const edges = [];
    let n = 0;
    for (const line of text.split("\n")) {
      const t = line.trim();
      if (!t) continue;
      const parts = t.split(/[,\s]+/).filter(Boolean).map(Number);
      if (parts.length < 2 || parts.some(Number.isNaN)) throw new Error("Bad edge line: " + line);
      const w = parts.length >= 3 ? parts[2] : 1;
      edges.push([parts[0], parts[1], w]);
      n = Math.max(n, parts[0] + 1, parts[1] + 1);
    }
    if (!edges.length) throw new Error("No edges given.");
    return { edges, n };
  }

  function parseMatrix(text) {
    const rows = [];
    for (const line of text.split("\n")) {
      const t = line.trim();
      if (!t) continue;
      const parts = t.split(/[,\s]+/).filter(Boolean).map(Number);
      if (parts.some(Number.isNaN)) throw new Error("Bad matrix row: " + line);
      rows.push(parts);
    }
    if (!rows.length) throw new Error("Empty matrix.");
    return rows;
  }

  function parseOffset(text) {
    const t = text.trim();
    if (!t) return 0;
    const value = Number(t);
    if (Number.isNaN(value)) throw new Error("Bad offset: " + text);
    return value;
  }

  function graphSVG(edges, n, partition) {
    const size = 300, cx = size / 2, cy = size / 2, R = 108;
    const pos = Array.from({ length: n }, (_, i) => {
      const a = (2 * Math.PI * i) / n - Math.PI / 2;
      return [cx + R * Math.cos(a), cy + R * Math.sin(a)];
    });
    let s = '<svg viewBox="0 0 ' + size + " " + size + '" role="img" aria-label="Problem graph" style="max-width:300px;display:block;margin:0 auto">';
    for (const [i, j, w] of edges) {
      const cut = partition && partition[i] !== partition[j];
      s += '<line x1="' + pos[i][0] + '" y1="' + pos[i][1] + '" x2="' + pos[j][0] + '" y2="' + pos[j][1] +
        '" stroke="' + (cut ? "#22d3ee" : "#3a4356") + '" stroke-width="' + (cut ? 2.6 : 1.2) + '"/>';
      if (n <= 10) {
        const mx = (pos[i][0] + pos[j][0]) / 2, my = (pos[i][1] + pos[j][1]) / 2;
        s += '<text x="' + mx + '" y="' + (my - 3) + '" fill="#8b94a7" font-size="10" text-anchor="middle">' + w + "</text>";
      }
    }
    for (let i = 0; i < n; i++) {
      const fill = partition ? (partition[i] ? "#f59e0b" : "#22d3ee") : "#e2e8f0";
      s += '<circle cx="' + pos[i][0] + '" cy="' + pos[i][1] + '" r="13" fill="#12161f" stroke="' + fill + '" stroke-width="2"/>';
      s += '<text x="' + pos[i][0] + '" y="' + (pos[i][1] + 4) + '" fill="#e2e8f0" font-size="11" text-anchor="middle">' + i + "</text>";
    }
    return s + "</svg>";
  }

  function scanChartSVG(trace, refined, optimum) {
    const w = 520, h = 110, pad = 8;
    let lo = Math.min(...trace, optimum), hi = Math.max(...trace);
    if (hi - lo < 1e-9) hi = lo + 1;
    const X = (i) => pad + (i / (trace.length - 1)) * (w - 2 * pad);
    const Y = (v) => pad + (1 - (v - lo) / (hi - lo)) * (h - 2 * pad);
    let pts = "";
    trace.forEach((v, i) => { pts += X(i).toFixed(1) + "," + Y(v).toFixed(1) + " "; });
    let s = '<svg viewBox="0 0 ' + w + " " + h + '" role="img" aria-label="Angle scan trace" style="width:100%;display:block">';
    s += '<line x1="' + pad + '" y1="' + Y(optimum) + '" x2="' + (w - pad) + '" y2="' + Y(optimum) + '" stroke="#34d399" stroke-width="1.4" stroke-dasharray="5 4"/>';
    s += '<text x="' + (w - pad) + '" y="' + (Y(optimum) - 4) + '" fill="#34d399" font-size="10" text-anchor="end">certified optimum ' + optimum.toFixed(2) + "</text>";
    s += '<polyline fill="none" stroke="#22d3ee" stroke-width="1.2" points="' + pts + '"/>';
    if (refined !== null && refined !== undefined) {
      s += '<circle cx="' + (w - pad) + '" cy="' + Y(refined) + '" r="4" fill="#f59e0b"/>';
      s += '<text x="' + (w - pad - 8) + '" y="' + (Y(refined) + 13) + '" fill="#f59e0b" font-size="10" text-anchor="end">refined ' + refined.toFixed(2) + "</text>";
    }
    return s + "</svg>";
  }

  function show(id, on) { $(id).hidden = !on; }
  const tick = () => new Promise((r) => setTimeout(r, 10));

  async function run() {
    const btn = $("run"), status = $("status");
    btn.disabled = true;
    for (const id of ["graph-card", "rec-card", "refusal-card", "exact-card", "circuit-card", "qaoa-card"]) show(id, false);
    try {
      status.textContent = "Parsing problem...";
      await tick();
      const mode = currentMode();
      let problem, edges = null;
      if (mode === "edges") {
        const parsed = parseEdges(edgesInput.value);
        edges = parsed.edges;
        problem = Q.maxcutQUBO(edges, parsed.n);
        problem.name = "maxcut";
      } else {
        problem = Q.makeQUBO(parseMatrix(quboInput.value), parseOffset(offsetInput.value), "qubo");
      }
      const n = problem.n;
      const budgetRaw = $("budget").value.trim();
      const budget = budgetRaw === "" ? null : parseInt(budgetRaw, 10);

      show("graph-card", true);
      if (edges) {
        $("graph-svg").innerHTML = graphSVG(edges, n, null);
        $("problem-stats").innerHTML =
          "MaxCut on <b>" + n + "</b> nodes, <b>" + edges.length + "</b> edges. Coupling density <b>" +
          Q.density(problem).toFixed(3) + "</b>. Node colours and highlighted edges appear after the exact solve: they show the certified optimal partition and its cut.";
      } else {
        $("graph-svg").innerHTML = "";
        $("problem-stats").innerHTML =
          "QUBO on <b>" + n + "</b> variables" +
          (problem.offset ? ", constant offset <b>" + problem.offset + "</b>" : "") +
          ". Coupling density <b>" + Q.density(problem).toFixed(3) + "</b>.";
      }

      status.textContent = "Running the recommendation engine...";
      await tick();
      const rec = Q.recommend(problem, budget);
      if (rec.algorithm === "none") {
        show("refusal-card", true);
        $("refusal-reasons").innerHTML = rec.rationale.map((r) => "<li>" + r + "</li>").join("");
        status.textContent = "The builder refused. See the refusal card.";
        return;
      }
      show("rec-card", true);
      $("rec-algo").textContent = rec.algorithm + "  (p = " + rec.params.p + ")";
      $("rec-reasons").innerHTML = rec.rationale.map((r) => "<li>" + r + "</li>").join("");

      let optimum = null;
      if (n <= Q.EXACT_CLASSICAL_LIMIT) {
        status.textContent = "Solving exactly by enumeration (2^" + n + " states)...";
        await tick();
        const t0 = performance.now();
        const bf = Q.bruteForce(problem);
        optimum = bf.value;
        const ms = Math.max(1, Math.round(performance.now() - t0));
        show("exact-card", true);
        $("exact-line").innerHTML =
          "Certified optimum cost <b>" + optimum.toFixed(3) + "</b> at x = [" + bf.x.join(", ") +
          "], found in " + ms + " ms." + (edges ? " For MaxCut this is a maximum cut of weight <b>" + (-optimum).toFixed(3) + "</b>." : "");
        if (edges) $("graph-svg").innerHTML = graphSVG(edges, n, bf.x);
      }

      const ising = Q.isingTerms(problem);
      const stats = Q.circuitStats(problem, ising, rec.params.p);
      show("circuit-card", true);
      const counts = Object.entries(stats.counts).map(([k, v]) => v + " " + k.toUpperCase()).join(", ");
      const angleNote = stats.params === 2 * rec.params.p
        ? "gammas and betas for p = " + rec.params.p
        : "betas only for p = " + rec.params.p + ", because this cost function has no terms for a gamma to act on";
      $("circuit-stats").innerHTML =
        "<b>" + stats.qubits + "</b> qubits, depth <b>" + stats.depth + "</b>, <b>" + stats.total +
        "</b> gates (" + counts + "), <b>" + stats.params + "</b> free parameter" + (stats.params === 1 ? "" : "s") + " (" + angleNote + ").";
      $("circuit-svg").innerHTML = Q.renderCircuitSVG(stats.ops, n, Q.schedule(stats.ops, n));

      if (n <= Q.SIMULATION_LIMIT) {
        status.textContent = "Scanning 200 angle samples on the statevector simulator...";
        await tick();
        const scan = Q.scanAngles(ising, n, rec.params.p, 200, 20261110);
        status.textContent = "Refining the best scanned angles...";
        await tick();
        const refined = Q.refineAngles(ising, n, rec.params.p, scan.best, scan.costs);
        const top = Q.topBitstrings(ising, n, rec.params.p, refined.gammas, refined.betas, 8);
        show("qaoa-card", true);
        let line = "Coarse scan best expectation <b>" + scan.best.value.toFixed(3) + "</b>. After refinement: <b>" +
          refined.value.toFixed(3) + "</b>.";
        if (optimum !== null) line += " Certified optimum <b>" + optimum.toFixed(3) + "</b>; gap after refinement <b>" + (refined.value - optimum).toFixed(3) + "</b>.";
        line += "<br>Refined angles: gammas [" + refined.gammas.map((g) => g.toFixed(3)).join(", ") +
          "], betas [" + refined.betas.map((b) => b.toFixed(3)).join(", ") + "].";
        $("qaoa-line").innerHTML = line;
        $("scan-chart").innerHTML = optimum !== null
          ? scanChartSVG(scan.trace, refined.value, optimum)
          : scanChartSVG(scan.trace, refined.value, scan.best.value);
        const tbody = $("top-table").querySelector("tbody");
        tbody.innerHTML = top.map((row) => {
          const isOpt = optimum !== null && Math.abs(row.cost - optimum) < 1e-9;
          return "<tr" + (isOpt ? ' class="optimal"' : "") + "><td>" + row.bits.join("") + "</td><td>" +
            row.prob.toFixed(4) + "</td><td>" + row.cost.toFixed(3) + "</td><td>" + (isOpt ? "optimal" : "") + "</td></tr>";
        }).join("");
        status.textContent = "Done.";
      } else {
        show("qaoa-card", true);
        $("qaoa-line").textContent = "n = " + n + " is above the statevector limit of " + Q.SIMULATION_LIMIT +
          ", so no QAOA run was simulated. The recommendation and the circuit above are complete" +
          (optimum !== null ? ", and so is the certified optimum" : "") +
          ". In the Python package the same circuit is a Qiskit QuantumCircuit, ready to be run on a simulator or on hardware.";
        $("scan-chart").innerHTML = "";
        $("top-table").querySelector("tbody").innerHTML = "";
        status.textContent = "Done (simulation skipped for n > " + Q.SIMULATION_LIMIT + ").";
      }
    } catch (err) {
      status.textContent = "Error: " + err.message;
    } finally {
      btn.disabled = false;
    }
  }

  $("run").addEventListener("click", run);
  applyPreset("demo");
  run();
})();
