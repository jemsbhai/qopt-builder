/* QAOA circuit structure: op list, gate counts, depth, SVG diagram.
 * Gate order mirrors qopt_builder/build.py: H on all qubits, then per
 * layer, for each i, RZ on i followed by RZZ on (i, j) for j > i, then
 * RX on all qubits. Depth is the earliest-layer schedule: a gate takes
 * the first layer after every earlier gate sharing one of its wires,
 * which is how Qiskit counts circuit depth. */
(function (global) {
  "use strict";

  function circuitOps(Q, ising, p) {
    const n = Q.n, { h, J } = ising;
    const ops = [];
    for (let q = 0; q < n; q++) ops.push({ type: "h", qubits: [q], layer: -1, label: "H" });
    for (let layer = 0; layer < p; layer++) {
      for (let i = 0; i < n; i++) {
        if (Math.abs(h[i]) > 1e-12) ops.push({ type: "rz", qubits: [i], layer, label: "Rz" });
        for (let j = i + 1; j < n; j++) {
          if (Math.abs(J[i][j]) > 1e-12) ops.push({ type: "rzz", qubits: [i, j], layer, label: "Rzz" });
        }
      }
      for (let q = 0; q < n; q++) ops.push({ type: "rx", qubits: [q], layer, label: "Rx" });
    }
    return ops;
  }

  function schedule(ops, n) {
    const wireFree = new Array(n).fill(0);
    const columns = [];
    let depth = 0;
    for (const op of ops) {
      let col = 0;
      for (const q of op.qubits) col = Math.max(col, wireFree[q]);
      op.column = col;
      for (const q of op.qubits) wireFree[q] = col + 1;
      depth = Math.max(depth, col + 1);
      if (!columns[col]) columns[col] = [];
      columns[col].push(op);
    }
    return { depth, columns };
  }

  function circuitStats(Q, ising, p) {
    const ops = circuitOps(Q, ising, p);
    const { depth } = schedule(ops, Q.n);
    const counts = {};
    for (const op of ops) counts[op.type] = (counts[op.type] || 0) + 1;
    // Free parameters are the angles that appear in a gate, which is how
    // Qiskit counts them: p betas always, and p gammas unless the cost
    // function has no terms at all (then there is no RZ or RZZ to carry one).
    const params = (counts.rz || counts.rzz) ? 2 * p : p;
    return { qubits: Q.n, depth, total: ops.length, counts, params, ops };
  }

  function renderCircuitSVG(ops, n, depthInfo) {
    const colW = 34, rowH = 26, padX = 44, padTop = 26;
    const width = padX * 2 + depthInfo.depth * colW;
    const height = padTop * 2 + n * rowH;
    const y = (q) => padTop + q * rowH + rowH / 2;
    const x = (c) => padX + c * colW + colW / 2;
    let s = '<svg viewBox="0 0 ' + width + " " + height + '" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="QAOA circuit diagram">';
    for (let q = 0; q < n; q++) {
      s += '<line x1="' + (padX - 14) + '" y1="' + y(q) + '" x2="' + (width - padX + 14) + '" y2="' + y(q) + '" stroke="#3a4356" stroke-width="1"/>';
      s += '<text x="14" y="' + (y(q) + 4) + '" fill="#8b94a7" font-size="11" font-family="monospace">q' + q + "</text>";
    }
    const layerColors = ["#22d3ee", "#f59e0b", "#a78bfa", "#34d399", "#f472b6", "#facc15"];
    for (const op of ops) {
      const color = op.layer < 0 ? "#e2e8f0" : layerColors[op.layer % layerColors.length];
      if (op.type === "rzz") {
        const [a, b] = op.qubits;
        s += '<line x1="' + x(op.column) + '" y1="' + y(a) + '" x2="' + x(op.column) + '" y2="' + y(b) + '" stroke="' + color + '" stroke-width="1.6"/>';
        for (const q of op.qubits) s += '<circle cx="' + x(op.column) + '" cy="' + y(q) + '" r="4.5" fill="' + color + '"/>';
      } else {
        const q = op.qubits[0];
        s += '<rect x="' + (x(op.column) - 11) + '" y="' + (y(q) - 10) + '" width="22" height="20" rx="3" fill="#12161f" stroke="' + color + '" stroke-width="1.4"/>';
        s += '<text x="' + x(op.column) + '" y="' + (y(q) + 3.5) + '" fill="' + color + '" font-size="9.5" text-anchor="middle" font-family="monospace">' + op.label + "</text>";
      }
    }
    s += "</svg>";
    return s;
  }

  const api = { circuitOps, schedule, circuitStats, renderCircuitSVG };
  global.QOPT = Object.assign(global.QOPT || {}, api);
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
