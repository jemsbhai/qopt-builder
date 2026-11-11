/* QUBO problem intake and the recommendation engine.
 * Faithful JavaScript port of qopt_builder/problems.py and
 * qopt_builder/recommend.py. A problem is a matrix Q plus an offset,
 * minimised over binary x: cost(x) = x^T Q x + offset. */
(function (global) {
  "use strict";

  function makeQUBO(matrix, offset, name) {
    const n = matrix.length;
    for (const row of matrix) {
      if (row.length !== n) throw new Error("QUBO matrix must be square");
    }
    return { matrix: matrix.map((r) => r.slice()), offset: offset || 0, name: name || "qubo", n };
  }

  function cost(Q, x) {
    const M = Q.matrix, n = Q.n;
    let c = Q.offset;
    for (let i = 0; i < n; i++) {
      if (!x[i]) continue;
      c += M[i][i];
      for (let j = i + 1; j < n; j++) {
        if (x[j]) c += (M[i][j] + M[j][i]) * x[j];
      }
    }
    return c;
  }

  /* Fraction of variable pairs {i, j} that are coupled. A pair is
   * coupled when Q[i][j] + Q[j][i] is nonzero, so a symmetric layout and
   * an upper-triangular layout of the same cost function report the same
   * value. For a graph with m edges this is the usual edge density
   * m / (n (n - 1) / 2). Mirrors QUBO.density() in problems.py. */
  function density(Q) {
    const n = Q.n;
    if (n < 2) return 0;
    let coupled = 0;
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++)
        if (Math.abs(Q.matrix[i][j] + Q.matrix[j][i]) > 1e-12) coupled++;
    return coupled / (n * (n - 1) / 2);
  }

  /* Assignments whose cost is within this distance of the minimum count
   * as tied. Same constant as TIE_TOLERANCE in problems.py. */
  const TIE_TOLERANCE = 1e-9;

  /* Exact optimum by Gray-code enumeration: flipping one bit at a
   * time keeps the incremental cost update O(n) per state.
   *
   * Assignment number k has x_i = bit i of k (the Gray code word is
   * that number). When several assignments are optimal, the one with
   * the lowest number is reported, which is the rule brute_force() in
   * problems.py follows, so both ports report the same x. The value
   * returned is the cost of that x computed directly, not the running
   * total, so it carries no accumulated rounding. */
  function bruteForce(Q) {
    const n = Q.n, M = Q.matrix;
    const x = new Array(n).fill(0);
    let cur = Q.offset;
    let best = cur, bestIndex = 0;
    let prevGray = 0;
    const total = 1 << n;
    for (let mask = 1; mask < total; mask++) {
      const gray = mask ^ (mask >> 1);
      const diff = gray ^ prevGray;
      const k = 31 - Math.clz32(diff);
      prevGray = gray;
      let s = M[k][k];
      for (let j = 0; j < n; j++) {
        if (j !== k && x[j]) s += M[k][j] + M[j][k];
      }
      cur += x[k] ? -s : s;
      x[k] = x[k] ? 0 : 1;
      if (cur < best - TIE_TOLERANCE) {
        best = cur; bestIndex = gray;
      } else if (cur <= best + TIE_TOLERANCE && gray < bestIndex) {
        bestIndex = gray;
      }
    }
    const bestX = [];
    for (let i = 0; i < n; i++) bestX.push((bestIndex >> i) & 1);
    return { value: cost(Q, bestX), x: bestX };
  }

  function maxcutQUBO(edges, n) {
    const q = Array.from({ length: n }, () => new Array(n).fill(0));
    for (const [i, j, w] of edges) {
      q[i][i] -= w;
      q[j][j] -= w;
      q[i][j] += 2 * w;
    }
    return makeQUBO(q, 0, "maxcut");
  }

  /* QUBO -> Ising: cost = sum_i h_i z_i + sum_{i<j} J_ij z_i z_j + const,
   * under x = (1 - z) / 2. Mirrors build.py exactly. */
  function isingTerms(Q) {
    const n = Q.n, q = Q.matrix;
    const h = new Array(n).fill(0);
    const J = Array.from({ length: n }, () => new Array(n).fill(0));
    let const_ = Q.offset;
    for (let i = 0; i < n; i++) {
      const_ += q[i][i] / 2;
      h[i] -= q[i][i] / 2;
      for (let j = i + 1; j < n; j++) {
        const w = q[i][j] + q[j][i];
        const_ += w / 4;
        h[i] -= w / 4;
        h[j] -= w / 4;
        J[i][j] = w / 4;
      }
    }
    return { h, J, offset: const_ };
  }

  /* Fixed-point text that matches Python's format(x, ".{digits}f").
   * toFixed rounds an exact tie away from zero; Python rounds it to the
   * even digit. A double is an exact tie at `digits` decimals only when
   * it is an odd multiple of 2^-(digits+1), so that is the case handled
   * here. Example: a density of 0.125 prints as 0.12, as it does in the
   * Python rationale, where toFixed alone would print 0.13. */
  function fixed(x, digits) {
    if (Object.is(x, -0)) return "-" + (0).toFixed(digits); // Python keeps the sign of -0.0
    const a = Math.abs(x);
    const scaled = a * Math.pow(2, digits + 1);
    if (Number.isInteger(scaled) && scaled % 2 === 1) {
      const lower = Math.floor(a * Math.pow(10, digits));
      const even = lower % 2 === 0 ? lower : lower + 1;
      return (x < 0 ? "-" : "") + (even / Math.pow(10, digits)).toFixed(digits);
    }
    return x.toFixed(digits);
  }

  const EXACT_CLASSICAL_LIMIT = 24;

  function recommend(Q, qubitBudget) {
    const n = Q.n, d = density(Q);
    const reasons = [];
    let base;
    if (n <= EXACT_CLASSICAL_LIMIT) {
      reasons.push(
        "n=" + n + " <= " + EXACT_CLASSICAL_LIMIT + ": an exact classical solve is cheap " +
        "and certifies the quantum result; recommending exact with QAOA built " +
        "alongside for comparison.");
      base = "exact";
    } else {
      base = "quantum";
      reasons.push("n=" + n + ": exact enumeration is not feasible (2^" + n + " states).");
    }
    if (qubitBudget !== null && qubitBudget !== undefined && n > qubitBudget) {
      reasons.push(
        "n=" + n + " exceeds the qubit budget of " + qubitBudget + ": no full-width " +
        "circuit fits; the builder will refuse rather than silently truncate.");
      return { algorithm: "none", params: { n }, rationale: reasons };
    }
    let p;
    if (d > 0.5) {
      p = 3;
      reasons.push("coupling density " + fixed(d, 2) + " is high: deeper QAOA (p=" + p + ") earns its depth on dense graphs.");
    } else if (d > 0.15) {
      p = 2;
      reasons.push("coupling density " + fixed(d, 2) + " is moderate: QAOA p=" + p + ".");
    } else {
      p = 1;
      reasons.push("coupling density " + fixed(d, 2) + " is sparse: QAOA p=" + p + " first; depth buys little on near-tree coupling graphs.");
    }
    if (base === "exact") return { algorithm: "exact+qaoa", params: { p, also_build: "qaoa" }, rationale: reasons };
    return { algorithm: "qaoa", params: { p }, rationale: reasons };
  }

  const api = { makeQUBO, cost, density, bruteForce, maxcutQUBO, isingTerms, recommend, fixed, EXACT_CLASSICAL_LIMIT, TIE_TOLERANCE };
  global.QOPT = Object.assign(global.QOPT || {}, api);
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
