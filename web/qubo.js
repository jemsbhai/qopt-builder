/* QUBO problem intake and the recommendation engine.
 * Faithful JavaScript port of qopt_builder/problems.py and
 * qopt_builder/recommend.py. A problem is a matrix Q plus an offset,
 * minimised over binary x: cost(x) = x^T Q x + offset.
 *
 * Input is checked on the way in, by the same checks in the same order
 * and with the same wording as problems.py. tests/test_web_parity.py
 * gives both ports the same bad inputs and requires the same message
 * from each. */
(function (global) {
  "use strict";

  /* Q[i][j] and Q[j][i] count as equal when they differ by no more than
   * this, relative to the larger of the two (or to 1). */
  const SYMMETRY_TOLERANCE = 1e-9;
  /* An entry no larger than this counts as zero when deciding whether a
   * matrix is triangular. density() uses the same size for "not coupled". */
  const ZERO_TOLERANCE = 1e-12;

  const isFiniteNumber = (v) => typeof v === "number" && Number.isFinite(v);
  const isWhole = (v) => typeof v === "number" && Number.isInteger(v);

  /* A problem from its matrix and constant offset. The matrix is a
   * square table of finite numbers, symmetric or triangular; it is
   * copied. The checks run in a fixed order: it is a list of rows; it
   * is not empty; every row is as long as the number of rows; every
   * entry is a number, and finite; the layout is symmetric or
   * triangular. */
  function makeQUBO(matrix, offset, name) {
    if (!Array.isArray(matrix) || matrix.some((row) => !Array.isArray(row))) {
      throw new Error("QUBO matrix must be a square table of numbers, given as a list of rows.");
    }
    if (!matrix.some((row) => row.length)) {
      throw new Error("QUBO matrix is empty; a problem needs at least one variable.");
    }
    const n = matrix.length;
    for (let i = 0; i < n; i++) {
      if (matrix[i].length !== n) {
        throw new Error("QUBO matrix must be square: number of rows is " + n + ", but row " + i +
          " has length " + matrix[i].length + ".");
      }
    }
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const value = matrix[i][j];
        if (typeof value !== "number") throw new Error("QUBO matrix entry [" + i + "][" + j + "] is not a number.");
        if (!Number.isFinite(value)) throw new Error("QUBO matrix entry [" + i + "][" + j + "] is not finite.");
      }
    }
    let above = false, below = false;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (Math.abs(matrix[i][j]) > ZERO_TOLERANCE) above = true;
        if (Math.abs(matrix[j][i]) > ZERO_TOLERANCE) below = true;
      }
    }
    if (above && below) {
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const a = matrix[i][j], b = matrix[j][i];
          if (Math.abs(a - b) > SYMMETRY_TOLERANCE * Math.max(1, Math.abs(a), Math.abs(b))) {
            throw new Error("QUBO matrix must be symmetric or triangular. Entries [" + i + "][" + j + "] and [" +
              j + "][" + i + "] differ, and there are nonzero entries on both sides of the diagonal.");
          }
        }
      }
    }
    const constant = offset === undefined ? 0 : offset;
    if (!isFiniteNumber(constant)) throw new Error("QUBO offset must be a finite number.");
    return { matrix: matrix.map((r) => r.slice()), offset: constant, name: name || "qubo", n };
  }

  /* Cost of the assignment x: one entry per variable, each 0 or 1
   * (true and false count as 1 and 0). */
  function cost(Q, x) {
    const M = Q.matrix, n = Q.n;
    if (!Array.isArray(x)) throw new Error("Assignment must be a list of 0s and 1s, one per variable.");
    if (x.length !== n) {
      throw new Error("Assignment must have one entry per variable: this problem has " + n + ", but x has " + x.length + ".");
    }
    for (let i = 0; i < n; i++) {
      const v = x[i];
      if (v !== 0 && v !== 1 && v !== true && v !== false) {
        throw new Error("Assignment entries must be 0 or 1; entry " + i + " is not.");
      }
    }
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
        if (Math.abs(Q.matrix[i][j] + Q.matrix[j][i]) > ZERO_TOLERANCE) coupled++;
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
   * total, so it carries no accumulated rounding.
   *
   * The solver is resumable so that a page can run it a slice at a
   * time and draw its progress: step(count) scores up to `count` more
   * assignments and returns the lowest and highest cost it saw in that
   * slice. bruteForce() below runs the same code to completion.
   *
   * Assignment numbers are handled with 32-bit operations, which are
   * exact up to n = 30; a larger problem is refused here instead of
   * being enumerated wrongly. (The page solves exactly only up to
   * EXACT_CLASSICAL_LIMIT, and brute_force() in problems.py has no
   * such bound.) */
  const EXACT_SOLVER_MAX_N = 30;

  function exactSolver(Q) {
    const n = Q.n, M = Q.matrix;
    if (n > EXACT_SOLVER_MAX_N) {
      throw new Error("The exact solver in the browser stops at n = " + EXACT_SOLVER_MAX_N +
        "; this problem has n = " + n + ".");
    }
    const x = new Array(n).fill(0);
    const total = Math.pow(2, n);
    let cur = Q.offset;
    let best = cur, bestIndex = 0;
    let prevGray = 0, mask = 1; // the all-zero assignment is scored on creation

    function step(count) {
      const end = Math.min(total, mask + count);
      let lo = Infinity, hi = -Infinity;
      let c = cur, b = best, bi = bestIndex, pg = prevGray, m = mask;
      for (; m < end; m++) {
        const gray = m ^ (m >> 1);
        const k = 31 - Math.clz32(gray ^ pg);
        pg = gray;
        let s = M[k][k];
        for (let j = 0; j < n; j++) {
          if (j !== k && x[j]) s += M[k][j] + M[j][k];
        }
        c += x[k] ? -s : s;
        x[k] = x[k] ? 0 : 1;
        if (c < b - TIE_TOLERANCE) {
          b = c; bi = gray;
        } else if (c <= b + TIE_TOLERANCE && gray < bi) {
          bi = gray;
        }
        if (c < lo) lo = c;
        if (c > hi) hi = c;
      }
      cur = c; best = b; bestIndex = bi; prevGray = pg; mask = m;
      return { lo, hi };
    }

    function result() {
      const bestX = [];
      for (let i = 0; i < n; i++) bestX.push((bestIndex >> i) & 1);
      return { value: cost(Q, bestX), x: bestX };
    }

    return {
      total,
      step,
      result,
      get visited() { return mask; },
      get done() { return mask >= total; },
      get best() { return best; }, // lowest running cost so far
    };
  }

  function bruteForce(Q) {
    const solver = exactSolver(Q);
    solver.step(Infinity);
    return solver.result();
  }

  /* MaxCut as a minimisation QUBO: cut(x) = sum over edges of
   * w (x_i + x_j - 2 x_i x_j), and the cost is -cut.
   *
   * `edges` is a list of [i, j, weight], or [i, j] for weight 1, on
   * nodes numbered 0 to n - 1. An edge may be written in either
   * direction: its coupling is stored above the diagonal both ways.
   * Edges given more than once add up. Mirrors maxcut_qubo() in
   * problems.py, checks and wording included. */
  function maxcutQUBO(edges, n) {
    if (!isWhole(n) || n < 1) throw new Error("Number of nodes must be a positive whole number.");
    if (!Array.isArray(edges)) throw new Error("Edges must be a list of (i, j) or (i, j, weight) entries.");
    const q = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let k = 0; k < edges.length; k++) {
      const edge = edges[k];
      if (!Array.isArray(edge) || (edge.length !== 2 && edge.length !== 3)) {
        throw new Error("edges[" + k + "] must be (i, j) or (i, j, weight).");
      }
      const i = edge[0], j = edge[1];
      if (!isWhole(i) || !isWhole(j)) throw new Error("edges[" + k + "]: node numbers must be whole numbers.");
      for (const node of [i, j]) {
        if (node < 0 || node >= n) {
          throw new Error("edges[" + k + "]: node " + node + " is outside the range 0 to " + (n - 1) + ".");
        }
      }
      if (i === j) throw new Error("edges[" + k + "] joins node " + i + " to itself.");
      const w = edge.length === 3 ? edge[2] : 1;
      if (!isFiniteNumber(w)) throw new Error("edges[" + k + "]: weight must be a finite number.");
      q[i][i] -= w;
      q[j][j] -= w;
      q[Math.min(i, j)][Math.max(i, j)] += 2 * w;
    }
    return makeQUBO(q, 0, "maxcut");
  }

  /* ------------------------------------------------- text typed into the page
   * The page reads problems from text boxes. These functions turn that
   * text into the arguments of maxcutQUBO() and makeQUBO(), and say
   * which line is at fault when they cannot. They have no counterpart
   * in the Python package, which takes lists and arrays, not text. */

  // a plain decimal number: optional sign, digits, optional point and exponent
  const NUMBER_TEXT = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;

  function numberFrom(token, where) {
    if (!NUMBER_TEXT.test(token)) throw new Error(where + '"' + token + '" is not a number.');
    const value = Number(token);
    if (!Number.isFinite(value)) throw new Error(where + '"' + token + '" is too large to use.');
    return value;
  }

  function tooManyVariables(n, limit) {
    return "This page handles problems of up to " + limit + " variables, and this one has " + n +
      ". The Python package has no such limit.";
  }

  /* An edge list, one edge per line: two node numbers and an optional
   * weight (1 when left out), separated by spaces or commas. Blank
   * lines are skipped. Returns { edges, n }, with n one more than the
   * largest node number used. `maxVariables`, when given, caps n. */
  function parseEdgeList(text, maxVariables) {
    const edges = [];
    let n = 0;
    String(text).split(/\r?\n/).forEach((line, index) => {
      const tokens = line.trim().split(/[,\s]+/).filter(Boolean);
      if (!tokens.length) return;
      const where = "Line " + (index + 1) + ": ";
      if (tokens.length < 2 || tokens.length > 3) {
        throw new Error(where + "an edge is two node numbers and an optional weight, for example 0 1 2.5.");
      }
      const values = tokens.map((token) => numberFrom(token, where));
      const i = values[0], j = values[1];
      if (!Number.isInteger(i) || !Number.isInteger(j) || i < 0 || j < 0) {
        throw new Error(where + "node numbers must be whole numbers, 0 or more.");
      }
      if (i === j) throw new Error(where + "an edge cannot join node " + i + " to itself.");
      edges.push([i, j, values.length === 3 ? values[2] : 1]);
      n = Math.max(n, i + 1, j + 1);
    });
    if (!edges.length) {
      throw new Error("No edges given. Enter one edge per line: two node numbers and an optional weight.");
    }
    if (maxVariables !== undefined && n > maxVariables) throw new Error(tooManyVariables(n, maxVariables));
    return { edges, n };
  }

  /* A matrix, one row per line, numbers separated by spaces or commas.
   * Blank lines are skipped. Returns the rows, checked to be square;
   * makeQUBO() then checks the layout. `maxVariables`, when given, caps
   * the number of rows. */
  function parseMatrix(text, maxVariables) {
    const rows = [], lineOf = [];
    String(text).split(/\r?\n/).forEach((line, index) => {
      const tokens = line.trim().split(/[,\s]+/).filter(Boolean);
      if (!tokens.length) return;
      rows.push(tokens.map((token) => numberFrom(token, "Line " + (index + 1) + ": ")));
      lineOf.push(index + 1);
    });
    if (!rows.length) {
      throw new Error("No matrix given. Enter one row per line, numbers separated by spaces or commas.");
    }
    if (maxVariables !== undefined && rows.length > maxVariables) {
      throw new Error(tooManyVariables(rows.length, maxVariables));
    }
    rows.forEach((row, k) => {
      if (row.length !== rows.length) {
        throw new Error("Line " + lineOf[k] + " has " + row.length + (row.length === 1 ? " number" : " numbers") +
          ", but the matrix has " + rows.length + (rows.length === 1 ? " row" : " rows") +
          ". A QUBO matrix is square: every row needs as many numbers as there are rows.");
      }
    });
    return rows;
  }

  /* The constant offset box: empty means 0. */
  function parseOffset(text) {
    const token = String(text).trim();
    return token === "" ? 0 : numberFrom(token, "Offset: ");
  }

  /* The qubit budget box: empty means no budget (null); otherwise a
   * whole number of qubits, 0 or more. */
  function parseBudget(text) {
    const token = String(text).trim();
    if (token === "") return null;
    if (!/^\d+$/.test(token) || !Number.isSafeInteger(Number(token))) {
      throw new Error('Qubit budget: "' + token + '" is not a whole number, 0 or more.');
    }
    return Number(token);
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

  const api = {
    makeQUBO, cost, density, exactSolver, bruteForce, maxcutQUBO, isingTerms, recommend, fixed,
    parseEdgeList, parseMatrix, parseOffset, parseBudget,
    EXACT_CLASSICAL_LIMIT, EXACT_SOLVER_MAX_N, TIE_TOLERANCE, SYMMETRY_TOLERANCE, ZERO_TOLERANCE,
  };
  global.QOPT = Object.assign(global.QOPT || {}, api);
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
