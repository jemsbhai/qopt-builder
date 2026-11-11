/* QAOA statevector simulation in the browser.
 * Gate conventions match Qiskit: basis index bit i is qubit i;
 * z_i = +1 when bit i is 0. RZ(t) = diag(e^-it/2, e^+it/2);
 * RZZ(t) applies e^-it/2 when the two bits agree, e^+it/2 otherwise;
 * RX(t) = cos(t/2) I - i sin(t/2) X. The circuit matches
 * qopt_builder/build.py: H on all qubits, then per layer the phase
 * separator RZ(2*gamma*h_i) and RZZ(2*gamma*J_ij) followed by the
 * mixer RX(2*beta) on every qubit. */
(function (global) {
  "use strict";

  function newState(n) {
    const size = 1 << n;
    return { re: new Float64Array(size), im: new Float64Array(size), n };
  }

  function applyH(st, q) {
    const step = 1 << q, s = Math.SQRT1_2;
    const { re, im } = st;
    for (let base = 0; base < re.length; base += step * 2) {
      for (let i = base; i < base + step; i++) {
        const a = i, b = i + step;
        const ar = re[a], ai = im[a], br = re[b], bi = im[b];
        re[a] = (ar + br) * s; im[a] = (ai + bi) * s;
        re[b] = (ar - br) * s; im[b] = (ai - bi) * s;
      }
    }
  }

  function applyRZ(st, q, theta) {
    const step = 1 << q;
    const c = Math.cos(theta / 2), s = Math.sin(theta / 2);
    const { re, im } = st;
    for (let idx = 0; idx < re.length; idx++) {
      const bit = (idx >> q) & 1;
      const sign = bit ? 1 : -1; // e^{-i t/2} on |0>, e^{+i t/2} on |1>
      const cr = c, ci = sign * s;
      const r = re[idx], i2 = im[idx];
      re[idx] = r * cr - i2 * ci;
      im[idx] = r * ci + i2 * cr;
    }
  }

  function applyRZZ(st, a, b, theta) {
    const c = Math.cos(theta / 2), s = Math.sin(theta / 2);
    const { re, im } = st;
    for (let idx = 0; idx < re.length; idx++) {
      const agree = (((idx >> a) & 1) === ((idx >> b) & 1));
      const sign = agree ? -1 : 1;
      const cr = c, ci = sign * s;
      const r = re[idx], i2 = im[idx];
      re[idx] = r * cr - i2 * ci;
      im[idx] = r * ci + i2 * cr;
    }
  }

  function applyRX(st, q, theta) {
    const step = 1 << q;
    const c = Math.cos(theta / 2), s = Math.sin(theta / 2);
    const { re, im } = st;
    for (let base = 0; base < re.length; base += step * 2) {
      for (let i = base; i < base + step; i++) {
        const a = i, b = i + step;
        const ar = re[a], ai = im[a], br = re[b], bi = im[b];
        // RX = [[c, -i s], [-i s, c]]
        re[a] = c * ar + s * bi; im[a] = c * ai - s * br;
        re[b] = s * ai + c * br; im[b] = -s * ar + c * bi;
      }
    }
  }

  function runQAOA(ising, n, p, gammas, betas) {
    const st = newState(n);
    st.re[0] = 1;
    for (let q = 0; q < n; q++) applyH(st, q);
    const { h, J } = ising;
    for (let layer = 0; layer < p; layer++) {
      const g = gammas[layer], b = betas[layer];
      for (let i = 0; i < n; i++) {
        if (Math.abs(h[i]) > 1e-12) applyRZ(st, i, 2 * g * h[i]);
        for (let j = i + 1; j < n; j++) {
          if (Math.abs(J[i][j]) > 1e-12) applyRZZ(st, i, j, 2 * g * J[i][j]);
        }
      }
      for (let q = 0; q < n; q++) applyRX(st, q, 2 * b);
    }
    return st;
  }

  function basisCosts(ising, n) {
    const size = 1 << n, { h, J, offset } = ising;
    const out = new Float64Array(size);
    for (let idx = 0; idx < size; idx++) {
      let e = offset;
      for (let i = 0; i < n; i++) {
        const zi = ((idx >> i) & 1) === 0 ? 1 : -1;
        e += h[i] * zi;
        for (let j = i + 1; j < n; j++) {
          if (J[i][j] !== 0) {
            const zj = ((idx >> j) & 1) === 0 ? 1 : -1;
            e += J[i][j] * zi * zj;
          }
        }
      }
      out[idx] = e;
    }
    return out;
  }

  function expectation(ising, n, p, gammas, betas, costs) {
    const st = runQAOA(ising, n, p, gammas, betas);
    const cs = costs || basisCosts(ising, n);
    let e = 0;
    for (let idx = 0; idx < cs.length; idx++) {
      const pr = st.re[idx] * st.re[idx] + st.im[idx] * st.im[idx];
      if (pr > 1e-15) e += pr * cs[idx];
    }
    return e;
  }

  function topBitstrings(ising, n, p, gammas, betas, k) {
    const st = runQAOA(ising, n, p, gammas, betas);
    const cs = basisCosts(ising, n);
    const rows = [];
    for (let idx = 0; idx < cs.length; idx++) {
      const pr = st.re[idx] * st.re[idx] + st.im[idx] * st.im[idx];
      const bits = [];
      for (let i = 0; i < n; i++) bits.push((idx >> i) & 1);
      rows.push({ bits, prob: pr, cost: cs[idx] });
    }
    rows.sort((a, b2) => b2.prob - a.prob);
    return rows.slice(0, k || 8);
  }

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* Probability of measuring an optimal assignment: the total weight
   * the state at these angles puts on assignments whose cost is within
   * `tolerance` of `optimum`. */
  function successProbability(ising, n, p, gammas, betas, optimum, tolerance) {
    const st = runQAOA(ising, n, p, gammas, betas);
    const cs = basisCosts(ising, n);
    const tol = tolerance === undefined ? 1e-9 : tolerance;
    let total = 0;
    for (let idx = 0; idx < cs.length; idx++) {
      if (cs[idx] <= optimum + tol) total += st.re[idx] * st.re[idx] + st.im[idx] * st.im[idx];
    }
    return total;
  }

  /* Coarse random scan (200 uniform samples by default), as a resumable
   * object so a page can run it a few samples at a time and draw the
   * trace as it grows. step(count) scores up to `count` more samples.
   * Each sample draws p gammas and then p betas, uniformly in [0, pi). */
  function angleScan(ising, n, p, samples, seed) {
    const rng = mulberry32(seed || 20261110);
    const costs = basisCosts(ising, n);
    const total = samples || 200;
    let best = { value: Infinity, gammas: null, betas: null };
    const trace = [];
    return {
      total,
      step(count) {
        const end = Math.min(total, trace.length + count);
        while (trace.length < end) {
          const gammas = Array.from({ length: p }, () => rng() * Math.PI);
          const betas = Array.from({ length: p }, () => rng() * Math.PI);
          const v = expectation(ising, n, p, gammas, betas, costs);
          trace.push(v);
          if (v < best.value) best = { value: v, gammas, betas };
        }
      },
      result() { return { best, trace, costs }; },
      get done() { return trace.length >= total; },
      get best() { return best; },
      get trace() { return trace; },
    };
  }

  /* The whole scan in one call. */
  function scanAngles(ising, n, p, samples, seed) {
    const scan = angleScan(ising, n, p, samples, seed);
    scan.step(Infinity);
    return scan.result();
  }

  /* Nelder-Mead refinement of the scanned angles (2p parameters), as a
   * resumable object: step() performs one iteration, so a page can draw
   * the value as it converges. It stops after 300 iterations, or when
   * the simplex has collapsed to within 1e-10 in value. */
  function angleRefiner(ising, n, p, start, costs) {
    const cs = costs || basisCosts(ising, n);
    const x0 = [...start.gammas, ...start.betas];
    const d = x0.length;
    const f = (x) => expectation(ising, n, p, x.slice(0, p), x.slice(p), cs);
    let simplex = [x0.slice()];
    for (let i = 0; i < d; i++) {
      const pt = x0.slice();
      pt[i] += 0.15;
      simplex.push(pt);
    }
    let vals = simplex.map(f);
    let iter = 0, done = false;

    function step() {
      if (done) return;
      if (iter >= 300) { done = true; return; }
      const order = vals.map((v, i) => i).sort((a, b) => vals[a] - vals[b]);
      simplex = order.map((i) => simplex[i]);
      vals = order.map((i) => vals[i]);
      const centroid = new Array(d).fill(0);
      for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) centroid[j] += simplex[i][j] / d;
      const worst = simplex[d];
      const refl = centroid.map((c, j) => c + (c - worst[j]));
      const fr = f(refl);
      if (fr < vals[0]) {
        const exp = centroid.map((c, j) => c + 2 * (c - worst[j]));
        const fe = f(exp);
        if (fe < fr) { simplex[d] = exp; vals[d] = fe; } else { simplex[d] = refl; vals[d] = fr; }
      } else if (fr < vals[d - 1]) {
        simplex[d] = refl; vals[d] = fr;
      } else {
        const con = centroid.map((c, j) => c + 0.5 * (worst[j] - c));
        const fc = f(con);
        if (fc < vals[d]) { simplex[d] = con; vals[d] = fc; }
        else {
          for (let i = 1; i <= d; i++) {
            simplex[i] = simplex[i].map((v, j) => simplex[0][j] + 0.5 * (v - simplex[0][j]));
            vals[i] = f(simplex[i]);
          }
        }
      }
      iter++;
      if (Math.abs(vals[d] - vals[0]) < 1e-10 || iter >= 300) done = true;
    }

    function result() {
      const bi = vals.indexOf(Math.min(...vals));
      const bx = simplex[bi];
      return { value: vals[bi], gammas: bx.slice(0, p), betas: bx.slice(p) };
    }

    return {
      step,
      result,
      get done() { return done; },
      get iteration() { return iter; },
      get value() { return Math.min(...vals); }, // best value in the simplex so far
    };
  }

  /* The whole refinement in one call. */
  function refineAngles(ising, n, p, start, costs) {
    const refiner = angleRefiner(ising, n, p, start, costs);
    while (!refiner.done) refiner.step();
    return refiner.result();
  }

  /* Largest n simulated: 2^14 amplitudes per sample. Same value as
   * SIMULATION_LIMIT in qopt_builder/simulate.py. */
  const SIMULATION_LIMIT = 14;

  const api = {
    runQAOA, basisCosts, expectation, topBitstrings, successProbability,
    angleScan, scanAngles, angleRefiner, refineAngles, SIMULATION_LIMIT,
  };
  global.QOPT = Object.assign(global.QOPT || {}, api);
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
