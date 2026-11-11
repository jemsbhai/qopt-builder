/* Runs the web port's own source files under Node.js and prints, as
 * JSON, what they compute for the problems read from stdin.
 *
 * Used by tests/test_web_parity.py, which computes the same quantities
 * with the Python package and compares. Nothing here is loaded by the
 * site: web/ stays a static, dependency-free set of files. */
"use strict";

const path = require("path");

const web = path.join(__dirname, "..", "web");
const Q = Object.assign(
  {},
  require(path.join(web, "qubo.js")),
  require(path.join(web, "qaoa.js")),
  require(path.join(web, "circuit.js")),
  require(path.join(web, "presets.js"))
);

function runCase(c) {
  const problem = Q.makeQUBO(c.matrix, c.offset, c.name);
  const n = problem.n;
  const rec = Q.recommend(problem, c.budget === undefined ? null : c.budget);
  const ising = Q.isingTerms(problem);
  const out = {
    name: c.name,
    n,
    density: Q.density(problem),
    recommendation: rec,
    ising,
    exact: c.solve ? Q.bruteForce(problem) : null,
    circuits: {},
    expectations: {},
    scan: null,
  };
  for (const p of c.depths || []) {
    const s = Q.circuitStats(problem, ising, p);
    out.circuits[p] = { qubits: s.qubits, depth: s.depth, total: s.total, counts: s.counts, params: s.params };
  }
  for (const [p, sets] of Object.entries(c.angles || {})) {
    out.expectations[p] = sets.map(([gammas, betas]) => Q.expectation(ising, n, Number(p), gammas, betas));
  }
  if (c.scan) {
    const p = rec.params.p;
    const scan = Q.scanAngles(ising, n, p, c.scan.samples, c.scan.seed);
    const refined = Q.refineAngles(ising, n, p, scan.best, scan.costs);
    out.scan = { p, best: scan.best, trace: Array.from(scan.trace), refined };
    out.scan.success = Q.successProbability(ising, n, p, refined.gammas, refined.betas, out.exact.value);
    out.stepped = stepped(problem, ising, p, c.scan);
  }
  return out;
}

/* The same work done through the resumable objects the guided demo
 * uses, in uneven slices. It has to land on the same numbers. */
function stepped(problem, ising, p, options) {
  const n = problem.n;
  const solver = Q.exactSolver(problem);
  let lo = Infinity, hi = -Infinity, slices = 0;
  while (!solver.done) {
    const part = solver.step(7);
    if (part.lo < lo) lo = part.lo;
    if (part.hi > hi) hi = part.hi;
    slices++;
  }
  const scan = Q.angleScan(ising, n, p, options.samples, options.seed);
  while (!scan.done) scan.step(13);
  const scanned = scan.result();
  const refiner = Q.angleRefiner(ising, n, p, scanned.best, scanned.costs);
  const history = [];
  while (!refiner.done) { refiner.step(); history.push(refiner.value); }
  return {
    exact: solver.result(), visited: solver.visited, total: solver.total, slices, lo, hi,
    best: scanned.best, trace: Array.from(scanned.trace),
    refined: refiner.result(), iterations: refiner.iteration, history,
  };
}

let input = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => { input += chunk; });
process.stdin.on("end", () => {
  const { cases } = JSON.parse(input);
  process.stdout.write(JSON.stringify({
    constants: {
      EXACT_CLASSICAL_LIMIT: Q.EXACT_CLASSICAL_LIMIT,
      TIE_TOLERANCE: Q.TIE_TOLERANCE,
      SIMULATION_LIMIT: Q.SIMULATION_LIMIT,
    },
    // the page's preset list, each built into the matrix the page would solve
    presets: Q.PRESETS.map((preset) => {
      const problem = Q.buildPreset(preset);
      return {
        name: preset.name, title: preset.title, kind: preset.kind, n: preset.n, about: preset.about,
        matrix: problem.matrix, offset: problem.offset,
      };
    }),
    cases: cases.map(runCase),
  }));
});
