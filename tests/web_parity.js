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
  require(path.join(web, "presets.js")),
  require(path.join(web, "package-text.js")),
  require(path.join(web, "package.js"))
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
    if (c.scan.package) {
      // the package the page offers after this run: the refined angles, as app.js passes them
      out.scan.package = packed(Object.assign({}, c.scan.package, {
        problem, recommendation: rec, optimum: out.exact.x, layers: p, gammas: refined.gammas, betas: refined.betas,
      }));
    }
    out.stepped = stepped(problem, ising, p, c.scan);
    // a seed of 0 is a seed like any other, not a request for the default one
    out.seedZero = Array.from(Q.scanAngles(ising, n, 1, 3, 0).trace);
  }
  return out;
}

/* One input given to the input checks (tests/validation_cases.py): what
 * came back, or the message of the error. */
function runCheck(c) {
  try {
    return { ok: call(c) };
  } catch (error) {
    return { error: error.message };
  }
}

function call(c) {
  if (c.call === "qubo") {
    const built = "offset" in c ? Q.makeQUBO(c.matrix, c.offset) : Q.makeQUBO(c.matrix);
    return { matrix: built.matrix, offset: built.offset };
  }
  if (c.call === "maxcut") {
    const built = Q.maxcutQUBO(c.edges, c.n);
    return { matrix: built.matrix, offset: built.offset };
  }
  const problem = Q.makeQUBO(c.matrix);
  if (c.call === "cost") return Q.cost(problem, c.x);
  const ising = Q.isingTerms(problem);
  if (c.call === "circuit") {
    const stats = Q.circuitStats(problem, ising, c.p);
    return { depth: stats.depth, gates: stats.total };
  }
  if (c.call === "scan") return Q.scanAngles(ising, problem.n, c.p, c.samples, 1).trace.length;
  throw new Error("unknown call " + c.call);
}

/* A hardware package: its five files, and the archive that holds them
 * as base64. */
function packed(spec) {
  const files = Q.packageFiles(spec);
  return { files, zip: Buffer.from(Q.zipBytes(files, spec.folder, spec.when)).toString("base64") };
}

function runPackage(c) {
  try {
    const problem = Q.makeQUBO(c.matrix, c.offset);
    const recommendation = Q.recommend(problem, c.budget === undefined ? null : c.budget);
    const optimum = c.solve ? Q.bruteForce(problem).x : null;
    return packed(Object.assign({}, c, { problem, recommendation, optimum }));
  } catch (error) {
    return { error: error.message };
  }
}

/* Text as it would be typed into the page, through the page's parsers. */
function runText(t) {
  try {
    return { ok: readText(t) };
  } catch (error) {
    return { error: error.message };
  }
}

function readText(t) {
  if (t.kind === "edges") return Q.parseEdgeList(t.text, t.limit);
  if (t.kind === "matrix") return Q.parseMatrix(t.text, t.limit);
  if (t.kind === "offset") return Q.parseOffset(t.text);
  if (t.kind === "budget") return Q.parseBudget(t.text);
  if (t.kind === "edges-problem") {
    const parsed = Q.parseEdgeList(t.text, t.limit);
    const built = Q.maxcutQUBO(parsed.edges, parsed.n);
    return { matrix: built.matrix, offset: built.offset };
  }
  if (t.kind === "matrix-problem") {
    const built = Q.makeQUBO(Q.parseMatrix(t.text, t.limit), Q.parseOffset(t.offset));
    return { matrix: built.matrix, offset: built.offset };
  }
  if (t.kind === "solver") {
    const zero = Array.from({ length: t.n }, () => new Array(t.n).fill(0));
    return Q.exactSolver(Q.makeQUBO(zero)).total;
  }
  throw new Error("unknown kind " + t.kind);
}

/* The circuit shown in the primer (web/learn.html): a triangle at one
 * layer, drawn by the page's own drawing code. */
function triangleFigure() {
  const problem = Q.maxcutQUBO([[0, 1], [1, 2], [0, 2]], 3);
  const stats = Q.circuitStats(problem, Q.isingTerms(problem), 1);
  return { svg: Q.renderCircuitSVG(stats.ops, 3, Q.schedule(stats.ops, 3)), gates: stats.total, depth: stats.depth };
}

/* JSON cannot write NaN or the infinities, so the test spells them out
 * (to_json in tests/validation_cases.py) and this turns them back. */
function revive(key, value) {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const keys = Object.keys(value);
    if (keys.length === 1 && keys[0] === "$") return { nan: NaN, inf: Infinity, "-inf": -Infinity }[value.$];
  }
  return value;
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
  const { cases, checks, texts, packages, numbers } = JSON.parse(input, revive);
  process.stdout.write(JSON.stringify({
    constants: {
      EXACT_CLASSICAL_LIMIT: Q.EXACT_CLASSICAL_LIMIT,
      TIE_TOLERANCE: Q.TIE_TOLERANCE,
      SIMULATION_LIMIT: Q.SIMULATION_LIMIT,
      SYMMETRY_TOLERANCE: Q.SYMMETRY_TOLERANCE,
      ZERO_TOLERANCE: Q.ZERO_TOLERANCE,
      EXACT_SOLVER_MAX_N: Q.EXACT_SOLVER_MAX_N,
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
    checks: Object.fromEntries(checks.map((c) => [c.name, runCheck(c)])),
    figures: { triangle: triangleFigure() },
    texts: texts.map(runText),
    packageConstants: {
      VERSION: Q.VERSION, PACKAGE_FORMAT: Q.PACKAGE_FORMAT, PACKAGE_NAME: Q.PACKAGE_NAME, PACKAGE_FILES: Q.PACKAGE_FILES,
    },
    packages: Object.fromEntries(packages.map((c) => [c.name, runPackage(c)])),
    numbers: numbers.map((x) => {
      try { return Q.numberText(x); } catch (error) { return { error: error.message }; }
    }),
  }));
});
