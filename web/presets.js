/* Named preset problems for the browser demo: the same list, with the
 * same names, numbers and wording, as qopt_builder/presets.py. The
 * parity test (tests/test_web_parity.py) loads this file under Node and
 * checks that both sides describe the same problems. Load after qubo.js. */
(function (global) {
  "use strict";

  /* The 6-node weighted graph of the demo record. */
  const DEMO_EDGES = [
    [0, 1, 1], [0, 2, 1], [1, 2, 1], [1, 3, 2],
    [2, 4, 1], [3, 4, 1], [3, 5, 1], [4, 5, 2],
  ];

  /* Edges joining every node i to node i + step, around a ring of n nodes. */
  function ring(n, step) {
    return Array.from({ length: n }, (_, i) => {
      const j = (i + step) % n;
      return [Math.min(i, j), Math.max(i, j), 1];
    });
  }

  const PRESETS = [
    {
      name: "demo",
      title: "Demo graph: weighted MaxCut, 6 nodes",
      kind: "maxcut",
      n: 6,
      about: "Two triangles joined by two edges, with weights 1 and 2. A triangle always " +
        "leaves one of its edges uncut, so the best cut misses one weight-1 edge in each.",
      edges: DEMO_EDGES,
    },
    {
      name: "tree-14",
      title: "Sparse tree: MaxCut, 14 nodes",
      kind: "maxcut",
      n: 14,
      about: "A binary tree with unit weights. A tree can be two-coloured level by level, so " +
        "every edge can be cut. This is the largest size the browser simulates.",
      edges: Array.from({ length: 13 }, (_, k) => [Math.floor(k / 2), k + 1, 1]),
    },
    {
      name: "ring-10",
      title: "Ring: MaxCut, 10 nodes",
      kind: "maxcut",
      n: 10,
      about: "A ring of ten nodes with unit weights. An even ring alternates sides and cuts every edge.",
      edges: ring(10, 1),
    },
    {
      name: "weighted-8",
      title: "Dense weighted graph: MaxCut, 8 nodes",
      kind: "maxcut",
      n: 8,
      about: "Sixteen weighted edges on eight nodes, more than half of all possible pairs.",
      edges: [
        [0, 1, 2], [0, 3, 1], [1, 2, 3], [1, 4, 1], [2, 5, 2], [3, 4, 2],
        [3, 6, 1], [4, 5, 1], [4, 7, 3], [5, 7, 2], [6, 7, 1], [2, 3, 1],
        [5, 6, 2], [0, 5, 1], [1, 6, 1], [3, 7, 2],
      ],
    },
    {
      name: "chorded-ring-24",
      title: "Chorded ring: MaxCut, 24 nodes",
      kind: "maxcut",
      n: 24,
      about: "A problem at the exact-solve limit of 24 variables. Every node is joined to its " +
        "neighbours one and two steps around a ring: 48 unit edges, full of triangles. " +
        "Each triangle leaves an edge uncut, which bounds the cut at 36, and the pattern " +
        "1100 repeated reaches it. Too large to simulate, so the builder certifies the " +
        "optimum and builds the circuit only.",
      edges: ring(24, 1).concat(ring(24, 2)),
    },
    {
      name: "vertex-cover-6",
      title: "Vertex cover: QUBO, 6 variables",
      kind: "qubo",
      n: 6,
      about: "Minimum vertex cover, a different problem on the demo graph: pick the fewest " +
        "nodes that touch every edge. Each chosen node costs 1 and each uncovered edge " +
        "costs 2, which puts linear terms on the diagonal and a constant offset of 16. " +
        "The optimum cost is the size of the smallest cover.",
      /* cost(x) = sum_i x_i + 2 * sum_{(i,j)} (1 - x_i)(1 - x_j), expanded:
       * 1 - 2 * degree(i) on the diagonal, 2 per edge above it, offset 2 * 8. */
      matrix: [
        [-3, 2, 2, 0, 0, 0],
        [0, -5, 2, 2, 0, 0],
        [0, 0, -5, 0, 2, 0],
        [0, 0, 0, -5, 2, 2],
        [0, 0, 0, 0, -5, 2],
        [0, 0, 0, 0, 0, -3],
      ],
      offset: 16,
    },
    {
      name: "dense-qubo-6",
      title: "Dense symmetric QUBO, 6 variables",
      kind: "qubo",
      n: 6,
      about: "A matrix entered in symmetric form, with every pair of variables coupled and " +
        "linear terms on the diagonal.",
      matrix: [
        [-2, 1, 2, 0.5, 1, 2],
        [1, -3, 1, 2, 0.5, 1],
        [2, 1, -2, 1, 2, 0.5],
        [0.5, 2, 1, -3, 1, 2],
        [1, 0.5, 2, 1, -2, 1],
        [2, 1, 0.5, 2, 1, -3],
      ],
      offset: 0,
    },
  ];

  /* A preset as a QUBO object, ready for recommend() and bruteForce(). */
  function buildPreset(preset) {
    const Q = global.QOPT;
    const problem = preset.kind === "maxcut"
      ? Q.maxcutQUBO(preset.edges, preset.n)
      : Q.makeQUBO(preset.matrix, preset.offset, preset.name);
    problem.name = preset.name;
    return problem;
  }

  const api = { PRESETS, buildPreset };
  global.QOPT = Object.assign(global.QOPT || {}, api);
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
