/* Hardware package: one problem and its circuit, ready for a quantum
 * computer. A port of qopt_builder/package.py: the same five files,
 * written by the same rules, so that one problem with one set of angles
 * gives the same bytes here and from the command line.
 * tests/test_web_parity.py holds the two ports to that.
 *
 * Nothing here is left to a routine whose output could differ between
 * the ports: numbers are written by numberText(), JSON line by line,
 * and the zip archive byte by byte.
 *
 * Load after qubo.js, circuit.js and package-text.js. */
(function (global) {
  "use strict";

  const VERSION = "0.1.0"; // qopt_builder.__version__
  const PACKAGE_FORMAT = "qopt-builder hardware package, version 1";
  const PACKAGE_NAME = "qopt-package";
  const PACKAGE_FILES = ["README.txt", "problem.json", "circuit.qasm", "run_on_ibm.py", "requirements.txt"];
  const GATES = ["h", "rz", "rzz", "rx"]; // the order the gates appear in the circuit

  const isFiniteNumber = (v) => typeof v === "number" && Number.isFinite(v);

  /* A finite number as text that reads back as the same number. The
   * digits are the shortest that identify the number, written with a
   * decimal point: 8.0, -0.25, 1234.5. A number with a decimal exponent
   * below -6 or above 15 is written as 1.5e-7 or 1.0e+21. Zero is 0.0
   * whatever its sign. The text is valid in JSON and in OpenQASM 2. */
  function numberText(value) {
    if (!isFiniteNumber(value)) throw new Error("A number written into a package must be finite.");
    if (value === 0) return "0.0";
    const sign = value < 0 ? "-" : "";
    const parts = Math.abs(value).toExponential().split("e");
    const digits = parts[0].replace(".", "");
    const power = Number(parts[1]); // the number is d1.d2d3... x 10^power
    const point = power + 1;
    if (power < -6 || power > 15) {
      return sign + digits[0] + "." + (digits.slice(1) || "0") + "e" + (power < 0 ? "-" : "+") + Math.abs(power);
    }
    if (point <= 0) return sign + "0." + "0".repeat(-point) + digits;
    if (point >= digits.length) return sign + digits + "0".repeat(point - digits.length) + ".0";
    return sign + digits.slice(0, point) + "." + digits.slice(point);
  }

  /* A string as JSON text, in ASCII: the text Python's json.dumps gives. */
  function jsonText(text) {
    return JSON.stringify(text).replace(/[\u0080-￿]/g,
      (c) => "\\u" + c.charCodeAt(0).toString(16).padStart(4, "0"));
  }

  /* "1 edge", "8 edges". */
  function count(number, thing) {
    return number + " " + thing + (number === 1 ? "" : "s");
  }

  /* One line naming the problem: its kind and its size. */
  function describeProblem(kind, n, edges) {
    if (kind === "maxcut") return "MaxCut on " + count(n, "node") + ", " + count(edges, "edge");
    return "QUBO on " + count(n, "variable");
  }

  /* cost(x) summed term by term in a fixed order, so that both ports
   * arrive at the same number down to the last bit. */
  function exactCost(problem, x) {
    let total = problem.offset;
    for (let i = 0; i < problem.n; i++) {
      if (x[i]) {
        for (let j = 0; j < problem.n; j++) {
          if (x[j]) total += problem.matrix[i][j];
        }
      }
    }
    return total;
  }

  /* The QAOA circuit as OpenQASM 2 text, with the angles filled in and
   * every qubit measured. The gates and their order are those of
   * circuit.js and of qopt_builder/build.py. The file defines rzz
   * itself, because the standard gate file of OpenQASM 2 does not have
   * it. */
  function circuitQasm(problem, layers, gammas, betas, description) {
    const n = problem.n, ising = global.QOPT.isingTerms(problem), h = ising.h, J = ising.J;
    const lines = [
      "// QAOA circuit made by qopt-builder " + VERSION,
      "// Problem: " + description,
      "// " + count(layers, "layer") + ", angles filled in; problem.json lists them",
      "// Qubit q[i] stands for variable x_i and is measured into bit c[i]",
      "OPENQASM 2.0;",
      'include "qelib1.inc";',
      "gate rzz(theta) a,b { cx a,b; rz(theta) b; cx a,b; }",
      "qreg q[" + n + "];",
      "creg c[" + n + "];",
    ];
    for (let i = 0; i < n; i++) lines.push("h q[" + i + "];");
    for (let layer = 0; layer < layers; layer++) {
      const g = gammas[layer], b = betas[layer];
      lines.push("// layer " + (layer + 1) + " of " + layers + ": cost");
      for (let i = 0; i < n; i++) {
        if (Math.abs(h[i]) > 1e-12) lines.push("rz(" + numberText(2 * g * h[i]) + ") q[" + i + "];");
        for (let j = i + 1; j < n; j++) {
          if (Math.abs(J[i][j]) > 1e-12) {
            lines.push("rzz(" + numberText(2 * g * J[i][j]) + ") q[" + i + "],q[" + j + "];");
          }
        }
      }
      lines.push("// layer " + (layer + 1) + " of " + layers + ": mixer");
      for (let i = 0; i < n; i++) lines.push("rx(" + numberText(2 * b) + ") q[" + i + "];");
    }
    for (let i = 0; i < n; i++) lines.push("measure q[" + i + "] -> c[" + i + "];");
    return lines.join("\n") + "\n";
  }

  function checkedAngles(values, layers, name) {
    if (!Array.isArray(values) || values.length !== layers || !values.every(isFiniteNumber)) {
      throw new Error(name + " must be a list of " + count(layers, "number") + ", one per layer.");
    }
    return values.slice();
  }

  /* The five files of a hardware package, as a list of { name, text }.
   *
   *   problem         the QUBO, from makeQUBO() or maxcutQUBO()
   *   kind            "maxcut" or "qubo"; with "maxcut", `edges` is the
   *                   number of edges, for the description
   *   recommendation  what recommend() returned for the problem
   *   optimum         the assignment bruteForce() returned, or null when
   *                   the problem was not solved exactly
   *   layers          the number of QAOA layers built
   *   gammas, betas   the angles, one of each per layer
   *   anglesFrom      one line saying where the angles came from
   *
   * Throws for a refused problem (there is no circuit to package) and
   * for angles that do not fit the layers. */
  function packageFiles(input) {
    const Q = global.QOPT;
    const { problem, kind, edges, recommendation, optimum, layers, anglesFrom } = input;
    if (recommendation.algorithm === "none") {
      throw new Error("The builder refused this problem, so there is no circuit to package.");
    }
    if (!Number.isInteger(layers) || layers < 1) throw new Error("QAOA depth p must be a positive whole number.");
    const gammas = checkedAngles(input.gammas, layers, "gammas");
    const betas = checkedAngles(input.betas, layers, "betas");
    if (kind !== "maxcut" && kind !== "qubo") throw new Error('kind must be "maxcut" or "qubo".');
    const n = problem.n;
    const description = describeProblem(kind, n, edges);
    const stats = Q.circuitStats(problem, Q.isingTerms(problem), layers);

    const out = ["{"];
    out.push('  "format": ' + jsonText(PACKAGE_FORMAT) + ",");
    out.push('  "made_by": ' + jsonText("qopt-builder " + VERSION) + ",");
    out.push('  "problem": {');
    out.push('    "kind": ' + jsonText(kind) + ",");
    out.push('    "description": ' + jsonText(description) + ",");
    out.push('    "variables": ' + n + ",");
    out.push('    "offset": ' + numberText(problem.offset) + ",");
    out.push('    "matrix": [');
    out.push(problem.matrix.map((row) => "      [" + row.map(numberText).join(", ") + "]").join(",\n"));
    out.push("    ]");
    out.push("  },");
    if (optimum === null || optimum === undefined) {
      out.push('  "certified_optimum": null,');
    } else {
      out.push('  "certified_optimum": {');
      out.push('    "cost": ' + numberText(exactCost(problem, optimum)) + ",");
      out.push('    "assignment": [' + Array.from(optimum).join(", ") + "]");
      out.push("  },");
    }
    out.push('  "recommendation": {');
    out.push('    "algorithm": ' + jsonText(recommendation.algorithm) + ",");
    out.push('    "layers": ' + recommendation.params.p + ",");
    out.push('    "reasons": [');
    out.push(recommendation.rationale.map((reason) => "      " + jsonText(reason)).join(",\n"));
    out.push("    ]");
    out.push("  },");
    out.push('  "qaoa": {');
    out.push('    "layers": ' + layers + ",");
    out.push('    "gammas": [' + gammas.map(numberText).join(", ") + "],");
    out.push('    "betas": [' + betas.map(numberText).join(", ") + "],");
    out.push('    "angles_from": ' + jsonText(anglesFrom));
    out.push("  },");
    out.push('  "circuit": {');
    out.push('    "file": "circuit.qasm",');
    out.push('    "qubits": ' + stats.qubits + ",");
    out.push('    "depth": ' + stats.depth + ",");
    out.push('    "gates": ' + stats.total + ",");
    for (const gate of GATES) out.push('    "' + gate + '": ' + (stats.counts[gate] || 0) + ",");
    out.push('    "measurements": ' + n);
    out.push("  }");
    out.push("}");

    const values = {
      description,
      layers: count(layers, "layer"),
      qubits: count(n, "qubit"),
      angles_from: anglesFrom,
      version: VERSION,
    };
    const text = Q.packageText;
    return [
      { name: "README.txt", text: text.README_TEMPLATE.replace(/\{(\w+)\}/g, (_, key) => values[key]) },
      { name: "problem.json", text: out.join("\n") + "\n" },
      { name: "circuit.qasm", text: circuitQasm(problem, layers, gammas, betas, description) },
      { name: "run_on_ibm.py", text: text.RUN_SCRIPT },
      { name: "requirements.txt", text: text.REQUIREMENTS },
    ];
  }

  /* ------------------------------------------------------- zip archive */

  const CRC_TABLE = (function () {
    const table = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      table[n] = c >>> 0;
    }
    return table;
  })();

  function crc32(bytes) {
    let c = 0xffffffff;
    for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
  }

  /* Numbers as little-endian bytes: each field is [value, size in bytes]. */
  function fields(list) {
    const out = [];
    for (const [value, size] of list) {
      for (let k = 0; k < size; k++) out.push(Math.floor(value / Math.pow(256, k)) % 256);
    }
    return out;
  }

  /* The files as a zip archive that unpacks to one folder, as a
   * Uint8Array. The files are stored without compression. `when` is the
   * date and time stamped on them, as [year, month, day, hour, minute,
   * second]; leaving it out means now, in local time. The same files,
   * folder name and time always give the same bytes. */
  function zipBytes(files, folder, when) {
    const top = folder === undefined || folder === null ? PACKAGE_NAME : folder;
    let stamp = when;
    if (stamp === undefined || stamp === null) {
      const now = new Date();
      stamp = [now.getFullYear(), now.getMonth() + 1, now.getDate(), now.getHours(), now.getMinutes(), now.getSeconds()];
    }
    const [year, month, day, hour, minute, second] = stamp;
    if (year < 1980 || year > 2107) throw new Error("A zip archive can only hold dates from 1980 to 2107.");
    const dosTime = (hour << 11) | (minute << 5) | (second >> 1);
    const dosDate = ((year - 1980) << 9) | (month << 5) | day;
    const encoder = new TextEncoder();

    const body = [], directory = [];
    for (const file of files) {
      const path = Array.from(encoder.encode(top + "/" + file.name));
      const data = encoder.encode(file.text);
      const flags = path.some((byte) => byte > 127) ? 0x800 : 0; // bit 11: the name is UTF-8
      // the fields a file's own header and its entry in the directory share
      const common = fields([
        [20, 2], [flags, 2], [0, 2], [dosTime, 2], [dosDate, 2], [crc32(data), 4],
        [data.length, 4], [data.length, 4], [path.length, 2], [0, 2],
      ]);
      directory.push(0x50, 0x4b, 0x01, 0x02, ...fields([[20, 2]]), ...common,
        ...fields([[0, 2], [0, 2], [0, 2], [0, 4], [body.length, 4]]), ...path);
      body.push(0x50, 0x4b, 0x03, 0x04, ...common, ...path);
      for (let i = 0; i < data.length; i++) body.push(data[i]);
    }
    const end = [0x50, 0x4b, 0x05, 0x06, ...fields([
      [0, 2], [0, 2], [files.length, 2], [files.length, 2], [directory.length, 4], [body.length, 4], [0, 2],
    ])];
    return Uint8Array.from(body.concat(directory, end));
  }

  const api = {
    numberText, describeProblem, exactCost, circuitQasm, packageFiles, zipBytes, crc32,
    VERSION, PACKAGE_FORMAT, PACKAGE_NAME, PACKAGE_FILES,
  };
  global.QOPT = Object.assign(global.QOPT || {}, api);
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
