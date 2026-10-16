"""Run the traffic QAOA pipeline for a given scenario.

Example: python run.py --scenario <scenario.json> --seed <int> --out <dir> [--p 4]

Builds the traffic QUBO, trains QAOA with at most 300 objective evaluations,
runs the circuit for exactly 4096 shots, and saves qubo.json, circuit.qasm,
counts.json, schedule.json, and metrics.json.
"""
import argparse
import json
import time
from pathlib import Path

import traffic_qaoa as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--p", type=int, default=4, choices=[1, 2, 3, 4])
    ap.add_argument("--no-diagnostics", action="store_true",
                    help="skip benchmark enumeration (C*, lift) since it's not needed for the outputs")
    a = ap.parse_args()
    t0 = time.time()
    sc = T.load_scenario(a.scenario)
    r = T.run_pipeline(sc, seed=a.seed, p=a.p, shots=4096, out=a.out,
                       with_diagnostics=not a.no_diagnostics)
    summary = {"scenario": sc.get("name", Path(a.scenario).stem), "seed": a.seed, "p": a.p,
               "qubits": T.N_BITS, "training_evaluations": len(r["history"]),
               "circuit_depth": r["circuit"].depth(), "ops": dict(r["circuit"].count_ops()),
               "runtime_s": round(time.time() - t0, 2),
               "most_frequent_sample": r["most_frequent"]}
    if "metrics" in r:
        summary.update(r["metrics"]); summary["hybrid_HYBRID_NOT_RAW_QAOA"] = r["hybrid"]
    Path(a.out, "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()