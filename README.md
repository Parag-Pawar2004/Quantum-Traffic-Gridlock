# Quantum Traffic Gridlock — QUBO + QAOA

IBM Qiskit Fall Fest 2026 · QC@IITI · Q-knight (Parag Vijay Pawar)

## Run

```bash
pip install -r requirements.txt
python run.py --scenario scenarios/scenario_R.json --seed 0 --out outputs/R
```

`run.py` builds the QUBO, trains QAOA (p = 4, at most 300 objective evaluations), runs the bound
circuit for exactly 4096 shots and writes:

| File | Content |
| --- | --- |
| `qubo.json` | QUBO matrix + offset, Ising h/J, variable-to-schedule-bit mapping (no auxiliary variables) |
| `circuit.qasm` | Final bound-parameter QAOA circuit, OpenQASM 3 |
| `counts.json` | 4096 raw shots, bitstring → count; key = schedule A1A2A3A4B1B2B3B4 |
| `schedule.json` | Decoded schedule(s), bit order, raw-QAOA vs. separately labelled HYBRID result |
| `metrics.json` | Benchmark diagnostics (C*, C_rand, lift, enrichment) |

Bit order: A slots 1–4 then B slots 1–4; 1 = NS green, 0 = EW green.
`./make_outputs.sh` regenerates `outputs/` for every scenario in `scenarios/`.

## Files

- `traffic_qaoa.py` — traffic recursion, QUBO builder, QAOA, metrics, baselines
- `run.py` — required CLI
- `quantum_traffic_gridlock.ipynb` — full analysis with outputs: scorer check, baselines, 5-seed robustness, ablations, generalisation
- `scorer.py`, `scenarios/scenario_R.json` — official files from the organisers, unmodified
- `scenarios/scenario_H.json` — shared evaluation scenario, added by the team once released

## Results on R (seeds 0–4)

QUBO minimiser = true optimum 11101010 (cost 72.0, regret 0 %). Raw QAOA: lift 0.774–0.782,
elite enrichment 11.4–12.0, E_Q ≈ 89.6 (C_rand 151.25, C* 72.0). Circuit: 8 qubits, depth 118
(192 CX, 128 RZ, 32 RX, 8 H), 300 training evaluations, 4096 shots, under 1 s per seed.
See the report for baselines and ablations.

## Scenario H

Place the released file at `scenarios/scenario_H.json` and run `./make_outputs.sh`
(writes `outputs/H/`). The notebook processes every JSON in `scenarios/` unchanged.

Results on H (seeds 0–4, same code and settings as R):

| Quantity | Value |
| --- | --- |
| Exact optimum C* / schedule | 234.0 / 10000111 |
| Uniform-random mean C_rand | 300.28 |
| QUBO minimiser cost (regret) | 236.0 (regret 0.85 %), schedule 10010110 |
| Best alternating / greedy cost | 261.0 / 244.0 |
| Raw QAOA E_Q (seed 0) | 243.85 |
| Worst-seed lift / mean enrichment | 0.835 / 16.20 |
| Lowest-energy raw sample / HYBRID | 10010110 (cost 236.0) / 10000111 (cost 234.0) |

Lift ranges 0.835–0.872 across seeds 0–4 (mean 0.857); enrichment 15.3–16.8.

## Report and AI use

The 3-page technical report and the AI-use disclosure are in the team's Google Drive folder: https://drive.google.com/drive/folders/1j56_9tw90O9lJvf2DVsiwCj4XSub0y3K?usp=sharing.
