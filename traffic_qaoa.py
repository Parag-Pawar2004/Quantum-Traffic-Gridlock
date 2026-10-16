"""Quantum traffic gridlock: build the QUBO and run the QAOA pipeline.

The code is designed to work with different traffic scenarios.

Bit order throughout the file:
x = (A1, A2, A3, A4, B1, B2, B3, B4)
A value of 1 means NS is green; 0 means EW is green.

Schedule bit i maps to qubit (7 - i). This means a Qiskit counts key
(q7 ... q0) can be read directly as the schedule string
"A1A2A3A4B1B2B3B4".
"""
from __future__ import annotations

import itertools
import json
import math
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

N_BITS = 8
NODES = ("A", "B")
PHASES = ("NS", "EW")
BIT_NAMES = [f"{n}{t + 1}" for n in NODES for t in range(4)]
ALL_X = np.array(list(itertools.product([0, 1], repeat=N_BITS)), dtype=int)  # Each row contains one schedule; MSB is A1


# ----------------------------------------------------------------------------------------
# 1. Traffic model: follows the official scorer and supplies values for the QUBO
# ----------------------------------------------------------------------------------------
def load_scenario(path):
    return json.loads(Path(path).read_text())


def transfer(fraction, departures):
    """Round fraction * departures to the nearest integer, with halves rounded up."""
    amt = Decimal(str(fraction)) * Decimal(int(departures))
    return int(amt.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def simulate(sc, bits):
    """Run the queue update for a schedule.

    Returns the queue after service and any overflow for each node, phase, and time:
    terms[(node, phase, t)] = (queue after service, overflow above storage).
    """
    plans = {"A": list(bits[:4]), "B": list(bits[4:])}
    cap, sto = sc["service_capacity"], sc["storage_capacity"]
    link = sc["link"]
    q = {(n, p): 0 for n in NODES for p in PHASES}
    pending = [0] * 4
    terms = {}
    for t in range(4):
        for n in NODES:
            for p in PHASES:
                inflow = sc["arrivals"][n][p][t] + (pending[t] if (n == "B" and p == link["phase"]) else 0)
                avail = q[(n, p)] + inflow
                green = plans[n][t] == (1 if p == "NS" else 0)
                dep = min(avail, cap) if green else 0
                q[(n, p)] = avail - dep
                terms[(n, p, t)] = (q[(n, p)], max(0, q[(n, p)] - sto))
                if n == "A" and p == link["phase"] and t < 3:
                    pending[t + 1] += transfer(link["fraction"], dep)
    return terms


def official_cost(sc, bits):
    terms = simulate(sc, bits)
    delay = sum(v[0] for v in terms.values())
    over = sum(v[1] for v in terms.values())
    sw = sum(bits[n * 4 + t] != bits[n * 4 + t - 1] for n in range(2) for t in range(1, 4))
    return float(delay + sc["overflow_weight"] * over + sc["switch_weight"] * sw)


def all_costs(sc):
    """Calculate the official cost for all 256 schedules.

    Index k corresponds to format(k, '08b'); this is used for benchmarking.
    """
    return np.array([official_cost(sc, list(x)) for x in ALL_X])


def term_support(sc, node, phase, t):
    """Find the schedule bits that can affect a queue term at time t."""
    own = [NODES.index(node) * 4 + s for s in range(t + 1)]
    if node == "B" and phase == sc["link"]["phase"]:
        own += [s for s in range(t)]  # Earlier A slots can feed B through the one-slot link delay
    return sorted(set(own))


# ----------------------------------------------------------------------------------------
# 2. Build the QUBO
# ----------------------------------------------------------------------------------------
def _walsh_quadratic(values, support):
    """Fit the degree-two Walsh expansion on a local set of schedule bits.

    values[j] corresponds to the local assignment bits of j, with the MSB
    matching support[0]. Returns the constant, linear terms, and pairwise
    terms in spin variables s = 1 - 2x, where s is the Z eigenvalue.
    """
    k = len(support)
    loc = np.array(list(itertools.product([0, 1], repeat=k)), dtype=int)
    s = 1 - 2 * loc
    const = values.mean()
    h, J = {}, {}
    for a in range(k):
        h[support[a]] = float((values * s[:, a]).mean())
        for b in range(a + 1, k):
            J[(support[a], support[b])] = float((values * s[:, a] * s[:, b]).mean())
    return float(const), h, J


def build_qubo(sc, method="walsh2", overflow_scale=1.0):
    """Build the Ising and QUBO forms of the traffic cost.

    The returned dictionary contains the Ising terms (const, h[8], and
    upper-triangular J) and the QUBO terms (offset and upper-triangular Q).

    With method="walsh2" (the default), each queue and overflow term is
    evaluated over the bits in its causal light-cone, then approximated
    using a degree-two Walsh expansion. Switching costs remain exact.

    With method="fluid", a linear fluid-queue approximation is used without
    the max(0, ...) clipping.
    """
    const, h, J = 0.0, np.zeros(N_BITS), np.zeros((N_BITS, N_BITS))
    wo, ws = sc["overflow_weight"] * overflow_scale, sc["switch_weight"]
    cap, sto, link = sc["service_capacity"], sc["storage_capacity"], sc["link"]
    if method == "walsh2":
        for n in NODES:
            for p in PHASES:
                for t in range(4):
                    sup = term_support(sc, n, p, t)
                    vals = []
                    for loc in itertools.product([0, 1], repeat=len(sup)):
                        bits = [0] * N_BITS
                        for i, b in zip(sup, loc):
                            bits[i] = b
                        qv, ov = simulate(sc, bits)[(n, p, t)]
                        vals.append(qv + wo * ov)
                    c0, hh, JJ = _walsh_quadratic(np.array(vals, float), sup)
                    const += c0
                    for i, v in hh.items():
                        h[i] += v
                    for (i, j), v in JJ.items():
                        J[i, j] += v
    elif method == "fluid":
        # Approximate the queue using cumulative arrivals and transfers minus service capacity.
        # For NS, green is x; for EW, green is 1-x. The result is converted to spin variables below.
        Qx = np.zeros((N_BITS, N_BITS)); lin = np.zeros(N_BITS); off = 0.0
        def add_sq(coef_vec, c, w):  # Add w * (coef.x + c)^2 to the quadratic expression
            nonlocal off
            for i in range(N_BITS):
                lin[i] += w * (coef_vec[i] ** 2 + 2 * c * coef_vec[i])
                for j in range(i + 1, N_BITS):
                    Qx[i, j] += w * 2 * coef_vec[i] * coef_vec[j]
            off += w * c * c
        for n in NODES:
            for p in PHASES:
                for t in range(4):
                    vec = np.zeros(N_BITS); c = float(sum(sc["arrivals"][n][p][:t + 1]))
                    for tau in range(t + 1):
                        i = NODES.index(n) * 4 + tau
                        if p == "NS":
                            vec[i] -= cap
                        else:
                            vec[i] += cap; c -= cap
                        if n == "B" and p == link["phase"] and tau >= 1:
                            j = tau - 1  # The earlier A slot that feeds this B slot
                            add = link["fraction"] * cap
                            if link["phase"] == "NS":
                                vec[j] += add
                            else:
                                vec[j] -= add; c += add
                    lin += vec; off += c                         # Add the approximate queue delay
                    add_sq(vec, c - sto, wo / (2.0 * max(sto, 1)))  # Approximate overflow with a quadratic penalty
        # Convert the polynomial in x to spin variables using x = (1 - s)/2
        for i in range(N_BITS):
            off += lin[i] / 2; h[i] += -lin[i] / 2
            for j in range(i + 1, N_BITS):
                q = Qx[i, j]
                off += q / 4; h[i] += -q / 4; h[j] += -q / 4; J[i, j] += q / 4
        const += off
    else:
        raise ValueError(method)
    # Switching cost is exact: ws * [x_t != x_{t+1}] = ws * (1 - s_t s_{t+1}) / 2
    for n in range(2):
        for t in range(3):
            i, j = n * 4 + t, n * 4 + t + 1
            const += ws / 2
            J[i, j] += -ws / 2
    return ising_to_qubo({"const": const, "h": h, "J": J, "method": method})


def ising_to_qubo(model):
    """Convert the Ising expression into its QUBO form.

    E(s) = const + sum h_i s_i + sum J_ij s_i s_j, with s_i = 1 - 2x_i.
    The result is E(x) = offset + sum_i Q_ii x_i + sum_{i<j} Q_ij x_i x_j.
    """
    const, h, J = model["const"], model["h"], model["J"]
    Q = np.zeros((N_BITS, N_BITS)); off = const
    for i in range(N_BITS):
        off += h[i]; Q[i, i] += -2 * h[i]
        for j in range(i + 1, N_BITS):
            off += J[i, j]; Q[i, i] += -2 * J[i, j]; Q[j, j] += -2 * J[i, j]; Q[i, j] += 4 * J[i, j]
    model.update({"Q": Q, "offset": off})
    return model


def qubo_energies(model):
    """Calculate the surrogate energy for all 256 schedules."""
    X = ALL_X
    Q = model["Q"]
    return model["offset"] + np.einsum("ki,ij,kj->k", X, np.triu(Q), X)


# ----------------------------------------------------------------------------------------
# 3. QAOA: train with an exact NumPy statevector and build the final Qiskit circuit
# ----------------------------------------------------------------------------------------
def qubit_of(i):
    return N_BITS - 1 - i


def basis_energy_vector(model, scale):
    """Return the normalised cost Hamiltonian diagonal in Qiskit's basis order.

    In basis index k, qubit q holds bit q of k. Since qubit 7-i stores
    schedule bit i, format(k, '08b') gives the schedule string.
    """
    return (qubo_energies(model) - model["const"]) / scale  # Drop the constant because it only adds a global phase


def ising_scale(model):
    """Scale the spectrum so its standard deviation over uniform schedules is 1."""
    return float(math.sqrt(np.sum(model["h"] ** 2) + np.sum(np.triu(model["J"], 1) ** 2))) or 1.0


def qaoa_state(Ediag, gammas, betas):
    psi = np.full(2 ** N_BITS, 1 / 16.0, dtype=complex)
    for g, b in zip(gammas, betas):
        psi = psi * np.exp(-1j * g * Ediag)
        c, s = math.cos(b), -1j * math.sin(b)
        t = psi.reshape([2] * N_BITS)
        for ax in range(N_BITS):  # Apply exp(-i b X) to each qubit
            t0 = np.take(t, 0, axis=ax); t1 = np.take(t, 1, axis=ax)
            t = np.stack([c * t0 + s * t1, s * t0 + c * t1], axis=ax)
        psi = t.reshape(-1)
    return psi


class BudgetExceeded(Exception):
    pass


def optimise_qaoa(Ediag, p=4, seed=0, max_evals=300, init="tqa", scan=10):
    """Find the best gamma and beta values and return them with the evaluation history.

    Each statevector expectation calculation counts as one evaluation.
    """
    rng = np.random.default_rng(seed)
    hist = []
    best = {"f": np.inf, "x": None}

    def f(theta):
        if len(hist) >= max_evals:
            raise BudgetExceeded
        psi = qaoa_state(Ediag, theta[:p], theta[p:])
        val = float(np.real(np.vdot(psi, Ediag * psi)))
        hist.append(val)
        if val < best["f"]:
            best["f"], best["x"] = val, np.array(theta, float)
        return val

    def tqa(dt):
        k = (np.arange(p) + 0.5) / p
        # Use a Trotterised linear anneal from -sum X to H_C; this gives negative beta values.
        # exp(-i dt (1-s)(-X)) = exp(-i beta X), so beta = -(1-s) dt.
        return np.concatenate([k * dt, -(1 - k) * dt])

    try:
        if init == "tqa":
            grid = np.linspace(0.5, 1.25 * p + 0.5, scan)
            vals = [f(tqa(dt)) for dt in grid]
            x0 = tqa(grid[int(np.argmin(vals))])
            x0 = x0 + rng.normal(0, 0.02, size=x0.shape)  # Add a small seed-dependent variation
        else:  # Use random initial values for the ablation run
            x0 = np.concatenate([rng.uniform(0, math.pi, p), rng.uniform(-math.pi / 2, math.pi / 2, p)])
        remaining = max_evals - len(hist)
        minimize(f, x0, method="COBYLA", options={"maxiter": remaining, "rhobeg": 0.25})
        # If there is budget left, try a final refinement with Nelder-Mead
        remaining = max_evals - len(hist)
        if remaining > 2 * p + 5:
            minimize(f, best["x"], method="Nelder-Mead",
                     options={"maxfev": remaining, "xatol": 1e-4, "fatol": 1e-7, "initial_simplex": None})
    except BudgetExceeded:
        pass
    assert len(hist) <= max_evals
    th = best["x"]
    return th[:p], th[p:], hist

def build_circuit(model, scale, gammas, betas, measure=True):
    """Build the parameterised QAOA circuit.

    The cost layer applies exp(-i g H), where H contains the normalised
    h_i Z_i and J_ij Z_i Z_j terms.
    """
    from qiskit import QuantumCircuit
    h, J = model["h"] / scale, model["J"] / scale
    qc = QuantumCircuit(N_BITS)
    qc.h(range(N_BITS))
    for g, b in zip(gammas, betas):
        for i in range(N_BITS):
            for j in range(i + 1, N_BITS):
                if abs(J[i, j]) > 1e-12:
                    qi, qj = qubit_of(i), qubit_of(j)
                    qc.cx(qi, qj); qc.rz(float(2 * g * J[i, j]), qj); qc.cx(qi, qj)
        for i in range(N_BITS):
            if abs(h[i]) > 1e-12:
                qc.rz(float(2 * g * h[i]), qubit_of(i))
        for q in range(N_BITS):
            qc.rx(float(2 * b), q)
    if measure:
        qc.measure_all()
    return qc


def sample_circuit(qc, shots, seed):
    from qiskit.primitives import StatevectorSampler
    res = StatevectorSampler(default_shots=shots, seed=seed).run([qc]).result()
    return dict(sorted(res[0].data.meas.get_counts().items()))


def circuit_probabilities(qc):
    from qiskit.quantum_info import Statevector
    return Statevector(qc.remove_final_measurements(inplace=False)).probabilities()


# ----------------------------------------------------------------------------------------
# 4. Metrics, baselines, and the hybrid approach
# ----------------------------------------------------------------------------------------
def bits_of(key):
    return [int(c) for c in key]


def elite_set(costs, k=13):
    order = sorted(range(256), key=lambda i: (costs[i], format(i, "08b")))
    return set(order[:k])


def sampling_metrics(costs, counts):
    shots = sum(counts.values())
    EQ = sum(costs[int(k, 2)] * v for k, v in counts.items()) / shots
    Cstar, Crand = float(costs.min()), float(costs.mean())
    lift = (Crand - EQ) / (Crand - Cstar) if Crand > Cstar else float("nan")
    el = elite_set(costs)
    frac = sum(v for k, v in counts.items() if int(k, 2) in el) / shots
    return {"E_Q": EQ, "C_star": Cstar, "C_rand": Crand, "lift": lift,
            "elite_fraction": frac, "enrichment": frac / (13 / 256),
            "best_sampled_cost": min(costs[int(k, 2)] for k in counts)}


def baseline_alternating(sc):
    best = None
    for a in ("1010", "0101"):
        for b in ("1010", "0101"):
            c = official_cost(sc, bits_of(a + b))
            if best is None or c < best[1]:
                best = (a + b, c)
    return best


def available_at(sc, bits, t):
    """Get the vehicles present at each approach in slot t.

    Arrivals and transfers are included, while decisions from earlier slots
    are treated as fixed.
    """
    cap, link = sc["service_capacity"], sc["link"]
    q = {(n, p): 0 for n in NODES for p in PHASES}
    pending = [0] * 4
    for s in range(t + 1):
        avail = {}
        for n in NODES:
            for p in PHASES:
                avail[(n, p)] = q[(n, p)] + sc["arrivals"][n][p][s] + (pending[s] if (n == "B" and p == link["phase"]) else 0)
        if s == t:
            return avail
        for n_i, n in enumerate(NODES):
            for p in PHASES:
                green = bits[n_i * 4 + s] == (1 if p == "NS" else 0)
                dep = min(avail[(n, p)], cap) if green else 0
                q[(n, p)] = avail[(n, p)] - dep
                if n == "A" and p == link["phase"] and s < 3:
                    pending[s + 1] += transfer(link["fraction"], dep)


def baseline_greedy(sc):
    """Choose the phase with the larger queue in each slot; ties go to NS."""
    bits = [0] * 8
    for t in range(4):
        av = available_at(sc, bits, t)
        for n_i, n in enumerate(NODES):
            bits[n_i * 4 + t] = 1 if av[(n, "NS")] >= av[(n, "EW")] else 0
    return "".join(map(str, bits)), official_cost(sc, bits)


def hybrid_refine(sc, counts, top=16):
    """Run a separate local search starting from the most frequent QAOA samples.

    The top samples are refined using steepest descent with single-bit flips
    and the official cost function. This hybrid result is reported separately
    and is not part of the raw QAOA output.
    """
    cand = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    best = None; evals = 0
    for key, _ in cand:
        x = bits_of(key); c = official_cost(sc, x); evals += 1
        while True:
            nb = []
            for i in range(8):
                y = x.copy(); y[i] ^= 1; nb.append((official_cost(sc, y), y)); evals += 1
            cmin, ymin = min(nb, key=lambda t: t[0])
            if cmin < c - 1e-12:
                x, c = ymin, cmin
            else:
                break
        if best is None or c < best[1]:
            best = ("".join(map(str, x)), c)
    return {"schedule": best[0], "cost": best[1], "official_evaluations": evals}


# ----------------------------------------------------------------------------------------
# 5. Main pipeline, called by run.py
# ----------------------------------------------------------------------------------------
def run_pipeline(sc, seed=0, p=4, shots=4096, out=None, method="walsh2", max_evals=300,
                 init="tqa", overflow_scale=1.0, with_diagnostics=True):
    model = build_qubo(sc, method=method, overflow_scale=overflow_scale)
    scale = ising_scale(model)
    Ediag = basis_energy_vector(model, scale)
    gammas, betas, hist = optimise_qaoa(Ediag, p=p, seed=seed, max_evals=max_evals, init=init)
    qc = build_circuit(model, scale, gammas, betas)
    counts = sample_circuit(qc, shots, seed)
    assert sum(counts.values()) == shots
    # Check that the circuit and NumPy statevector give matching probabilities
    p_np = np.abs(qaoa_state(Ediag, gammas, betas)) ** 2
    p_qk = circuit_probabilities(qc)
    consistency = float(np.max(np.abs(p_np - p_qk)))

    E = qubo_energies(model)
    kq = int(np.argmin(E))
    most = max(counts.items(), key=lambda kv: (kv[1], -int(kv[0], 2)))[0]
    lowest_qubo_sample = min(counts, key=lambda k: (E[int(k, 2)], k))
    result = {"model": model, "scale": scale, "gammas": gammas, "betas": betas, "history": hist,
              "circuit": qc, "counts": counts, "consistency_max_abs_prob_diff": consistency,
              "qubo_argmin": format(kq, "08b"), "most_frequent": most,
              "lowest_qubo_sample": lowest_qubo_sample}
    if with_diagnostics:
        costs = all_costs(sc)  # Enumerate all schedules for benchmarking only
        result["costs"] = costs
        result["metrics"] = sampling_metrics(costs, counts)
        result["hybrid"] = hybrid_refine(sc, counts)
    if out is not None:
        write_outputs(sc, result, out, seed, p)
    return result


def write_outputs(sc, r, out, seed, p):
    from qiskit import qasm3
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    m = r["model"]
    bit_order = "A1 A2 A3 A4 B1 B2 B3 B4 (left to right); 1 = NS green, 0 = EW green"
    variables = [{"index": i, "name": BIT_NAMES[i], "qubit": qubit_of(i), "schedule_bit": i,
                  "auxiliary": False} for i in range(N_BITS)]
    qubo = {
        "scenario": sc.get("name"), "method": m["method"],
        "energy_convention": "E(x) = offset + sum_i Q[i][i] x_i + sum_{i<j} Q[i][j] x_i x_j, x_i in {0,1}",
        "Q": np.triu(m["Q"]).round(10).tolist(), "offset": round(float(m["offset"]), 10),
        "ising": {"convention": "E(s) = const + sum_i h_i s_i + sum_{i<j} J_ij s_i s_j, s_i = 1 - 2 x_i = <Z_qubit(i)>",
                  "h": m["h"].round(10).tolist(), "J": np.triu(m["J"], 1).round(10).tolist(),
                  "const": round(float(m["const"]), 10)},
        "circuit_normalisation": {"divide_h_J_by": r["scale"], "note": "constant dropped (global phase)"},
        "variables": variables, "auxiliary_variables": [], "num_qubits": N_BITS,
        "bit_order": bit_order,
        "qiskit_key_mapping": "counts key (q7..q0) == schedule string A1..B4",
    }
    (out / "qubo.json").write_text(json.dumps(qubo, indent=2))
    (out / "circuit.qasm").write_text(qasm3.dumps(r["circuit"]))
    (out / "counts.json").write_text(json.dumps(r["counts"], indent=1))
    sched = {
        "bit_order": bit_order,
        "raw_qaoa": {
            "most_frequent_sample": r["most_frequent"],
            "lowest_surrogate_energy_sample": r["lowest_qubo_sample"],
        },
        "qubo_exact_minimiser": r["qubo_argmin"],
        "seed": seed, "p": p, "training_evaluations": len(r["history"]),
        "gammas": [float(g) for g in r["gammas"]], "betas": [float(b) for b in r["betas"]],
        "consistency_max_abs_prob_diff": r["consistency_max_abs_prob_diff"],
    }
    sel = r["lowest_qubo_sample"]
    sched["selected_schedule"] = {"bits": sel, "A": bits_of(sel)[:4], "B": bits_of(sel)[4:],
                                  "source": "raw QAOA: among the 4096 sampled bitstrings, the one with the lowest "
                                            "surrogate (QUBO) energy - no official-scorer calls, no local search"}
    if "metrics" in r:
        costs = r["costs"]
        for key in ("most_frequent_sample", "lowest_surrogate_energy_sample"):
            b = sched["raw_qaoa"][key]
            sched["raw_qaoa"][key] = {"bits": b, "official_cost": float(costs[int(b, 2)])}
        sched["qubo_exact_minimiser"] = {"bits": r["qubo_argmin"],
                                         "official_cost": float(costs[int(r["qubo_argmin"], 2)])}
        sched["hybrid_HYBRID_NOT_RAW_QAOA"] = r["hybrid"]
        sched["diagnostics_benchmark_only"] = r["metrics"]
    (out / "schedule.json").write_text(json.dumps(sched, indent=2))


# ----------------------------------------------------------------------------------------
# 6. Random scenario generator, used only for generalisation experiments
# ----------------------------------------------------------------------------------------
def make_random_scenario(rng, name="rand"):
    """Generate a scenario using the hidden-scenario ranges from the problem statement."""
    return {"name": name,
            "arrivals": {n: {p: [int(v) for v in rng.integers(0, 9, 4)] for p in PHASES} for n in NODES},
            "service_capacity": int(rng.integers(4, 7)), "storage_capacity": int(rng.integers(5, 9)),
            "overflow_weight": float(rng.integers(2, 7)),
            "switch_weight": float(rng.choice([0.25, 0.5, 0.75, 1.0, 1.25, 1.5])),
            "link": {"from": "A", "to": "B", "phase": str(rng.choice(["NS", "EW"])),
                     "fraction": float(rng.choice([0.4, 0.5, 0.6, 0.7, 0.8]))}}