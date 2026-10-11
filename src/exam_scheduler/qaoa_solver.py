"""QAOA variational quantum solver module for exam scheduling using Qiskit 2.5.2."""

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.primitives import SamplerV2
from scipy.optimize import minimize

from exam_scheduler.data import DEFAULT_SEED, ExamInstance
from exam_scheduler.ising import qubo_to_ising
from exam_scheduler.qubo import (
    bitstring_to_bits,
    build_qubo,
    decode_bits,
    valid_state_mask,
)
from exam_scheduler.results import SolverResult, make_result
from exam_scheduler.validator import schedule_objective, validate_schedule


@dataclass(frozen=True)
class QaoaOutcome:
    """Comprehensive execution outcome of the QAOA variational quantum algorithm."""

    result: SolverResult
    counts: Dict[str, int]
    shots: int
    n_valid_shots: int
    feasibility_yield: float
    top_bitstring: str
    top_bitstring_valid: bool
    exact_feasible_probability: float
    random_guess_probability: float
    final_energy: float
    best_restart: int
    params: List[float]
    param_names: List[str]
    restarts: List[Dict[str, Any]]
    circuit_stats: Dict[str, Any]
    training_time_s: float
    sampling_time_s: float
    penalty: float
    seed: int


def qaoa_schedule(
    instance: ExamInstance,
    n_slots: int,
    reps: int = 1,
    penalty: float = 1.5,
    shots: int = 4096,
    n_restarts: int = 3,
    maxiter: int = 150,
    seed: int = DEFAULT_SEED,
) -> QaoaOutcome:
    """Solve exam scheduling using the Quantum Approximate Optimization Algorithm (QAOA)."""
    # -------------------------------------------------------------------------
    # Step 1: Mathematical Problem Mapping to QUBO & Ising Cost Hamiltonian
    # -------------------------------------------------------------------------
    # Map decision variables x[c,t] to qubits q = c*K + t.
    N = instance.n_courses * n_slots
    if N > 20:
        raise ValueError(
            f"N={N} qubits is too many qubits for statevector simulation on a laptop (limit is 20)."
        )

    qubo_model = build_qubo(
        instance,
        n_slots=n_slots,
        penalty_one_hot=penalty,
        penalty_conflict=penalty,
    )
    ising_model = qubo_to_ising(qubo_model)

    # -------------------------------------------------------------------------
    # Step 2: Parameterized Quantum Circuit Construction & ISA Transpilation
    # -------------------------------------------------------------------------
    # QAOA alternates Cost Hamiltonian evolution e^(-i * gamma * H_C) and
    # Transverse Mixer Hamiltonian evolution e^(-i * beta * H_M) over p layers.
    raw_ansatz = QAOAAnsatz(cost_operator=ising_model.operator, reps=reps)

    # MEMORY TRAP: passing the raw QAOAAnsatz (a PauliEvolutionGate) to StatevectorEstimator
    # makes it build a dense 2^N x 2^N matrix = 16 GiB at N=15 and crashes.
    # ALWAYS transpile first with generate_preset_pass_manager(optimization_level=1, basis_gates=["rz","sx","x","cx"], seed_transpiler=seed)
    # and use the transpiled circuit for every simulation.
    pass_manager = generate_preset_pass_manager(
        optimization_level=1,
        basis_gates=["rz", "sx", "x", "cx"],
        seed_transpiler=seed,
    )
    isa_circuit = pass_manager.run(raw_ansatz)

    circuit_stats = {
        "num_qubits": int(isa_circuit.num_qubits),
        "num_parameters": int(isa_circuit.num_parameters),
        "reps": int(reps),
        "depth": int(isa_circuit.depth()),
        "size": int(isa_circuit.size()),
        "cx": int(isa_circuit.count_ops().get("cx", 0)),
        "gate_counts": dict(isa_circuit.count_ops()),
        "basis_gates": ["rz", "sx", "x", "cx"],
    }
    param_names = [p.name for p in isa_circuit.parameters]

    # -------------------------------------------------------------------------
    # Step 3: Classical Variational Training (Estimator + COBYLA Optimization)
    # -------------------------------------------------------------------------
    # Run multiple random restarts using a deterministic single random number generator.
    t_train_start = time.perf_counter()
    rng = np.random.default_rng(seed)
    estimator = StatevectorEstimator()

    restarts_data: List[Dict[str, Any]] = []
    best_restart = 0
    best_energy = float("inf")
    best_params: List[float] = []

    for r in range(n_restarts):
        start_params = rng.uniform(0, np.pi, size=2 * reps)
        eval_history: List[float] = []

        def eval_cost(p: np.ndarray) -> float:
            job = estimator.run([(isa_circuit, ising_model.operator, p)])
            energy_val = float(job.result()[0].data.evs)
            eval_history.append(energy_val)
            return energy_val

        opt_result = minimize(
            eval_cost,
            start_params,
            method="COBYLA",
            options={"maxiter": maxiter},
        )
        final_e = float(opt_result.fun)
        opt_p = [float(val) for val in opt_result.x]

        restarts_data.append({
            "restart": r,
            "seed": seed,
            "start_params": [float(val) for val in start_params],
            "final_energy": final_e,
            "n_evaluations": int(opt_result.nfev),
            "params": opt_p,
            "history": eval_history,
        })

        # ---------------------------------------------------------------------
        # Step 4: Selection of Best Parameter Setting by Lowest Cost Energy
        # ---------------------------------------------------------------------
        # Select purely based on energy expectation without peeking at validity
        if final_e < best_energy:
            best_energy = final_e
            best_restart = r
            best_params = opt_p

    training_time_s = time.perf_counter() - t_train_start

    # -------------------------------------------------------------------------
    # Step 5: Exact Diagnostic Statevector Analysis
    # -------------------------------------------------------------------------
    # Compute theoretical ground-truth feasible probability across the full statevector
    mask = valid_state_mask(instance, n_slots)
    random_guess_prob = float(mask.sum() / (1 << N))

    bound_circuit = isa_circuit.assign_parameters(best_params)
    statevector = Statevector(bound_circuit)
    probabilities = statevector.probabilities()
    exact_feasible_prob = float(probabilities[mask].sum())

    # -------------------------------------------------------------------------
    # Step 6: Measurement Sampling via Aer SamplerV2
    # -------------------------------------------------------------------------
    t_sample_start = time.perf_counter()
    meas_circuit = isa_circuit.copy()
    meas_circuit.measure_all()

    sampler = SamplerV2(default_shots=shots, seed=seed)
    samp_job = sampler.run([(meas_circuit, best_params)])
    samp_result = samp_job.result()
    counts = dict(samp_result[0].data.meas.get_counts())
    sampling_time_s = time.perf_counter() - t_sample_start

    # -------------------------------------------------------------------------
    # Step 7: Decoding Sampled Bitstrings & Schedule Validation
    # -------------------------------------------------------------------------
    # Determine the most frequently observed bitstring
    top_bitstring = max(counts.keys(), key=lambda bs: counts[bs])
    top_bits = bitstring_to_bits(top_bitstring)
    top_assign = decode_bits(qubo_model, top_bits)
    top_bitstring_valid = (
        None not in top_assign
        and validate_schedule(instance, top_assign, n_slots).is_valid
    )

    valid_schedules: Dict[str, Tuple[List[int], float]] = {}
    n_valid_shots = 0

    for bs, shot_count in counts.items():
        bits = bitstring_to_bits(bs)
        assignment = decode_bits(qubo_model, bits)
        if None not in assignment:
            val = validate_schedule(instance, assignment, n_slots)
            if val.is_valid:
                n_valid_shots += shot_count
                obj_val = schedule_objective(instance, assignment)
                valid_schedules[bs] = (assignment, obj_val)

    feasibility_yield = float(n_valid_shots / shots)

    # -------------------------------------------------------------------------
    # Step 8: Selection of Final Best Valid Schedule
    # -------------------------------------------------------------------------
    # Pick the lowest objective among valid samples; tie-break lexicographically.
    if valid_schedules:
        _, (best_assignment, _) = min(
            valid_schedules.items(),
            key=lambda item: (item[1][1], item[1][0]),
        )
    else:
        best_assignment = None

    total_runtime_s = training_time_s + sampling_time_s
    result = make_result(
        method=f"qaoa-p{reps}",
        instance=instance,
        assignment=best_assignment,
        n_slots=n_slots,
        runtime_s=total_runtime_s,
        extra={
            "feasibility_yield": feasibility_yield,
            "n_valid_shots": n_valid_shots,
            "shots": shots,
            "reps": reps,
            "penalty": penalty,
            "seed": seed,
            "exact_feasible_probability": exact_feasible_prob,
            "top_bitstring_valid": top_bitstring_valid,
            "final_energy": best_energy,
        },
    )

    return QaoaOutcome(
        result=result,
        counts=counts,
        shots=shots,
        n_valid_shots=n_valid_shots,
        feasibility_yield=feasibility_yield,
        top_bitstring=top_bitstring,
        top_bitstring_valid=top_bitstring_valid,
        exact_feasible_probability=exact_feasible_prob,
        random_guess_probability=random_guess_prob,
        final_energy=best_energy,
        best_restart=best_restart,
        params=best_params,
        param_names=param_names,
        restarts=restarts_data,
        circuit_stats=circuit_stats,
        training_time_s=training_time_s,
        sampling_time_s=sampling_time_s,
        penalty=penalty,
        seed=seed,
    )
