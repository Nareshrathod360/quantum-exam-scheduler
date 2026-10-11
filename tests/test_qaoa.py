"""Tests for QAOA solver, random sampling baseline, and circuit characteristics."""

import numpy as np
import pytest

from exam_scheduler.classical import random_sampling_schedule
from exam_scheduler.data import generate_instance, ring_instance, sample_instance
from exam_scheduler.ising import qubo_to_ising
from exam_scheduler.qaoa_solver import qaoa_schedule
from exam_scheduler.qubo import (
    bitstring_to_bits,
    build_qubo,
    decode_bits,
    valid_state_mask,
)
from exam_scheduler.validator import schedule_objective, validate_schedule


def test_valid_state_mask():
    """Test valid_state_mask length and true count for sample and ring instances."""
    m_sample = valid_state_mask(sample_instance(), n_slots=3)
    assert len(m_sample) == 1 << 15
    assert m_sample.sum() == 30

    inst_ring3 = ring_instance(3, seed=42)
    m_ring = valid_state_mask(inst_ring3, n_slots=3)
    assert len(m_ring) == 1 << 9
    assert m_ring.sum() == 6  # Exactly 3! proper colorings for triangle K3


def test_qaoa_circuit_stats():
    """Test circuit statistics and parameter naming for p=1 and p=2."""
    inst = ring_instance(3, seed=42)
    model = build_qubo(inst, 3)
    ising = qubo_to_ising(model)
    zz_terms_count = len(ising.J)

    # Reps = 1
    outcome_p1 = qaoa_schedule(
        inst,
        n_slots=3,
        reps=1,
        n_restarts=1,
        maxiter=10,
        shots=128,
        seed=42,
    )
    assert outcome_p1.circuit_stats["num_qubits"] == 9
    assert outcome_p1.circuit_stats["num_parameters"] == 2
    assert outcome_p1.circuit_stats["cx"] == 2 * zz_terms_count * 1
    # Check that transpiled circuit used required basis gates
    assert outcome_p1.circuit_stats["basis_gates"] == ["rz", "sx", "x", "cx"]

    # Reps = 2
    outcome_p2 = qaoa_schedule(
        inst,
        n_slots=3,
        reps=2,
        n_restarts=1,
        maxiter=5,
        shots=128,
        seed=42,
    )
    assert outcome_p2.circuit_stats["num_parameters"] == 4
    assert outcome_p2.param_names == ["β[0]", "β[1]", "γ[0]", "γ[1]"]


def test_qaoa_outcome_consistency():
    """Test consistency across shots, yield, validity, and decoded minimum objective."""
    inst = ring_instance(3, seed=42)
    outcome = qaoa_schedule(
        inst,
        n_slots=3,
        reps=1,
        n_restarts=1,
        maxiter=25,
        shots=512,
        seed=42,
    )

    assert outcome.feasibility_yield == pytest.approx(
        outcome.n_valid_shots / outcome.shots
    )
    assert sum(outcome.counts.values()) == outcome.shots

    model = build_qubo(inst, 3)
    valid_objs = []
    for bs in outcome.counts:
        bits = bitstring_to_bits(bs)
        assignment = decode_bits(model, bits)
        if None not in assignment:
            if validate_schedule(inst, assignment, 3).is_valid:
                valid_objs.append(schedule_objective(inst, assignment))

    if outcome.n_valid_shots > 0:
        assert outcome.result.is_valid is True
        assert validate_schedule(inst, outcome.result.assignment, 3).is_valid
        assert outcome.result.objective == pytest.approx(min(valid_objs))
    else:
        assert outcome.result.assignment is None
        assert outcome.result.is_valid is False


def test_qaoa_exact_feasible_probability():
    """Test exact feasible probability computed from Statevector is in [0, 1]."""
    inst = ring_instance(3, seed=42)
    outcome = qaoa_schedule(
        inst,
        n_slots=3,
        reps=1,
        n_restarts=1,
        maxiter=15,
        shots=256,
        seed=42,
    )
    assert 0.0 <= outcome.exact_feasible_probability <= 1.0


def test_qaoa_reproducibility():
    """Test deterministic reproducibility of training history and sampling counts."""
    inst = ring_instance(3, seed=42)
    out1 = qaoa_schedule(
        inst,
        n_slots=3,
        reps=1,
        n_restarts=1,
        maxiter=20,
        shots=256,
        seed=101,
    )
    out2 = qaoa_schedule(
        inst,
        n_slots=3,
        reps=1,
        n_restarts=1,
        maxiter=20,
        shots=256,
        seed=101,
    )

    assert out1.counts == out2.counts
    assert np.allclose(out1.params, out2.params)
    assert np.allclose(out1.restarts[0]["history"], out2.restarts[0]["history"])


def test_random_sampling_baseline():
    """Test random sampling baseline determinism, validity, and yield bounds."""
    inst = ring_instance(3, seed=42)

    # Identical seed gives identical result
    res1 = random_sampling_schedule(inst, n_slots=3, shots=500, seed=77)
    res2 = random_sampling_schedule(inst, n_slots=3, shots=500, seed=77)
    assert res1.assignment == res2.assignment
    assert res1.extra["n_valid_shots"] == res2.extra["n_valid_shots"]

    # Large sample run yields between 0.006 and 0.02
    res_large = random_sampling_schedule(inst, n_slots=3, shots=20000, seed=42)
    yield_val = res_large.extra["feasibility_yield"]
    assert 0.006 <= yield_val <= 0.02
    if res_large.assignment is not None:
        assert res_large.is_valid is True


def test_too_many_qubits_error():
    """Test ValueError is raised when problem size exceeds 20 qubits."""
    inst24 = generate_instance(8, 40, seed=42)

    with pytest.raises(ValueError, match="too many qubits"):
        qaoa_schedule(inst24, n_slots=3)

    with pytest.raises(ValueError, match="exceeds 20"):
        valid_state_mask(inst24, n_slots=3)
