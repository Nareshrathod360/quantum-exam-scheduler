"""Tests for Ising Hamiltonian mapping and energy diagonal evaluation."""

import numpy as np
import pytest

from exam_scheduler.data import ring_instance, sample_instance
from exam_scheduler.ising import IsingModel, ising_energies_all, qubo_to_ising
from exam_scheduler.qubo import build_qubo, qubo_energies_all


def test_ising_sample_k3():
    """Test Ising Hamiltonian properties and equivalence with QUBO on all 2^15 states."""
    inst = sample_instance()
    qubo_model = build_qubo(inst, n_slots=3, penalty_one_hot=1.5, penalty_conflict=1.5)
    ising_model = qubo_to_ising(qubo_model)

    assert ising_model.n_qubits == 15

    # Check terms breakdown
    z_count = sum(1 for val in ising_model.h if abs(val) > 1e-12)
    zz_count = len(ising_model.J)
    pauli_count = len(ising_model.operator)

    assert z_count == 15
    assert zz_count == 30
    assert pauli_count == 46  # 15 Z + 30 ZZ + 1 Identity

    # Verify that Ising diagonal energies match QUBO energies on ALL 2^15 states
    q_energies = qubo_energies_all(qubo_model)
    i_energies = ising_energies_all(ising_model)

    assert np.allclose(i_energies, q_energies, atol=1e-8)


def test_ising_small_ring3_k2():
    """Test Ising conversion matches QUBO on all 2^6 states for ring_instance(3) with K=2."""
    inst = ring_instance(3, seed=42)
    qubo_model = build_qubo(inst, n_slots=2, penalty_one_hot=1.0, penalty_conflict=1.0)
    ising_model = qubo_to_ising(qubo_model)

    assert ising_model.n_qubits == 6

    q_energies = qubo_energies_all(qubo_model)
    i_energies = ising_energies_all(ising_model)

    assert np.allclose(i_energies, q_energies, atol=1e-8)


def test_ising_energies_too_large_error():
    """Test ValueError is raised when number of qubits exceeds 18."""
    dummy_ising = IsingModel(
        operator=None,
        h=np.array([]),
        J={},
        offset=0.0,
        n_qubits=19,
    )
    with pytest.raises(ValueError, match="exceeds 18"):
        ising_energies_all(dummy_ising)
