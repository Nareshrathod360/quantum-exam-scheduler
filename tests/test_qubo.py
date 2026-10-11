"""Tests for QUBO formulation, matrix building, energy evaluation, and bit decoding."""

import itertools
import numpy as np
import pytest

from exam_scheduler.data import sample_instance
from exam_scheduler.qubo import (
    bitstring_to_bits,
    bits_from_assignment,
    build_qubo,
    decode_bits,
    direct_energy,
    qubo_energies_all,
    qubo_energy,
)
from exam_scheduler.validator import schedule_objective, validate_schedule


def test_qubo_model_structure():
    """Test QUBO upper-triangular structure, dimensions, and constant offset."""
    inst = sample_instance()
    model = build_qubo(inst, n_slots=3, penalty_one_hot=1.5, penalty_conflict=1.5)

    assert model.n_vars == 15
    assert model.Q.shape == (15, 15)
    # Check upper-triangular
    assert np.allclose(model.Q, np.triu(model.Q))
    assert model.constant == pytest.approx(7.5)


def test_qubo_energy_vs_direct_energy():
    """Test qubo_energy matches direct_energy on 300 random bit vectors."""
    inst = sample_instance()
    model = build_qubo(inst, n_slots=3, penalty_one_hot=1.5, penalty_conflict=1.5)
    rng = np.random.default_rng(0)

    for _ in range(300):
        bits = [int(b) for b in rng.integers(0, 2, size=model.n_vars)]
        qe = qubo_energy(model, bits)
        de = direct_energy(inst, 3, bits, 1.5, 1.5)
        assert qe == pytest.approx(de, abs=1e-8)


def test_qubo_energies_all_vectorized():
    """Test vectorized qubo_energies_all matches qubo_energy on 200 sampled states."""
    inst = sample_instance()
    model = build_qubo(inst, n_slots=3, penalty_one_hot=1.5, penalty_conflict=1.5)
    all_energies = qubo_energies_all(model)

    assert len(all_energies) == 1 << 15
    rng = np.random.default_rng(0)
    sample_indices = rng.choice(1 << 15, size=200, replace=False)

    for s in sample_indices:
        bits = [(int(s) >> q) & 1 for q in range(model.n_vars)]
        assert all_energies[s] == pytest.approx(qubo_energy(model, bits), abs=1e-8)


def test_qubo_ground_state_sample_k3():
    """Test that sample instance K=3 has a unique valid ground state decoding to [0,1,0,2,1]."""
    inst = sample_instance()
    model = build_qubo(inst, n_slots=3, penalty_one_hot=1.5, penalty_conflict=1.5)
    all_energies = qubo_energies_all(model)

    min_e = float(all_energies.min())
    ground_indices = np.where(np.abs(all_energies - min_e) < 1e-9)[0]
    assert len(ground_indices) == 1

    s0 = int(ground_indices[0])
    bits0 = [(s0 >> q) & 1 for q in range(model.n_vars)]
    dec0 = decode_bits(model, bits0)

    assert dec0 == [0, 1, 0, 2, 1]
    assert min_e == pytest.approx(2.1, abs=1e-3)


def test_qubo_penalty_thresholds():
    """Test ground states are valid for penalties 1.5, 2.0, 3.0, but not all valid for 1.0."""
    inst = sample_instance()

    for p in [1.5, 2.0, 3.0]:
        model = build_qubo(inst, n_slots=3, penalty_one_hot=p, penalty_conflict=p)
        energies = qubo_energies_all(model)
        min_e = float(energies.min())
        ground_indices = np.where(np.abs(energies - min_e) < 1e-9)[0]
        assert all(
            (None not in decode_bits(model, [(int(idx) >> q) & 1 for q in range(15)]))
            and validate_schedule(
                inst,
                decode_bits(model, [(int(idx) >> q) & 1 for q in range(15)]),
                3,
            ).is_valid
            for idx in ground_indices
        )

    # For penalty 1.0, not all ground states are valid
    model_low = build_qubo(inst, n_slots=3, penalty_one_hot=1.0, penalty_conflict=1.0)
    energies_low = qubo_energies_all(model_low)
    min_e_low = float(energies_low.min())
    ground_indices_low = np.where(np.abs(energies_low - min_e_low) < 1e-9)[0]
    valid_flags = [
        (None not in decode_bits(model_low, [(int(idx) >> q) & 1 for q in range(15)]))
        and validate_schedule(
            inst,
            decode_bits(model_low, [(int(idx) >> q) & 1 for q in range(15)]),
            3,
        ).is_valid
        for idx in ground_indices_low
    ]
    assert not all(valid_flags)


def test_qubo_all_one_hot_assignments_relation():
    """Test QUBO energy formula against validator conflicts for all 3^5 one-hot assignments."""
    inst = sample_instance()
    model = build_qubo(inst, n_slots=3, penalty_one_hot=1.5, penalty_conflict=1.5)

    valid_count = 0
    for a in itertools.product(range(3), repeat=5):
        assignment = list(a)
        bits = bits_from_assignment(model, assignment)
        val = validate_schedule(inst, assignment, 3)

        expected_energy = schedule_objective(inst, assignment) + 1.5 * val.n_conflicts
        assert qubo_energy(model, bits) == pytest.approx(expected_energy, abs=1e-8)

        if val.is_valid:
            valid_count += 1
            assert qubo_energy(model, bits) == pytest.approx(
                schedule_objective(inst, assignment), abs=1e-8
            )

    assert valid_count == 30


def test_qubo_k2_no_valid_states():
    """Test that with K=2 on sample instance, no state decodes to a valid schedule."""
    inst = sample_instance()
    model_k2 = build_qubo(inst, n_slots=2)
    assert model_k2.n_vars == 10

    for s in range(1 << 10):
        bits = [(s >> q) & 1 for q in range(10)]
        dec = decode_bits(model_k2, bits)
        if None not in dec:
            val = validate_schedule(inst, dec, 2)
            assert not val.is_valid


def test_bit_encoding_decoding_helpers():
    """Test bitstring reversing, decode edge cases, and round-trip conversion."""
    assert bitstring_to_bits("001") == [1, 0, 0]
    assert bitstring_to_bits("1010") == [0, 1, 0, 1]

    inst = sample_instance()
    model = build_qubo(inst, 3)

    # Course with zero ones -> None; course with two ones -> None
    # 15 bits: course 0 has two 1s [1, 1, 0], course 1 has zero 1s [0, 0, 0]
    test_bits = [1, 1, 0, 0, 0, 0] + [1, 0, 0] * 3
    dec = decode_bits(model, test_bits)
    assert dec[0] is None
    assert dec[1] is None
    assert dec[2] == 0
    assert dec[3] == 0
    assert dec[4] == 0

    # Round trip test
    assignment = [0, 2, 1, 0, 1]
    bits = bits_from_assignment(model, assignment)
    assert decode_bits(model, bits) == assignment


def test_build_qubo_invalid_parameters():
    """Test ValueError for invalid n_slots and non-positive penalties."""
    inst = sample_instance()
    with pytest.raises(ValueError, match="n_slots must be >= 1"):
        build_qubo(inst, n_slots=0)

    with pytest.raises(ValueError, match="strictly positive"):
        build_qubo(inst, n_slots=3, penalty_one_hot=0.0)

    with pytest.raises(ValueError, match="strictly positive"):
        build_qubo(inst, n_slots=3, penalty_conflict=-1.0)
