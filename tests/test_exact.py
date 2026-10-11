"""Tests for exact solver and optimality gap calculations."""

import pytest

from exam_scheduler.classical import exact_schedule, greedy_schedule, min_slots_exact
from exam_scheduler.data import (
    generate_instance,
    ring_instance,
    sample_instance,
)
from exam_scheduler.results import optimality_gap


def test_exact_schedule_sample_k3():
    """Test exact solver on sample instance with K=3."""
    inst = sample_instance()
    res = exact_schedule(inst, n_slots=3)

    assert res.is_valid is True
    assert res.assignment == [0, 1, 0, 2, 1]
    assert res.objective == pytest.approx(2.1, abs=1e-3)
    assert res.extra["n_feasible"] == 30
    assert res.extra["n_optimal"] == 1
    assert res.extra["search_space"] == 243
    assert res.extra["feasible"] is True


def test_exact_schedule_sample_k2_infeasible():
    """Test exact solver on sample instance with K=2 reports infeasible."""
    inst = sample_instance()
    res = exact_schedule(inst, n_slots=2)

    assert res.is_valid is False
    assert res.assignment is None
    assert res.extra["feasible"] is False
    assert res.extra["n_feasible"] == 0
    assert res.extra["search_space"] == 32


def test_min_slots_exact():
    """Test minimum slots needed computed via exact solver."""
    assert min_slots_exact(sample_instance()) == 3
    assert min_slots_exact(ring_instance(4)) == 2
    assert min_slots_exact(ring_instance(3)) == 3


def test_exact_vs_greedy_ring_instances():
    """Test exact objective is always <= greedy objective for ring_instance(5) across seeds 0..9."""
    for s in range(10):
        inst = ring_instance(5, seed=s)
        res_exact = exact_schedule(inst, n_slots=3)
        res_greedy = greedy_schedule(inst, n_slots=3)

        assert res_exact.is_valid is True
        assert res_exact.objective is not None
        assert res_greedy.objective is not None
        assert res_exact.objective <= res_greedy.objective + 1e-9


def test_exact_vs_greedy_generated_instances():
    """Test exact objective is always <= greedy objective for generate_instance(6, 40) across seeds 0..4."""
    for s in range(5):
        inst = generate_instance(6, 40, seed=s)
        res_exact = exact_schedule(inst, n_slots=6)
        res_greedy = greedy_schedule(inst, n_slots=6)

        assert res_exact.is_valid is True
        assert res_exact.objective is not None
        assert res_greedy.objective is not None
        assert res_exact.objective <= res_greedy.objective + 1e-9


def test_optimality_gap_metrics():
    """Test optimality gap calculation and edge cases."""
    inst = sample_instance()
    exact_res = exact_schedule(inst, n_slots=3)
    greedy_res = greedy_schedule(inst, n_slots=3)

    gap = optimality_gap(greedy_res.objective, exact_res.objective)
    assert gap == pytest.approx(0.142857, abs=1e-5)

    assert optimality_gap(None, 2.1) is None
    assert optimality_gap(1.0, 0.0) is None
    assert optimality_gap(1.0, None) is None


def test_exact_schedule_max_assignments_error():
    """Test ValueError is raised when search space exceeds max_assignments threshold."""
    inst = sample_instance()
    with pytest.raises(ValueError, match="exceeds max_assignments"):
        exact_schedule(inst, n_slots=3, max_assignments=10)
