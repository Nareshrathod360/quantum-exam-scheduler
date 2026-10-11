"""Tests for classical greedy coloring scheduler and results module."""

import inspect
import pytest

from exam_scheduler.classical import greedy_schedule
from exam_scheduler.conflict_graph import build_conflict_graph
from exam_scheduler.data import (
    generate_instance,
    ring_instance,
    sample_instance,
)
from exam_scheduler.results import make_result


def test_greedy_sample_k3_largest_first():
    """Test largest_first on sample instance with K=3."""
    inst = sample_instance()
    res = greedy_schedule(inst, n_slots=3, strategy="largest_first")

    assert res.is_valid is True
    assert res.assignment == [0, 1, 0, 1, 2]
    assert res.slots_used == 3
    assert res.n_conflicts == 0
    assert res.affected_students == 0
    assert res.objective == pytest.approx(2.4, abs=1e-3)
    assert res.method == "greedy-largest_first"


def test_greedy_sample_k2_invalid():
    """Test that K=2 on sample instance gives an invalid schedule needing 3 slots."""
    inst = sample_instance()
    res = greedy_schedule(inst, n_slots=2, strategy="largest_first")

    assert res.is_valid is False
    assert res.slots_used == 3
    # Course 4 has slot 2, which is outside 0..1
    messages = res.extra.get("messages", [])
    assert any("outside 0..1" in m for m in messages)


def test_greedy_various_seeds_degree_bound():
    """Test greedy validity and degree bound across seeds 0..29."""
    for s in range(30):
        inst = generate_instance(8, 40, seed=s)
        res = greedy_schedule(inst, n_slots=8, strategy="largest_first")
        assert res.is_valid is True

        G = build_conflict_graph(inst)
        max_deg = max((d for _, d in G.degree()), default=0)
        assert res.slots_used <= max_deg + 1


def test_greedy_strategies_ring5():
    """Test all three strategies produce valid schedules for ring_instance(5) with K=3."""
    inst = ring_instance(5, seed=42)
    for strat in ["largest_first", "DSATUR", "smallest_last"]:
        res = greedy_schedule(inst, n_slots=3, strategy=strat)
        assert res.is_valid is True
        assert res.slots_used <= 3
        assert res.method == f"greedy-{strat}"


def test_greedy_deterministic():
    """Test that repeated calls produce identical assignments."""
    inst = generate_instance(8, 40, seed=12)
    res1 = greedy_schedule(inst, n_slots=8, strategy="largest_first")
    res2 = greedy_schedule(inst, n_slots=8, strategy="largest_first")
    assert res1.assignment == res2.assignment
    assert res1.slots_used == res2.slots_used


def test_class_weights_non_increasing_after_relabeling():
    """Test that color class total weights are non-increasing from slot 0 upwards."""
    inst = generate_instance(8, 40, seed=42)
    res = greedy_schedule(inst, n_slots=8, strategy="largest_first")
    weights = inst.weights()

    slot_weights = {}
    for c, slot in enumerate(res.assignment):
        slot_weights[slot] = slot_weights.get(slot, 0.0) + weights[c]

    totals = [slot_weights[s] for s in range(res.slots_used)]
    for i in range(len(totals) - 1):
        assert totals[i] >= totals[i + 1] - 1e-9


def test_make_result_signature_and_none_assignment():
    """Test that make_result has no is_valid parameter and assignment=None is invalid."""
    sig = inspect.signature(make_result)
    assert "is_valid" not in sig.parameters

    inst = sample_instance()
    res = make_result(
        method="test-method",
        instance=inst,
        assignment=None,
        n_slots=3,
        runtime_s=0.001,
    )
    assert res.is_valid is False
    assert res.assignment is None
    assert res.objective is None
    assert "reason" in res.extra
