"""Unit tests for app_helpers module."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from exam_scheduler.app_helpers import (
    build_instance,
    load_benchmark_results,
    qubit_budget,
    runs_to_dataframe,
    timetable_dataframe,
    top_counts_dataframe,
)
from exam_scheduler.classical import exact_schedule, greedy_schedule
from exam_scheduler.data import ExamInstance, sample_instance


def test_build_instance():
    """Verify build_instance constructs valid instances or raises ValueError."""
    inst_sample = build_instance("sample")
    assert isinstance(inst_sample, ExamInstance)
    assert inst_sample.n_courses == 5

    inst_ring = build_instance("ring", n_courses=4, seed=42)
    assert isinstance(inst_ring, ExamInstance)
    assert inst_ring.n_courses == 4

    inst_rand = build_instance("random", n_courses=3, n_students=30, seed=42)
    assert isinstance(inst_rand, ExamInstance)
    assert inst_rand.n_courses == 3
    assert inst_rand.n_students <= 30

    with pytest.raises(ValueError, match="Unknown instance kind"):
        build_instance("nonexistent_kind")


def test_qubit_budget():
    """Verify qubit memory and feasibility budget calculation."""
    inst = sample_instance()  # 5 courses

    # K=3: N = 5 * 3 = 15 -> state_bytes = 2^15 * 16 = 524,288
    b3 = qubit_budget(inst, n_slots=3)
    assert b3["n_qubits"] == 15
    assert b3["qaoa_allowed"] is True
    assert b3["qaoa_slow"] is False
    assert b3["qaoa_blocked"] is False
    assert b3["state_bytes"] == 524288
    assert b3["exact_allowed"] is True  # 3^5 = 243 <= 2,000,000

    # K=4: N = 5 * 4 = 20 -> qaoa_blocked is True
    b4 = qubit_budget(inst, n_slots=4)
    assert b4["n_qubits"] == 20
    assert b4["qaoa_allowed"] is False
    assert b4["qaoa_blocked"] is True


def test_runs_to_dataframe():
    """Verify runs_to_dataframe tabulates valid/invalid runs and optimality gap."""
    inst = sample_instance()
    res_exact = exact_schedule(inst, n_slots=3)
    res_greedy = greedy_schedule(inst, n_slots=3, strategy="largest_first")
    optimum = res_exact.objective

    runs = {
        "exact": res_exact,
        "greedy-largest_first": res_greedy,
    }
    df = runs_to_dataframe(runs, optimum=optimum)

    assert len(df) == 2
    assert "valid" in df.columns
    assert "optimality_gap_pct" in df.columns

    # Both are valid for K=3
    exact_row = df[df["method"] == "exact"].iloc[0]
    greedy_row = df[df["method"] == "greedy-largest_first"].iloc[0]

    assert exact_row["valid"] == "valid"
    assert greedy_row["valid"] == "valid"
    assert exact_row["optimality_gap_pct"] == pytest.approx(0.0, abs=1e-5)
    assert greedy_row["optimality_gap_pct"] == pytest.approx(14.2857, abs=0.1)

    # For K=2, 5 courses with clique size 3 cannot be scheduled without conflicts
    res_greedy_k2 = greedy_schedule(inst, n_slots=2, strategy="largest_first")
    df_k2 = runs_to_dataframe({"greedy-k2": res_greedy_k2}, optimum=None)
    k2_row = df_k2.iloc[0]

    assert k2_row["valid"] == "INVALID"
    assert pd.isna(k2_row["optimality_gap_pct"])


def test_timetable_dataframe():
    """Verify timetable formatting, slot student aggregates, and None guard."""
    inst = sample_instance()
    assignment = [0, 1, 0, 2, 1]
    tt = timetable_dataframe(inst, assignment, n_slots=3)

    assert len(tt) == 3
    assert list(tt.columns) == ["slot", "courses", "students"]

    # Slot 0 has CS101 (50) and MATH201 (40) -> 90 students
    s0 = tt.loc[tt["slot"] == "Slot 0 (earliest)"].iloc[0]
    assert s0["students"] == 90

    # Slot 1 has CS102 (35) and ENGL101 (30) -> 65 students
    s1 = tt.loc[tt["slot"] == "Slot 1"].iloc[0]
    assert s1["students"] == 65

    # Slot 2 has PHYS101 (20) -> 20 students
    s2 = tt.loc[tt["slot"] == "Slot 2"].iloc[0]
    assert s2["students"] == 20

    # Unassigned course None must raise ValueError
    with pytest.raises(ValueError, match="Assignment cannot contain None"):
        timetable_dataframe(inst, [0, None, 0, 2, 1], n_slots=3)


def test_top_counts_dataframe():
    """Verify top measured bitstring decoding and validation status."""
    inst = sample_instance()
    n_slots = 3

    # Assignment [0, 1, 0, 2, 1] has active qubits at indices:
    # course 0 slot 0: 0
    # course 1 slot 1: 4
    # course 2 slot 0: 6
    # course 3 slot 2: 11
    # course 4 slot 1: 13
    active_qubits = {0, 4, 6, 11, 13}
    bits = [1 if q in active_qubits else 0 for q in range(15)]
    # Little-endian: rightmost character in measured bitstring is qubit 0
    valid_bitstring = "".join(str(b) for b in reversed(bits))
    invalid_bitstring = "0" * 15

    counts = {
        valid_bitstring: 80,
        invalid_bitstring: 20,
    }

    df = top_counts_dataframe(inst, n_slots, counts, top=5)
    assert len(df) == 2

    row_valid = df[df["bitstring"] == valid_bitstring].iloc[0]
    assert bool(row_valid["valid"]) is True
    assert row_valid["schedule"] == "0 1 0 2 1"
    assert row_valid["share_pct"] == pytest.approx(80.0)

    row_invalid = df[df["bitstring"] == invalid_bitstring].iloc[0]
    assert bool(row_invalid["valid"]) is False
    assert row_invalid["schedule"] == "not one-hot"
    assert row_invalid["share_pct"] == pytest.approx(20.0)


def test_load_benchmark_results(tmp_path):
    """Verify loading benchmark artifacts returns None/empty when folder is empty."""
    empty_dir = tmp_path / "empty_results"
    empty_dir.mkdir()

    res = load_benchmark_results(empty_dir)
    assert res["runs"] is None
    assert res["summary"] is None
    assert res["figures"] == {}
    assert res["meta"] is None
