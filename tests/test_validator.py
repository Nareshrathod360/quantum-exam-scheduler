"""Tests for schedule validation and objective evaluation."""

import itertools
import numpy as np
import pytest

from exam_scheduler.data import generate_instance, sample_instance
from exam_scheduler.validator import schedule_objective, validate_schedule


def test_sample_valid_schedule():
    """Test valid assignment [0, 1, 0, 2, 1] with K=3 on sample instance."""
    inst = sample_instance()
    assignment = [0, 1, 0, 2, 1]
    res = validate_schedule(inst, assignment, n_slots=3)

    assert res.is_valid is True
    assert res.slots_used == 3
    assert res.n_conflicts == 0
    assert res.affected_students == 0
    assert res.unassigned == []
    assert res.out_of_range == []

    obj = schedule_objective(inst, assignment)
    assert obj == pytest.approx(2.1, abs=1e-3)


def test_all_zero_assignment():
    """Test that all-zero assignment has 5 conflicts and 44 affected students."""
    inst = sample_instance()
    assignment = [0, 0, 0, 0, 0]
    res = validate_schedule(inst, assignment, n_slots=3)

    assert res.is_valid is False
    assert res.n_conflicts == 5
    assert res.affected_students == 44


def test_unassigned_and_out_of_range():
    """Test tracking of unassigned courses and slots equal to or exceeding K."""
    inst = sample_instance()
    # Course 1 is None, Course 3 has slot 3 which equals K (valid range is 0..2)
    assignment = [0, None, 1, 3, 0]
    res = validate_schedule(inst, assignment, n_slots=3)

    assert res.is_valid is False
    assert res.unassigned == [1]
    assert res.out_of_range == [3]
    assert any("course 1" in m and "unassigned" in m for m in res.messages)
    assert any("course 3" in m and "outside 0..2" in m for m in res.messages)


def test_invalid_validator_arguments():
    """Test that invalid argument lengths and non-positive n_slots raise ValueError."""
    inst = sample_instance()
    # Wrong assignment length
    with pytest.raises(ValueError, match="must equal n_courses"):
        validate_schedule(inst, [0, 1, 2], n_slots=3)

    # n_slots < 1
    with pytest.raises(ValueError, match="n_slots must be >= 1"):
        validate_schedule(inst, [0, 1, 0, 1, 2], n_slots=0)

    # Objective with unassigned courses
    with pytest.raises(ValueError, match="unassigned"):
        schedule_objective(inst, [0, 1, None, 0, 1])


def test_exhaustive_sample_instance():
    """Test exhaustive enumeration over all assignments for K=2 and K=3."""
    inst = sample_instance()

    # Over all 2^5 = 32 assignments with K=2, zero are valid
    valid_k2 = []
    for p in itertools.product(range(2), repeat=5):
        res = validate_schedule(inst, list(p), n_slots=2)
        if res.is_valid:
            valid_k2.append(p)
    assert len(valid_k2) == 0

    # Over all 3^5 = 243 assignments with K=3, exactly 30 are valid
    valid_k3 = []
    best_obj = float("inf")
    for p in itertools.product(range(3), repeat=5):
        res = validate_schedule(inst, list(p), n_slots=3)
        if res.is_valid:
            valid_k3.append(p)
            obj = schedule_objective(inst, list(p))
            if obj < best_obj:
                best_obj = obj

    assert len(valid_k3) == 30
    assert best_obj == pytest.approx(2.1, abs=1e-3)


def test_affected_students_matches_independent_loop():
    """Test affected_students calculation matches independent student-by-student loop."""
    inst = generate_instance(8, 40, seed=3)
    rng = np.random.default_rng(0)
    random_assignment = [int(rng.integers(0, 4)) for _ in range(inst.n_courses)]

    res = validate_schedule(inst, random_assignment, n_slots=4)

    # Compute affected students independently by looping through enrollments DataFrame
    code_to_idx = {code: i for i, code in enumerate(inst.course_codes)}
    grouped = inst.enrollments.groupby("student_id")["course_code"].apply(list)
    ground_truth_affected = 0
    for _, courses in grouped.items():
        slots = [
            random_assignment[code_to_idx[c]]
            for c in courses
            if random_assignment[code_to_idx[c]] is not None
        ]
        if len(slots) != len(set(slots)):
            ground_truth_affected += 1

    assert res.affected_students == ground_truth_affected
