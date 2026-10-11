"""Tests for ExamInstance caching and validator optimizations."""

import pytest

from exam_scheduler.data import sample_instance
from exam_scheduler.validator import validate_schedule


def test_cached_arrays_read_only():
    """Test that all cached arrays in ExamInstance have writeable flag set to False."""
    inst = sample_instance()

    m = inst.incidence_matrix()
    assert not m.flags.writeable
    with pytest.raises(ValueError):
        m[0, 0] = 99

    s = inst.shared_students()
    assert not s.flags.writeable
    with pytest.raises(ValueError):
        s[0, 0] = 99

    c = inst.enrollment_counts()
    assert not c.flags.writeable
    with pytest.raises(ValueError):
        c[0] = 99

    w = inst.weights()
    assert not w.flags.writeable
    with pytest.raises(ValueError):
        w[0] = 99


def test_cached_object_identity():
    """Test that repeated calls to cached methods return the exact same object reference."""
    inst = sample_instance()
    m1 = inst.incidence_matrix()
    m2 = inst.incidence_matrix()
    assert m1 is m2

    s1 = inst.shared_students()
    s2 = inst.shared_students()
    assert s1 is s2

    c1 = inst.enrollment_counts()
    c2 = inst.enrollment_counts()
    assert c1 is c2

    w1 = inst.weights()
    w2 = inst.weights()
    assert w1 is w2


def test_validate_schedule_identical_result():
    """Test that validate_schedule yields expected ValidationResult for optimum schedule."""
    inst = sample_instance()
    optimum_assignment = [0, 1, 0, 2, 1]
    res = validate_schedule(inst, optimum_assignment, n_slots=3)

    assert res.is_valid is True
    assert res.slots_used == 3
    assert res.n_conflicts == 0
    assert res.affected_students == 0
    assert res.unassigned == []
    assert res.out_of_range == []
