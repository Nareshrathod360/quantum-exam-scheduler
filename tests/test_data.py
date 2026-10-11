"""Tests for data handling and instance generation."""

import numpy as np
import pandas as pd
import pytest

from exam_scheduler.data import (
    ExamInstance,
    generate_instance,
    load_instance,
    ring_instance,
    sample_instance,
    save_instance,
)


def test_sample_instance_properties():
    """Test sizes, enrollment counts, and weights for sample instance."""
    inst = sample_instance()
    assert inst.n_courses == 5
    assert inst.n_students == 131
    assert len(inst.enrollments) == 175

    expected_counts = [40, 30, 50, 20, 35]
    assert list(inst.enrollment_counts()) == expected_counts

    expected_weights = [0.8, 0.6, 1.0, 0.4, 0.7]
    np.testing.assert_allclose(np.round(inst.weights(), 1), expected_weights)


def test_shared_students_matrix():
    """Test symmetry, zero diagonal, and ring values of shared_students."""
    inst = sample_instance()
    S = inst.shared_students()

    # Symmetric matrix
    assert (S == S.T).all()

    # Zero diagonal
    assert (np.diag(S) == 0).all()

    # Ring values for 5-cycle
    ring_vals = [S[0, 1], S[1, 2], S[2, 3], S[3, 4], S[4, 0]]
    assert ring_vals == [10, 8, 12, 5, 9]


def test_csv_round_trip(tmp_path):
    """Test saving and loading ExamInstance from CSV files."""
    inst = sample_instance()
    c_path, e_path = save_instance(inst, tmp_path, prefix="test")

    loaded = load_instance(c_path, e_path)
    assert loaded.course_codes == inst.course_codes
    assert loaded.course_titles == inst.course_titles
    assert len(loaded.enrollments) == len(inst.enrollments)
    assert (loaded.enrollment_counts() == inst.enrollment_counts()).all()


def test_generate_instance_reproducibility():
    """Test seed reproducibility and differences across seeds."""
    inst1 = generate_instance(7, 50, seed=42)
    inst2 = generate_instance(7, 50, seed=42)
    inst3 = generate_instance(7, 50, seed=99)

    pd.testing.assert_frame_equal(inst1.enrollments, inst2.enrollments)
    assert not inst1.enrollments.equals(inst3.enrollments)

    # Check every course has students
    assert (inst1.enrollment_counts() > 0).all()


@pytest.mark.parametrize(
    "codes,titles,df,err_msg",
    [
        ((), (), pd.DataFrame({"student_id": ["S1"], "course_code": ["C1"]}), "At least one course"),
        (("C01", "C01"), ("T1", "T2"), pd.DataFrame({"student_id": ["S1", "S1"], "course_code": ["C01", "C01"]}), "Duplicate course codes"),
        (("C01", "C02"), ("T1",), pd.DataFrame({"student_id": ["S1", "S2"], "course_code": ["C01", "C02"]}), "length mismatch"),
        (("C01",), ("T1",), pd.DataFrame({"wrong_col": ["S1"], "course_code": ["C01"]}), "missing required columns"),
        (("C01",), ("T1",), pd.DataFrame({"student_id": ["S1"], "course_code": ["C99"]}), "Unknown course codes"),
        (("C01",), ("T1",), pd.DataFrame({"student_id": ["S1", "S1"], "course_code": ["C01", "C01"]}), "Duplicate student-course"),
        (("C01", "C02"), ("T1", "T2"), pd.DataFrame({"student_id": ["S1"], "course_code": ["C01"]}), "zero enrolled students"),
    ],
)
def test_invalid_exam_instance(codes, titles, df, err_msg):
    """Test validation errors in ExamInstance __post_init__."""
    with pytest.raises(ValueError, match=err_msg):
        ExamInstance(codes, titles, df)


def test_invalid_instance_generators():
    """Test parameter validation in ring_instance and generate_instance."""
    with pytest.raises(ValueError, match="at least 3 courses"):
        ring_instance(2)

    with pytest.raises(ValueError, match="n_students cannot be less"):
        generate_instance(n_courses=5, n_students=3)
