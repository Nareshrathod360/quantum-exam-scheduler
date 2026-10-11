"""Data handling and problem instance generation for exam scheduling."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

DEFAULT_SEED = 42


@dataclass(eq=False, frozen=True)
class ExamInstance:
    """Represents an exam scheduling instance with courses and student enrollments."""

    course_codes: Tuple[str, ...]
    course_titles: Tuple[str, ...]
    enrollments: pd.DataFrame

    def __post_init__(self) -> None:
        """Validate instance integrity."""
        if len(self.course_codes) == 0:
            raise ValueError("At least one course is required.")

        if len(set(self.course_codes)) != len(self.course_codes):
            raise ValueError("Duplicate course codes found.")

        if len(self.course_titles) != len(self.course_codes):
            raise ValueError("Course titles length mismatch.")

        required_cols = {"student_id", "course_code"}
        if not required_cols.issubset(set(self.enrollments.columns)):
            raise ValueError("Enrollments DataFrame missing required columns.")

        valid_codes = set(self.course_codes)
        enrolled_codes = set(self.enrollments["course_code"])
        unknown_codes = enrolled_codes - valid_codes
        if unknown_codes:
            raise ValueError(f"Unknown course codes in enrollments: {unknown_codes}")

        if self.enrollments.duplicated(subset=["student_id", "course_code"]).any():
            raise ValueError("Duplicate student-course enrollment pairs found.")

        missing_courses = valid_codes - enrolled_codes
        if missing_courses:
            raise ValueError(f"Courses with zero enrolled students found: {missing_courses}")

    @property
    def n_courses(self) -> int:
        """Return the number of courses."""
        return len(self.course_codes)

    @property
    def n_students(self) -> int:
        """Return the total number of unique students."""
        return int(self.enrollments["student_id"].nunique())

    def incidence_matrix(self) -> np.ndarray:
        """Compute 0/1 binary incidence matrix (students x courses)."""
        unique_students = sorted(self.enrollments["student_id"].unique())
        student_map = {sid: i for i, sid in enumerate(unique_students)}
        course_map = {code: i for i, code in enumerate(self.course_codes)}

        M = np.zeros((len(unique_students), self.n_courses), dtype=int)
        for _, row in self.enrollments.iterrows():
            s_idx = student_map[row["student_id"]]
            c_idx = course_map[row["course_code"]]
            M[s_idx, c_idx] = 1
        return M

    def enrollment_counts(self) -> np.ndarray:
        """Return 1D numpy array of student counts for each course."""
        M = self.incidence_matrix()
        return M.sum(axis=0)

    def shared_students(self) -> np.ndarray:
        """Compute symmetric matrix of shared student counts with zero diagonal."""
        M = self.incidence_matrix()
        S = M.T @ M
        np.fill_diagonal(S, 0)
        return S

    def weights(self) -> np.ndarray:
        """Compute normalized course enrollment weights (relative to maximum enrollment)."""
        counts = self.enrollment_counts()
        max_c = counts.max()
        if max_c == 0:
            return np.zeros_like(counts, dtype=float)
        return counts / max_c

    def course_table(self) -> pd.DataFrame:
        """Return DataFrame summary table of courses and statistics."""
        return pd.DataFrame({
            "course": list(range(self.n_courses)),
            "code": list(self.course_codes),
            "title": list(self.course_titles),
            "students": self.enrollment_counts(),
            "weight": np.round(self.weights(), 3),
        })


def sample_instance() -> ExamInstance:
    """Create hand-built synthetic sample exam instance forming a 5-cycle conflict graph."""
    course_codes = ("CS101", "MA101", "PH101", "CS201", "EE101")
    course_titles = (
        "Programming Fundamentals",
        "Calculus",
        "Engineering Physics",
        "Data Structures",
        "Basic Electronics",
    )

    rows = []
    student_idx = 1

    # Students taking pairs of adjacent courses in 5-cycle
    pairs = [
        (0, 1, 10),
        (1, 2, 8),
        (2, 3, 12),
        (3, 4, 5),
        (4, 0, 9),
    ]

    for c1, c2, count in pairs:
        for _ in range(count):
            sid = f"S{student_idx:04d}"
            student_idx += 1
            rows.append({"student_id": sid, "course_code": course_codes[c1]})
            rows.append({"student_id": sid, "course_code": course_codes[c2]})

    # Students taking a single course only
    singles = [
        (0, 21),
        (1, 12),
        (2, 30),
        (3, 3),
        (4, 21),
    ]

    for c, count in singles:
        for _ in range(count):
            sid = f"S{student_idx:04d}"
            student_idx += 1
            rows.append({"student_id": sid, "course_code": course_codes[c]})

    enrollments = pd.DataFrame(rows)
    return ExamInstance(course_codes, course_titles, enrollments)


def ring_instance(n_courses: int, seed: int = DEFAULT_SEED) -> ExamInstance:
    """Generate a ring conflict graph topology with n_courses >= 3."""
    if n_courses < 3:
        raise ValueError("Ring instance requires at least 3 courses.")

    rng = np.random.default_rng(seed)
    course_codes = tuple(f"R{i+1:02d}" for i in range(n_courses))
    course_titles = tuple(f"Ring course {i}" for i in range(n_courses))

    rows = []
    student_idx = 1

    # Shared students between adjacent courses in ring
    for i in range(n_courses):
        num_shared = int(rng.integers(4, 13))
        next_i = (i + 1) % n_courses
        for _ in range(num_shared):
            sid = f"S{student_idx:04d}"
            student_idx += 1
            rows.append({"student_id": sid, "course_code": course_codes[i]})
            rows.append({"student_id": sid, "course_code": course_codes[next_i]})

    # Single course students
    for i in range(n_courses):
        num_single = int(rng.integers(2, 25))
        for _ in range(num_single):
            sid = f"S{student_idx:04d}"
            student_idx += 1
            rows.append({"student_id": sid, "course_code": course_codes[i]})

    enrollments = pd.DataFrame(rows)
    return ExamInstance(course_codes, course_titles, enrollments)


def generate_instance(
    n_courses: int,
    n_students: int,
    seed: int = DEFAULT_SEED,
    max_courses_per_student: int = 3,
) -> ExamInstance:
    """Generate a random exam scheduling instance reproducibly."""
    if n_students < n_courses:
        raise ValueError("n_students cannot be less than n_courses.")

    rng = np.random.default_rng(seed)
    course_codes = tuple(f"C{i+1:02d}" for i in range(n_courses))
    course_titles = tuple(f"Course {i}" for i in range(n_courses))

    rows = []
    for s in range(n_students):
        sid = f"S{s+1:04d}"
        primary_course = s % n_courses
        k = int(rng.integers(1, max_courses_per_student + 1))
        extra_count = k - 1

        other_courses = [c for c in range(n_courses) if c != primary_course]
        if extra_count > 0 and other_courses:
            num_extra = min(extra_count, len(other_courses))
            chosen_extra = list(rng.choice(other_courses, size=num_extra, replace=False))
        else:
            chosen_extra = []

        student_courses = [primary_course] + chosen_extra
        for c in student_courses:
            rows.append({"student_id": sid, "course_code": course_codes[c]})

    enrollments = pd.DataFrame(rows)
    return ExamInstance(course_codes, course_titles, enrollments)


def save_instance(
    instance: ExamInstance, directory, prefix: str = "sample"
) -> Tuple[str, str]:
    """Save an ExamInstance to courses and enrollments CSV files."""
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    courses_path = out_dir / f"{prefix}_courses.csv"
    enrollments_path = out_dir / f"{prefix}_enrollments.csv"

    courses_df = pd.DataFrame({
        "course_code": list(instance.course_codes),
        "title": list(instance.course_titles),
    })
    courses_df.to_csv(courses_path, index=False, lineterminator="\n")

    instance.enrollments[["student_id", "course_code"]].to_csv(
        enrollments_path, index=False, lineterminator="\n"
    )

    return str(courses_path), str(enrollments_path)


def load_instance(courses_csv, enrollments_csv) -> ExamInstance:
    """Load an ExamInstance from courses and enrollments CSV files."""
    courses_df = pd.read_csv(courses_csv)
    enrollments_df = pd.read_csv(enrollments_csv)

    codes = tuple(courses_df["course_code"].astype(str))
    titles = tuple(courses_df["title"].astype(str))
    enrollments_df["student_id"] = enrollments_df["student_id"].astype(str)
    enrollments_df["course_code"] = enrollments_df["course_code"].astype(str)

    return ExamInstance(codes, titles, enrollments_df)
