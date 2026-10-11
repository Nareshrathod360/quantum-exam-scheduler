"""Schedule validation and objective evaluation module."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from exam_scheduler.conflict_graph import build_conflict_graph, conflict_edges
from exam_scheduler.data import ExamInstance


@dataclass(frozen=True)
class ValidationResult:
    """Stores the outcome of schedule validation against problem constraints."""

    is_valid: bool
    n_slots: int
    slots_used: int
    conflicts: List[Tuple[int, int, int, int]]  # (course_i, course_j, slot, shared_students)
    n_conflicts: int
    affected_students: int
    unassigned: List[int]
    out_of_range: List[int]
    messages: List[str]


def validate_schedule(
    instance: ExamInstance,
    assignment: List[Optional[int]],
    n_slots: int,
) -> ValidationResult:
    """Validate an exam schedule assignment against student conflicts and slot budgets."""
    if len(assignment) != instance.n_courses:
        raise ValueError(
            f"Assignment length ({len(assignment)}) must equal n_courses ({instance.n_courses})."
        )
    if n_slots < 1:
        raise ValueError(f"n_slots must be >= 1, got {n_slots}.")

    unassigned: List[int] = []
    out_of_range: List[int] = []
    messages: List[str] = []

    # Check for unassigned courses or slots outside valid range 0..n_slots-1
    for c, slot in enumerate(assignment):
        code = instance.course_codes[c]
        if slot is None:
            unassigned.append(c)
            messages.append(f"course {c} ({code}) is unassigned")
        elif slot < 0 or slot >= n_slots:
            out_of_range.append(c)
            messages.append(
                f"course {c} ({code}) has slot {slot} outside 0..{n_slots - 1}"
            )

    # Count distinct slots among assigned courses
    assigned_slots = {s for s in assignment if s is not None}
    slots_used = len(assigned_slots)

    # Detect conflicts between courses that share students
    graph = build_conflict_graph(instance)
    conflicts: List[Tuple[int, int, int, int]] = []
    for i, j in conflict_edges(graph):
        slot_i = assignment[i]
        slot_j = assignment[j]
        if slot_i is not None and slot_j is not None and slot_i == slot_j:
            shared = graph.edges[i, j]["shared"]
            conflicts.append((i, j, slot_i, shared))
            code_i = instance.course_codes[i]
            code_j = instance.course_codes[j]
            messages.append(
                f"conflict between course {i} ({code_i}) and course {j} ({code_j}) "
                f"at slot {slot_i} (shared={shared})"
            )

    n_conflicts = len(conflicts)

    # Compute affected students independently via the incidence matrix
    # A student is affected if two or more of their enrolled courses are in the same slot
    M = instance.incidence_matrix()
    affected_students = 0
    for s_idx in range(M.shape[0]):
        enrolled_courses = np.where(M[s_idx] == 1)[0]
        student_slots = [
            assignment[c] for c in enrolled_courses if assignment[c] is not None
        ]
        if len(student_slots) != len(set(student_slots)):
            affected_students += 1

    # Schedule is valid only if all courses are assigned within range with zero conflicts
    is_valid = (len(unassigned) == 0) and (len(out_of_range) == 0) and (n_conflicts == 0)

    return ValidationResult(
        is_valid=is_valid,
        n_slots=n_slots,
        slots_used=slots_used,
        conflicts=conflicts,
        n_conflicts=n_conflicts,
        affected_students=affected_students,
        unassigned=unassigned,
        out_of_range=out_of_range,
        messages=messages,
    )


def schedule_objective(
    instance: ExamInstance, assignment: List[Optional[int]]
) -> float:
    """Compute schedule objective value as sum(weight_c * slot_c).

    Lower is better; meaningful only for valid schedules.
    """
    if len(assignment) != instance.n_courses:
        raise ValueError(
            f"Assignment length ({len(assignment)}) must equal n_courses ({instance.n_courses})."
        )
    if any(s is None for s in assignment):
        raise ValueError("Cannot compute objective when courses are unassigned.")

    weights = instance.weights()
    total_obj = sum(weights[c] * assignment[c] for c in range(instance.n_courses))
    return float(total_obj)
