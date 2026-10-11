"""Standardized solver result structures and validation wrappers."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from exam_scheduler.data import ExamInstance
from exam_scheduler.validator import schedule_objective, validate_schedule


@dataclass(frozen=True)
class SolverResult:
    """Standardized representation of a schedule solver execution result."""

    method: str
    assignment: Optional[List[Optional[int]]]
    n_slots: int
    objective: Optional[float]
    slots_used: int
    n_conflicts: int
    affected_students: int
    is_valid: bool
    runtime_s: float
    extra: Dict[str, Any]


def make_result(
    method: str,
    instance: ExamInstance,
    assignment: Optional[List[Optional[int]]],
    n_slots: int,
    runtime_s: float,
    extra: Optional[Dict[str, Any]] = None,
) -> SolverResult:
    """Create a validated SolverResult; validity is always computed by validate_schedule."""
    extra_dict = dict(extra) if extra is not None else {}

    if assignment is None:
        # Evaluate missing schedule through validator to register all courses as unassigned
        dummy_assignment: List[Optional[int]] = [None] * instance.n_courses
        val = validate_schedule(instance, dummy_assignment, n_slots)
        extra_dict.setdefault("reason", "No assignment provided")
        extra_dict.setdefault("messages", val.messages)
        extra_dict.setdefault("unassigned", val.unassigned)
        return SolverResult(
            method=method,
            assignment=None,
            n_slots=n_slots,
            objective=None,
            slots_used=val.slots_used,
            n_conflicts=val.n_conflicts,
            affected_students=val.affected_students,
            is_valid=val.is_valid,
            runtime_s=runtime_s,
            extra=extra_dict,
        )

    val = validate_schedule(instance, assignment, n_slots)
    extra_dict.setdefault("messages", val.messages)
    extra_dict.setdefault("conflicts", val.conflicts)
    extra_dict.setdefault("unassigned", val.unassigned)
    extra_dict.setdefault("out_of_range", val.out_of_range)

    # Objective is defined only when every course has been assigned a slot
    if any(s is None for s in assignment):
        objective = None
    else:
        objective = schedule_objective(instance, assignment)

    return SolverResult(
        method=method,
        assignment=assignment,
        n_slots=n_slots,
        objective=objective,
        slots_used=val.slots_used,
        n_conflicts=val.n_conflicts,
        affected_students=val.affected_students,
        is_valid=val.is_valid,
        runtime_s=runtime_s,
        extra=extra_dict,
    )


def optimality_gap(
    value: Optional[float], optimum: Optional[float]
) -> Optional[float]:
    """Compute relative optimality gap: (value - optimum) / optimum.

    Returns None if value is None, optimum is None, or optimum == 0.
    """
    if value is None or optimum is None or optimum == 0:
        return None
    return float((value - optimum) / optimum)

