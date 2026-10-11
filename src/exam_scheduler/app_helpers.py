"""Pure helper functions for the Streamlit dashboard application.

Contains functions for instance generation, qubit memory budgeting,
result tabulation, timetable generation, bitstring measurement decoding,
and benchmark result loading. None of these functions depend on Streamlit.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import numpy as np
import pandas as pd

from exam_scheduler.data import (
    DEFAULT_SEED,
    ExamInstance,
    generate_instance,
    ring_instance,
    sample_instance,
)
from exam_scheduler.qubo import bitstring_to_bits, build_qubo, decode_bits
from exam_scheduler.results import SolverResult
from exam_scheduler.validator import validate_schedule


def build_instance(kind: str, **params: Any) -> ExamInstance:
    """Build an ExamInstance based on kind and configuration parameters.

    Args:
        kind: One of "sample", "ring", "random".
        **params: Parameters for the instance generator:
            - ring: n_courses (default 5), seed (default DEFAULT_SEED)
            - random: n_courses (default 5), n_students (default 60), seed (default DEFAULT_SEED)

    Returns:
        Constructed ExamInstance.

    Raises:
        ValueError: If kind is unknown.
    """
    if kind == "sample":
        return sample_instance()
    elif kind == "ring":
        n_courses = int(params.get("n_courses", 5))
        seed = int(params.get("seed", DEFAULT_SEED))
        return ring_instance(n_courses=n_courses, seed=seed)
    elif kind == "random":
        n_courses = int(params.get("n_courses", 5))
        n_students = int(params.get("n_students", 60))
        seed = int(params.get("seed", DEFAULT_SEED))
        return generate_instance(n_courses=n_courses, n_students=n_students, seed=seed)
    else:
        raise ValueError(
            f"Unknown instance kind: '{kind}'. Expected 'sample', 'ring', or 'random'."
        )


def qubit_budget(instance: ExamInstance, n_slots: int) -> Dict[str, Any]:
    """Calculate the qubit memory and solver capability budget for an instance.

    Args:
        instance: The ExamInstance problem.
        n_slots: Number of available exam slots (K).

    Returns:
        Dictionary containing:
            - n_qubits: Total binary variables/qubits N = n * K.
            - state_bytes: Full 2^N statevector size in bytes (complex128 = 16 bytes/amplitude).
            - qaoa_allowed: True if N <= 15 (fast noiseless simulation).
            - qaoa_slow: True if 15 < N <= 18 (simulation is slow but feasible with warning).
            - qaoa_blocked: True if N > 18 (statevector simulation exceeds practical limits).
            - exact_allowed: True if total search space K^n <= 2,000,000 assignments.
    """
    n_qubits = instance.n_courses * n_slots
    state_bytes = (1 << n_qubits) * 16
    qaoa_allowed = bool(n_qubits <= 15)
    qaoa_slow = bool(15 < n_qubits <= 18)
    qaoa_blocked = bool(n_qubits > 18)
    exact_allowed = bool((n_slots ** instance.n_courses) <= 2_000_000)

    return {
        "n_qubits": n_qubits,
        "state_bytes": state_bytes,
        "qaoa_allowed": qaoa_allowed,
        "qaoa_slow": qaoa_slow,
        "qaoa_blocked": qaoa_blocked,
        "exact_allowed": exact_allowed,
    }


def runs_to_dataframe(
    runs: Dict[str, SolverResult], optimum: Optional[float] = None
) -> pd.DataFrame:
    """Convert a dictionary of solver results to a comparative DataFrame.

    Args:
        runs: Dict mapping method display names to SolverResult instances.
        optimum: Exact optimal objective value, if available and known.

    Returns:
        DataFrame with columns: method, valid, slots_used, conflicts,
        affected_students, objective, optimality_gap_pct, runtime_s, note.
    """
    rows: List[Dict[str, Any]] = []

    for name, res in runs.items():
        # Valid string MUST be taken strictly from res.is_valid
        valid_str = "valid" if res.is_valid else "INVALID"

        # Optimality gap is only computed when schedule is valid and optimum is known and nonzero
        if res.is_valid and optimum is not None and optimum != 0 and res.objective is not None:
            gap_pct = float((res.objective - optimum) / optimum * 100.0)
        else:
            gap_pct = np.nan

        # Extract note: first validator message or diagnostic info
        note = ""
        if res.extra and "messages" in res.extra and res.extra["messages"]:
            note = str(res.extra["messages"][0])
        elif not res.is_valid:
            if res.extra and "shots" in res.extra:
                shots = res.extra["shots"]
                n_valid = res.extra.get("n_valid_shots", 0)
                if n_valid == 0:
                    note = f"no valid shot among {shots} shots"
                else:
                    note = "invalid schedule"
            else:
                note = "invalid schedule"

        rows.append({
            "method": name if name else (res.method if res.method else ""),
            "valid": valid_str,
            "slots_used": res.slots_used,
            "conflicts": res.n_conflicts,
            "affected_students": res.affected_students,
            "objective": res.objective,
            "optimality_gap_pct": gap_pct,
            "runtime_s": res.runtime_s,
            "note": note,
        })

    cols = [
        "method",
        "valid",
        "slots_used",
        "conflicts",
        "affected_students",
        "objective",
        "optimality_gap_pct",
        "runtime_s",
        "note",
    ]
    return pd.DataFrame(rows, columns=cols)


def timetable_dataframe(
    instance: ExamInstance, assignment: Union[List[Optional[int]], Tuple[Optional[int], ...]], n_slots: int
) -> pd.DataFrame:
    """Format an assignment into an exam timetable DataFrame by slot.

    Args:
        instance: ExamInstance problem.
        assignment: List of slot numbers (0..n_slots-1) for each course.
        n_slots: Total number of slots.

    Returns:
        DataFrame with columns: slot, courses, students.

    Raises:
        ValueError: If assignment contains any None values.
    """
    if assignment is None or any(s is None for s in assignment):
        raise ValueError("Assignment cannot contain None values for timetable display.")

    enr_counts = instance.enrollment_counts()
    rows: List[Dict[str, Any]] = []

    for t in range(n_slots):
        slot_label = "Slot 0 (earliest)" if t == 0 else f"Slot {t}"
        courses_in_slot = [
            instance.course_codes[c]
            for c in range(instance.n_courses)
            if assignment[c] == t
        ]
        courses_str = ", ".join(courses_in_slot)
        student_total = int(
            sum(enr_counts[c] for c in range(instance.n_courses) if assignment[c] == t)
        )

        rows.append({
            "slot": slot_label,
            "courses": courses_str,
            "students": student_total,
        })

    return pd.DataFrame(rows, columns=["slot", "courses", "students"])


def top_counts_dataframe(
    instance: ExamInstance, n_slots: int, counts: Dict[str, int], top: int = 15
) -> pd.DataFrame:
    """Decode and summarize top measured bitstrings from quantum or sampling shots.

    Args:
        instance: ExamInstance problem.
        n_slots: Number of available slots.
        counts: Dictionary of bitstring to frequency count.
        top: Number of top bitstrings to return.

    Returns:
        DataFrame with columns: bitstring, count, share_pct, valid, schedule.
    """
    model = build_qubo(instance, n_slots)
    total_shots = sum(counts.values()) if counts else 1

    sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top]
    rows: List[Dict[str, Any]] = []

    for bitstring, count in sorted_counts:
        share_pct = (count / total_shots) * 100.0
        # Qiskit measured bitstring is little-endian: rightmost char = qubit 0
        bits = bitstring_to_bits(bitstring)
        decoded = decode_bits(model, bits)

        if any(s is None for s in decoded):
            is_valid = False
            schedule_str = "not one-hot"
        else:
            val = validate_schedule(instance, decoded, n_slots)
            is_valid = bool(val.is_valid)
            schedule_str = " ".join(str(s) for s in decoded)

        rows.append({
            "bitstring": bitstring,
            "count": count,
            "share_pct": share_pct,
            "valid": is_valid,
            "schedule": schedule_str,
        })

    cols = ["bitstring", "count", "share_pct", "valid", "schedule"]
    return pd.DataFrame(rows, columns=cols)


def load_benchmark_results(results_dir: Union[str, Path]) -> Dict[str, Any]:
    """Load precomputed benchmark CSV runs, summary, figures, and metadata.

    Args:
        results_dir: Directory containing benchmark output files.

    Returns:
        Dictionary with keys:
            - runs: DataFrame or None
            - summary: DataFrame or None
            - figures: Dict[str, str] mapping figure name to absolute file path
            - meta: Dict or None
    """
    res_path = Path(results_dir)

    runs_csv = res_path / "benchmark_runs.csv"
    summary_csv = res_path / "benchmark_summary.csv"
    meta_json = res_path / "benchmark_meta.json"

    runs = pd.read_csv(runs_csv) if runs_csv.is_file() else None
    summary = pd.read_csv(summary_csv) if summary_csv.is_file() else None

    figures: Dict[str, str] = {}
    figure_names = [
        ("feasibility", "fig_feasibility.png"),
        ("objective", "fig_objective.png"),
        ("penalty", "fig_penalty.png"),
        ("scaling", "fig_scaling.png"),
    ]
    for key, filename in figure_names:
        fig_path = res_path / filename
        if fig_path.is_file():
            figures[key] = str(fig_path)

    meta = None
    if meta_json.is_file():
        try:
            meta = json.loads(meta_json.read_text(encoding="utf-8"))
        except Exception:
            meta = None

    return {
        "runs": runs,
        "summary": summary,
        "figures": figures,
        "meta": meta,
    }
