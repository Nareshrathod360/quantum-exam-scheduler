"""Classical graph coloring algorithms for exam scheduling."""

import time
from typing import Dict, List
import networkx as nx

from exam_scheduler.conflict_graph import build_conflict_graph
from exam_scheduler.data import ExamInstance
from exam_scheduler.results import SolverResult, make_result

ALLOWED_STRATEGIES = {"largest_first", "DSATUR", "smallest_last"}


def greedy_schedule(
    instance: ExamInstance, n_slots: int, strategy: str = "largest_first"
) -> SolverResult:
    """Solve exam scheduling using NetworkX greedy graph coloring with weight-optimal slot relabeling."""
    if strategy not in ALLOWED_STRATEGIES:
        raise ValueError(
            f"Strategy '{strategy}' not supported. Allowed: {sorted(ALLOWED_STRATEGIES)}"
        )

    t0 = time.perf_counter()

    graph = build_conflict_graph(instance)
    raw_coloring: Dict[int, int] = nx.greedy_color(graph, strategy=strategy)

    # Group courses by their assigned raw color class
    color_classes: Dict[int, List[int]] = {}
    for c in range(instance.n_courses):
        col = raw_coloring[c]
        color_classes.setdefault(col, []).append(c)

    weights = instance.weights()

    # Relabel the color classes by descending total course weight.
    # Relabeling permutes the color classes without altering the partition,
    # which preserves a proper (conflict-free) coloring. Assigning the class
    # with the largest total course weight to slot 0, the next largest to slot 1,
    # etc., provably minimizes the objective sum(weight_c * slot_c) for this partition.
    # Ties are broken deterministically by the smallest course number in the class.
    sorted_classes = sorted(
        color_classes.values(),
        key=lambda members: (-sum(weights[c] for c in members), min(members)),
    )

    # Assign new slots: 0 for heaviest class, 1 for second heaviest, etc.
    # Note: We do not clamp slots to n_slots so exceeding the slot budget is caught by the validator.
    assignment: List[int] = [0] * instance.n_courses
    for slot_idx, members in enumerate(sorted_classes):
        for c in members:
            assignment[c] = slot_idx

    runtime_s = time.perf_counter() - t0

    method_name = f"greedy-{strategy}"
    return make_result(
        method=method_name,
        instance=instance,
        assignment=assignment,
        n_slots=n_slots,
        runtime_s=runtime_s,
    )
