import itertools
import time
from typing import Dict, List, Optional
import networkx as nx

from exam_scheduler.conflict_graph import build_conflict_graph, conflict_edges
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


def exact_schedule(
    instance: ExamInstance,
    n_slots: int,
    max_assignments: int = 2_000_000,
) -> SolverResult:
    """Find the optimal schedule via exhaustive enumeration over all assignments."""
    search_space = n_slots ** instance.n_courses
    if search_space > max_assignments:
        raise ValueError(
            f"Search space {search_space} exceeds max_assignments ({max_assignments})."
        )

    t0 = time.perf_counter()

    weights = instance.weights()
    graph = build_conflict_graph(instance)
    edges = conflict_edges(graph)

    best_assignment = None
    best_obj = float("inf")
    n_feasible = 0
    feasible_objs = []

    # Enumerate all assignments in lexicographical product order
    for a in itertools.product(range(n_slots), repeat=instance.n_courses):
        if all(a[u] != a[v] for u, v in edges):
            n_feasible += 1
            obj = sum(weights[c] * a[c] for c in range(instance.n_courses))
            feasible_objs.append(obj)
            # Break ties by first found in product order
            if obj < best_obj - 1e-9:
                best_obj = obj
                best_assignment = list(a)

    runtime_s = time.perf_counter() - t0

    if n_feasible == 0:
        return make_result(
            method="exact",
            instance=instance,
            assignment=None,
            n_slots=n_slots,
            runtime_s=runtime_s,
            extra={
                "n_feasible": 0,
                "n_optimal": 0,
                "search_space": search_space,
                "feasible": False,
            },
        )

    n_optimal = sum(1 for obj in feasible_objs if abs(obj - best_obj) <= 1e-9)
    extra = {
        "n_feasible": n_feasible,
        "n_optimal": n_optimal,
        "search_space": search_space,
        "feasible": True,
    }
    return make_result(
        method="exact",
        instance=instance,
        assignment=best_assignment,
        n_slots=n_slots,
        runtime_s=runtime_s,
        extra=extra,
    )


def min_slots_exact(
    instance: ExamInstance,
    k_max: Optional[int] = None,
    max_assignments: int = 2_000_000,
) -> int:
    """Find the minimum number of slots needed for a valid schedule via exhaustive search."""
    if k_max is None:
        k_max = instance.n_courses

    graph = build_conflict_graph(instance)
    edges = conflict_edges(graph)

    for k in range(1, k_max + 1):
        search_space = k ** instance.n_courses
        if search_space > max_assignments:
            raise ValueError(
                f"Search space {search_space} exceeds max_assignments ({max_assignments})."
            )
        for a in itertools.product(range(k), repeat=instance.n_courses):
            if all(a[u] != a[v] for u, v in edges):
                return k

    raise ValueError(f"No valid schedule exists up to k_max={k_max}.")

