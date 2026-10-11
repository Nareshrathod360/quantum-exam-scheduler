"""CLI interface for Exam Scheduler."""

import argparse
from pathlib import Path
import sys

# Ensure src directory is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exam_scheduler.classical import greedy_schedule
from exam_scheduler.conflict_graph import (
    build_conflict_graph,
    clique_lower_bound,
    conflict_edges,
    draw_conflict_graph,
)
from exam_scheduler.data import load_instance, sample_instance, save_instance


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Quantum Exam Scheduler CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    make_cmd = subparsers.add_parser(
        "make-sample-data", help="Generate sample dataset CSVs"
    )
    make_cmd.add_argument(
        "--out",
        type=str,
        default=str(REPO_ROOT / "data"),
        help="Output directory",
    )

    graph_cmd = subparsers.add_parser(
        "show-graph", help="Display conflict graph and metrics"
    )
    graph_cmd.add_argument(
        "--from-csv",
        action="store_true",
        help="Load instance from CSV files in data folder",
    )
    graph_cmd.add_argument(
        "--save", type=str, default=None, help="Path to save graph PNG figure"
    )

    greedy_cmd = subparsers.add_parser(
        "greedy", help="Solve exam scheduling using classical greedy heuristic"
    )
    greedy_cmd.add_argument(
        "--slots",
        type=int,
        default=3,
        help="Number of available slots budget",
    )
    greedy_cmd.add_argument(
        "--strategy",
        type=str,
        default="largest_first",
        help="Greedy coloring strategy",
    )
    greedy_cmd.add_argument(
        "--from-csv",
        action="store_true",
        help="Load instance from CSV files in data folder",
    )

    args = parser.parse_args()

    if args.command == "make-sample-data":
        inst = sample_instance()
        courses_path, enrollments_path = save_instance(inst, args.out, prefix="sample")
        print(courses_path)
        print(enrollments_path)

    elif args.command == "show-graph":
        if args.from_csv:
            data_dir = REPO_ROOT / "data"
            courses_csv = data_dir / "sample_courses.csv"
            enrollments_csv = data_dir / "sample_enrollments.csv"
            inst = load_instance(courses_csv, enrollments_csv)
        else:
            inst = sample_instance()

        graph = build_conflict_graph(inst)
        edges = conflict_edges(graph)
        bound = clique_lower_bound(graph)

        print(
            f"courses: {inst.n_courses} | students: {inst.n_students} | enrolments: {len(inst.enrollments)}"
        )
        print(inst.course_table().to_string(index=False))
        print("conflict edges (course_i, course_j, shared students):")
        for u, v in edges:
            code_u = graph.nodes[u]["code"]
            code_v = graph.nodes[v]["code"]
            shared = graph.edges[u, v]["shared"]
            print(f"  ({u}, {v})  {code_u}-{code_v}  shared={shared}")
        print(f"clique lower bound on slots needed: {bound}")

        if args.save:
            save_path = Path(args.save)
            if not save_path.is_absolute():
                save_path = REPO_ROOT / save_path
            save_path.parent.mkdir(parents=True, exist_ok=True)

            fig, ax = plt.subplots(figsize=(6, 6))
            draw_conflict_graph(graph, ax=ax, title="Course Conflict Graph")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"saved figure to {save_path}")

    elif args.command == "greedy":
        if args.from_csv:
            data_dir = REPO_ROOT / "data"
            courses_csv = data_dir / "sample_courses.csv"
            enrollments_csv = data_dir / "sample_enrollments.csv"
            inst = load_instance(courses_csv, enrollments_csv)
        else:
            inst = sample_instance()

        result = greedy_schedule(inst, n_slots=args.slots, strategy=args.strategy)

        print(f"method: {result.method} | slots budget: {result.n_slots}")
        for c in range(inst.n_courses):
            code = inst.course_codes[c]
            slot_val = result.assignment[c] if result.assignment is not None else "None"
            print(f"{code} -> slot {slot_val}")
        print(
            f"slots used: {result.slots_used} | conflicts: {result.n_conflicts} | "
            f"affected students: {result.affected_students} | valid: {result.is_valid}"
        )
        if result.objective is not None:
            obj_str = f"{result.objective:.4f}"
        else:
            obj_str = "n/a"
        print(f"objective: {obj_str} | runtime: {result.runtime_s:.4f} s")

        if not result.is_valid:
            messages = result.extra.get("messages", [])
            print(f"problems: {'; '.join(messages)}")


if __name__ == "__main__":
    main()
