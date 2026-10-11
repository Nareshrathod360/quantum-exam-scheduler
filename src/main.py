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

import itertools
import numpy as np

from exam_scheduler.classical import exact_schedule, greedy_schedule, min_slots_exact
from exam_scheduler.conflict_graph import (
    build_conflict_graph,
    clique_lower_bound,
    conflict_edges,
    draw_conflict_graph,
)
from exam_scheduler.data import load_instance, sample_instance, save_instance
from exam_scheduler.ising import ising_energies_all, qubo_to_ising
from exam_scheduler.qubo import build_qubo, decode_bits, qubo_energies_all
from exam_scheduler.validator import validate_schedule


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

    exact_cmd = subparsers.add_parser(
        "exact", help="Solve exam scheduling using exact brute-force search"
    )
    exact_cmd.add_argument(
        "--slots",
        type=int,
        default=3,
        help="Number of available slots budget",
    )
    exact_cmd.add_argument(
        "--from-csv",
        action="store_true",
        help="Load instance from CSV files in data folder",
    )

    qubo_cmd = subparsers.add_parser(
        "qubo-info", help="Inspect QUBO formulation and Ising Hamiltonian"
    )
    qubo_cmd.add_argument(
        "--slots",
        type=int,
        default=3,
        help="Number of available slots budget",
    )
    qubo_cmd.add_argument(
        "--penalty",
        type=float,
        default=1.5,
        help="Penalty coefficient for one-hot and conflict terms",
    )
    qubo_cmd.add_argument(
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

    elif args.command == "exact":
        if args.from_csv:
            data_dir = REPO_ROOT / "data"
            courses_csv = data_dir / "sample_courses.csv"
            enrollments_csv = data_dir / "sample_enrollments.csv"
            inst = load_instance(courses_csv, enrollments_csv)
        else:
            inst = sample_instance()

        result = exact_schedule(inst, n_slots=args.slots)

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

        if result.extra.get("feasible", False):
            print(
                f"feasible schedules: {result.extra['n_feasible']} | "
                f"optimal schedules: {result.extra['n_optimal']} | "
                f"search space: {result.extra['search_space']}"
            )
        else:
            print(
                f"no valid schedule exists with {result.n_slots} slots "
                f"(searched {result.extra['search_space']} assignments)"
            )

        min_k = min_slots_exact(inst)
        print(f"minimum slots needed (exact): {min_k}")

    elif args.command == "qubo-info":
        if args.from_csv:
            data_dir = REPO_ROOT / "data"
            courses_csv = data_dir / "sample_courses.csv"
            enrollments_csv = data_dir / "sample_enrollments.csv"
            inst = load_instance(courses_csv, enrollments_csv)
        else:
            inst = sample_instance()

        model = build_qubo(
            inst,
            n_slots=args.slots,
            penalty_one_hot=args.penalty,
            penalty_conflict=args.penalty,
        )
        ising = qubo_to_ising(model)

        num_qubo_terms = int(np.count_nonzero(model.Q))
        num_z_terms = sum(1 for val in ising.h if abs(val) > 1e-12)
        num_zz_terms = len(ising.J)
        num_pauli_terms = len(ising.operator)

        print(
            f"qubits: {model.n_vars} | QUBO terms: {num_qubo_terms} | constant: {model.constant}"
        )
        print(
            f"ising: Z terms: {num_z_terms} | ZZ terms: {num_zz_terms} | "
            f"Pauli terms incl. identity: {num_pauli_terms}"
        )

        num_states = 1 << model.n_vars
        edges = conflict_edges(build_conflict_graph(inst))
        feasible_bitstrings = 0
        for a in itertools.product(range(args.slots), repeat=inst.n_courses):
            if all(a[u] != a[v] for u, v in edges):
                if validate_schedule(inst, list(a), args.slots).is_valid:
                    feasible_bitstrings += 1

        pct_feasible = (feasible_bitstrings / num_states) * 100.0
        print(
            f"feasible bitstrings: {feasible_bitstrings} of {num_states} ({pct_feasible:.4f}%)"
        )

        q_energies = qubo_energies_all(model)
        min_e = float(q_energies.min())
        ground_indices = np.where(np.abs(q_energies - min_e) < 1e-9)[0]
        G = len(ground_indices)

        all_ground_valid = True
        for g_idx in ground_indices:
            bits = [(int(g_idx) >> q) & 1 for q in range(model.n_vars)]
            dec = decode_bits(model, bits)
            if None in dec or not validate_schedule(inst, dec, args.slots).is_valid:
                all_ground_valid = False
                break

        print(
            f"ground-state energy: {min_e:.4f} | number of ground states: {G} | "
            f"all ground states valid: {all_ground_valid}"
        )

        ex_idx = ground_indices[0]
        ex_bits = [(int(ex_idx) >> q) & 1 for q in range(model.n_vars)]
        ex_dec = decode_bits(model, ex_bits)
        ex_valid = (None not in ex_dec) and validate_schedule(inst, ex_dec, args.slots).is_valid
        print(f"example ground state decodes to: {ex_dec} | valid: {ex_valid}")

        i_energies = ising_energies_all(ising)
        matches = bool(np.allclose(i_energies, q_energies))
        print(f"ising matches qubo on all 2^N states: {matches}")


if __name__ == "__main__":
    main()
