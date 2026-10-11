"""Comprehensive benchmarking module for classical and quantum exam schedulers."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from typing import Any, Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import qiskit
import qiskit_aer
import scipy

from exam_scheduler.classical import exact_schedule, greedy_schedule, random_sampling_schedule
from exam_scheduler.data import DEFAULT_SEED, ExamInstance, ring_instance, sample_instance
from exam_scheduler.qaoa_solver import qaoa_schedule
from exam_scheduler.qubo import build_qubo, qubo_energies_all, valid_state_mask
from exam_scheduler.results import optimality_gap


@dataclass
class BenchmarkConfig:
    """Benchmark experiment configuration parameters."""

    n_slots: int = 3
    random_seeds: int = 20
    p1_seeds: int = 10
    p2_seeds: int = 3
    penalty_seeds: int = 3
    scaling_seeds: int = 3
    shots: int = 4096
    restarts: int = 3
    maxiter: int = 150
    penalty: float = 1.5
    penalties: Tuple[float, ...] = (1.0, 1.1, 1.5, 2.0, 3.0)
    scaling_sizes: Tuple[int, ...] = (3, 4, 5)
    include_p2: bool = True
    base_seed: int = 42

    @classmethod
    def quick(cls) -> "BenchmarkConfig":
        """Return a fast configuration suitable for validation tests."""
        return cls(
            random_seeds=5,
            p1_seeds=2,
            p2_seeds=0,
            include_p2=False,
            penalty_seeds=1,
            scaling_seeds=1,
            shots=512,
            restarts=1,
            maxiter=40,
            penalties=(1.0, 1.5),
            scaling_sizes=(3, 4),
        )


BENCHMARK_COLUMNS = [
    "experiment",
    "instance",
    "n_courses",
    "n_slots",
    "n_qubits",
    "method",
    "seed",
    "penalty",
    "reps",
    "shots",
    "restarts",
    "maxiter",
    "assignment",
    "is_valid",
    "slots_used",
    "n_conflicts",
    "affected_students",
    "objective",
    "optimal_objective",
    "optimality_gap",
    "feasibility_yield",
    "n_valid_shots",
    "exact_feasible_probability",
    "random_guess_probability",
    "ground_state_valid",
    "final_energy",
    "runtime_s",
    "circuit_depth",
    "cx_gates",
    "top_bitstring_valid",
]


def _format_assignment(assignment: Optional[List[Optional[int]]]) -> str:
    """Format an assignment list to a space-separated string or empty string."""
    if assignment is None:
        return ""
    return " ".join(str(s) for s in assignment)


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Convert a pandas DataFrame to a GitHub-flavored Markdown table string without external dependencies."""
    headers = [str(c) for c in df.columns]
    col_widths = [len(h) for h in headers]
    rows = []
    for row in df.itertuples(index=False):
        row_str = [str(x) if x is not None else "" for x in row]
        for i, val in enumerate(row_str):
            col_widths[i] = max(col_widths[i], len(val))
        rows.append(row_str)

    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    separator_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
    data_lines = [
        "| " + " | ".join(val.ljust(col_widths[i]) for i, val in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator_line] + data_lines)



def run_benchmark(
    config: BenchmarkConfig,
    out_dir: Any,
    headline_instance: Optional[ExamInstance] = None,
    resume: bool = False,
    progress: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Execute full benchmark experiments across headline, penalty sweep, and scaling regimes."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if headline_instance is None:
        headline_instance = sample_instance()
        headline_name = "sample"
    else:
        headline_name = f"inst_{headline_instance.n_courses}"

    runs_csv = out_path / "benchmark_runs.csv"
    existing_keys = set()
    records: List[Dict[str, Any]] = []

    if resume and runs_csv.exists():
        existing_df = pd.read_csv(runs_csv)
        for _, row in existing_df.iterrows():
            key = (
                str(row["experiment"]),
                str(row["instance"]),
                int(row["n_slots"]),
                str(row["method"]),
                int(row["seed"]) if pd.notna(row["seed"]) else None,
                float(row["penalty"]) if pd.notna(row["penalty"]) else None,
                int(row["reps"]) if pd.notna(row["reps"]) else None,
            )
            existing_keys.add(key)
            records.append(row.to_dict())

    # Precompute / cache exact optimal objective per (instance_id, n_slots)
    exact_cache: Dict[Tuple[str, int], float] = {}

    def get_exact_opt(inst: ExamInstance, inst_name: str, k: int) -> float:
        cache_id = (inst_name, k)
        if cache_id not in exact_cache:
            res_ex = exact_schedule(inst, n_slots=k)
            exact_cache[cache_id] = float(res_ex.objective) if res_ex.objective is not None else float("nan")
        return exact_cache[cache_id]

    # Pre-populate headline optimum
    headline_opt = get_exact_opt(headline_instance, headline_name, config.n_slots)

    # Helper to append a single run to CSV immediately
    def save_record(rec: Dict[str, Any]) -> None:
        records.append(rec)
        row_df = pd.DataFrame([rec], columns=BENCHMARK_COLUMNS)
        file_exists = runs_csv.exists()
        row_df.to_csv(runs_csv, mode="a", header=not file_exists, index=False)

    # -------------------------------------------------------------------------
    # Plan Experiment Run Specifications
    # -------------------------------------------------------------------------
    plan: List[Dict[str, Any]] = []

    # E1: Headline experiments
    plan.append({"exp": "E1_headline", "inst": headline_instance, "name": headline_name, "k": config.n_slots, "method": "exact"})
    plan.append({"exp": "E1_headline", "inst": headline_instance, "name": headline_name, "k": config.n_slots, "method": "greedy-largest_first"})
    plan.append({"exp": "E1_headline", "inst": headline_instance, "name": headline_name, "k": config.n_slots, "method": "greedy-DSATUR"})

    for s_idx in range(config.random_seeds):
        seed = config.base_seed + s_idx
        plan.append({"exp": "E1_headline", "inst": headline_instance, "name": headline_name, "k": config.n_slots, "method": "random-sampling", "seed": seed, "shots": config.shots})

    for s_idx in range(config.p1_seeds):
        seed = config.base_seed + s_idx
        plan.append({"exp": "E1_headline", "inst": headline_instance, "name": headline_name, "k": config.n_slots, "method": "qaoa-p1", "reps": 1, "seed": seed, "shots": config.shots, "restarts": config.restarts, "maxiter": config.maxiter, "penalty": config.penalty})

    if config.include_p2 and config.p2_seeds > 0:
        for s_idx in range(config.p2_seeds):
            seed = config.base_seed + s_idx
            plan.append({"exp": "E1_headline", "inst": headline_instance, "name": headline_name, "k": config.n_slots, "method": "qaoa-p2", "reps": 2, "seed": seed, "shots": config.shots, "restarts": config.restarts, "maxiter": config.maxiter, "penalty": config.penalty})

    # E2: Penalty sweep experiments
    for p_val in config.penalties:
        for s_idx in range(config.penalty_seeds):
            seed = config.base_seed + s_idx
            plan.append({"exp": "E2_penalty_sweep", "inst": headline_instance, "name": headline_name, "k": config.n_slots, "method": "qaoa-p1", "reps": 1, "seed": seed, "shots": config.shots, "restarts": config.restarts, "maxiter": config.maxiter, "penalty": p_val})

    # E3: Scaling experiments
    for n_size in config.scaling_sizes:
        ring_inst = ring_instance(n_size, seed=config.base_seed)
        ring_name = f"ring_{n_size}"
        plan.append({"exp": "E3_scaling", "inst": ring_inst, "name": ring_name, "k": config.n_slots, "method": "exact"})
        plan.append({"exp": "E3_scaling", "inst": ring_inst, "name": ring_name, "k": config.n_slots, "method": "greedy-largest_first"})

        for s_idx in range(config.random_seeds):
            seed = config.base_seed + s_idx
            plan.append({"exp": "E3_scaling", "inst": ring_inst, "name": ring_name, "k": config.n_slots, "method": "random-sampling", "seed": seed, "shots": config.shots})

        for s_idx in range(config.scaling_seeds):
            seed = config.base_seed + s_idx
            plan.append({"exp": "E3_scaling", "inst": ring_inst, "name": ring_name, "k": config.n_slots, "method": "qaoa-p1", "reps": 1, "seed": seed, "shots": config.shots, "restarts": config.restarts, "maxiter": config.maxiter, "penalty": config.penalty})

    # -------------------------------------------------------------------------
    # Execute Plan
    # -------------------------------------------------------------------------
    total_runs = len(plan)
    seeds_used = set()

    for idx, item in enumerate(plan, 1):
        exp = item["exp"]
        inst = item["inst"]
        inst_name = item["name"]
        k = item["k"]
        method = item["method"]
        seed = item.get("seed")
        penalty = item.get("penalty", config.penalty if "qaoa" in method else None)
        reps = item.get("reps")
        shots = item.get("shots")
        restarts = item.get("restarts")
        maxiter = item.get("maxiter")

        if seed is not None:
            seeds_used.add(seed)

        key = (
            exp,
            inst_name,
            k,
            method,
            int(seed) if seed is not None else None,
            float(penalty) if penalty is not None else None,
            int(reps) if reps is not None else None,
        )

        if resume and key in existing_keys:
            continue

        n_qubits = inst.n_courses * k
        opt_obj = get_exact_opt(inst, inst_name, k)

        # Execute corresponding solver
        if method == "exact":
            res = exact_schedule(inst, n_slots=k)
            rec = {
                "experiment": exp,
                "instance": inst_name,
                "n_courses": inst.n_courses,
                "n_slots": k,
                "n_qubits": n_qubits,
                "method": method,
                "seed": np.nan,
                "penalty": np.nan,
                "reps": np.nan,
                "shots": np.nan,
                "restarts": np.nan,
                "maxiter": np.nan,
                "assignment": _format_assignment(res.assignment),
                "is_valid": res.is_valid,
                "slots_used": res.slots_used,
                "n_conflicts": res.n_conflicts,
                "affected_students": res.affected_students,
                "objective": res.objective,
                "optimal_objective": opt_obj,
                "optimality_gap": optimality_gap(res.objective, opt_obj) if res.is_valid else np.nan,
                "feasibility_yield": 1.0 if res.is_valid else 0.0,
                "n_valid_shots": np.nan,
                "exact_feasible_probability": np.nan,
                "random_guess_probability": np.nan,
                "ground_state_valid": np.nan,
                "final_energy": np.nan,
                "runtime_s": res.runtime_s,
                "circuit_depth": np.nan,
                "cx_gates": np.nan,
                "top_bitstring_valid": np.nan,
            }

        elif method.startswith("greedy-"):
            strat = method.replace("greedy-", "")
            res = greedy_schedule(inst, n_slots=k, strategy=strat)
            rec = {
                "experiment": exp,
                "instance": inst_name,
                "n_courses": inst.n_courses,
                "n_slots": k,
                "n_qubits": n_qubits,
                "method": method,
                "seed": np.nan,
                "penalty": np.nan,
                "reps": np.nan,
                "shots": np.nan,
                "restarts": np.nan,
                "maxiter": np.nan,
                "assignment": _format_assignment(res.assignment),
                "is_valid": res.is_valid,
                "slots_used": res.slots_used,
                "n_conflicts": res.n_conflicts,
                "affected_students": res.affected_students,
                "objective": res.objective,
                "optimal_objective": opt_obj,
                "optimality_gap": optimality_gap(res.objective, opt_obj) if res.is_valid else np.nan,
                "feasibility_yield": 1.0 if res.is_valid else 0.0,
                "n_valid_shots": np.nan,
                "exact_feasible_probability": np.nan,
                "random_guess_probability": np.nan,
                "ground_state_valid": np.nan,
                "final_energy": np.nan,
                "runtime_s": res.runtime_s,
                "circuit_depth": np.nan,
                "cx_gates": np.nan,
                "top_bitstring_valid": np.nan,
            }

        elif method == "random-sampling":
            res = random_sampling_schedule(inst, n_slots=k, shots=shots, seed=seed)
            mask = valid_state_mask(inst, k)
            rand_prob = float(mask.sum() / (1 << n_qubits))
            rec = {
                "experiment": exp,
                "instance": inst_name,
                "n_courses": inst.n_courses,
                "n_slots": k,
                "n_qubits": n_qubits,
                "method": method,
                "seed": seed,
                "penalty": np.nan,
                "reps": np.nan,
                "shots": shots,
                "restarts": np.nan,
                "maxiter": np.nan,
                "assignment": _format_assignment(res.assignment),
                "is_valid": res.is_valid,
                "slots_used": res.slots_used,
                "n_conflicts": res.n_conflicts,
                "affected_students": res.affected_students,
                "objective": res.objective,
                "optimal_objective": opt_obj,
                "optimality_gap": optimality_gap(res.objective, opt_obj) if res.is_valid else np.nan,
                "feasibility_yield": res.extra["feasibility_yield"],
                "n_valid_shots": res.extra["n_valid_shots"],
                "exact_feasible_probability": rand_prob,
                "random_guess_probability": rand_prob,
                "ground_state_valid": np.nan,
                "final_energy": np.nan,
                "runtime_s": res.runtime_s,
                "circuit_depth": np.nan,
                "cx_gates": np.nan,
                "top_bitstring_valid": np.nan,
            }

        elif method.startswith("qaoa-"):
            outcome = qaoa_schedule(
                inst,
                n_slots=k,
                reps=reps,
                penalty=penalty,
                shots=shots,
                n_restarts=restarts,
                maxiter=maxiter,
                seed=seed,
            )
            res = outcome.result

            # Check ground_state_valid if running E2 penalty sweep
            ground_valid = np.nan
            if exp == "E2_penalty_sweep":
                q_model = build_qubo(inst, k, penalty, penalty)
                q_energies = qubo_energies_all(q_model)
                min_e = q_energies.min()
                g_indices = np.where(np.abs(q_energies - min_e) < 1e-9)[0]
                mask = valid_state_mask(inst, k)
                ground_valid = bool(all(mask[g] for g in g_indices))

            rec = {
                "experiment": exp,
                "instance": inst_name,
                "n_courses": inst.n_courses,
                "n_slots": k,
                "n_qubits": n_qubits,
                "method": method,
                "seed": seed,
                "penalty": penalty,
                "reps": reps,
                "shots": shots,
                "restarts": restarts,
                "maxiter": maxiter,
                "assignment": _format_assignment(res.assignment),
                "is_valid": res.is_valid,
                "slots_used": res.slots_used,
                "n_conflicts": res.n_conflicts,
                "affected_students": res.affected_students,
                "objective": res.objective,
                "optimal_objective": opt_obj,
                "optimality_gap": optimality_gap(res.objective, opt_obj) if res.is_valid else np.nan,
                "feasibility_yield": outcome.feasibility_yield,
                "n_valid_shots": outcome.n_valid_shots,
                "exact_feasible_probability": outcome.exact_feasible_probability,
                "random_guess_probability": outcome.random_guess_probability,
                "ground_state_valid": ground_valid,
                "final_energy": outcome.final_energy,
                "runtime_s": res.runtime_s,
                "circuit_depth": outcome.circuit_stats["depth"],
                "cx_gates": outcome.circuit_stats["cx"],
                "top_bitstring_valid": outcome.top_bitstring_valid,
            }

        save_record(rec)
        yld_str = f"{rec['feasibility_yield'] * 100:.2f}%" if pd.notna(rec['feasibility_yield']) else "n/a"
        progress(
            f"[{exp} {idx}/{total_runs}] {method} seed={seed} -> valid={rec['is_valid']} "
            f"yield={yld_str} ({rec['runtime_s']:.2f} s)"
        )

    df_runs = pd.DataFrame(records, columns=BENCHMARK_COLUMNS)

    # -------------------------------------------------------------------------
    # Sanity Guard
    # -------------------------------------------------------------------------
    valid_rows = df_runs[df_runs["is_valid"] == True]
    violations = valid_rows[valid_rows["optimality_gap"] < -1e-9]
    if not violations.empty:
        violating_desc = violations.iloc[0].to_dict()
        raise ValueError(
            f"Sanity guard failure: solver beat exact optimum! Violating run: {violating_desc}"
        )

    # -------------------------------------------------------------------------
    # Summary Table Generation
    # -------------------------------------------------------------------------
    summary_rows: List[Dict[str, Any]] = []
    group_cols = ["experiment", "instance", "method", "reps", "penalty"]

    for keys, group in df_runs.groupby(group_cols, dropna=False):
        exp, inst_name, meth, reps_val, pen_val = keys
        valid_subset = group[group["is_valid"] == True]

        runs_count = len(group)
        valid_run_rate = float((group["is_valid"] == True).mean())
        mean_yield = float(group["feasibility_yield"].mean()) if group["feasibility_yield"].notna().any() else np.nan
        med_yield = float(group["feasibility_yield"].median()) if group["feasibility_yield"].notna().any() else np.nan

        # Optimum found rate: objective matches optimal_objective within 1e-9
        opt_matches = group.apply(
            lambda r: r["is_valid"] and abs(r["objective"] - r["optimal_objective"]) <= 1e-9,
            axis=1,
        )
        opt_found_rate = float(opt_matches.mean())

        mean_gap = float(valid_subset["optimality_gap"].mean()) if not valid_subset.empty else np.nan
        mean_slots = float(valid_subset["slots_used"].mean()) if not valid_subset.empty else np.nan
        med_rt = float(group["runtime_s"].median())
        min_rt = float(group["runtime_s"].min())
        max_rt = float(group["runtime_s"].max())

        summary_rows.append({
            "experiment": exp,
            "instance": inst_name,
            "method": meth,
            "reps": reps_val,
            "penalty": pen_val,
            "runs": runs_count,
            "valid_run_rate": valid_run_rate,
            "mean_feasibility_yield": mean_yield,
            "median_feasibility_yield": med_yield,
            "optimum_found_rate": opt_found_rate,
            "mean_optimality_gap": mean_gap,
            "mean_slots_used": mean_slots,
            "median_runtime_s": med_rt,
            "min_runtime_s": min_rt,
            "max_runtime_s": max_rt,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_path / "benchmark_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    # -------------------------------------------------------------------------
    # Markdown Summary File
    # -------------------------------------------------------------------------
    headline_mask = valid_state_mask(headline_instance, config.n_slots)
    f_count = int(headline_mask.sum())
    n_head_vars = headline_instance.n_courses * config.n_slots
    total_states = 1 << n_head_vars
    pct_rand = (f_count / total_states) * 100.0

    md_lines = [
        "# Quantum Exam Scheduler Benchmark Summary",
        "",
        f"**Random-guess feasibility = {f_count}/{total_states} ({pct_rand:.4f}%)**",
        "",
        "## Performance Comparison Table",
        "",
        _dataframe_to_markdown(summary_df),
        "",
    ]
    summary_md_path = out_path / "benchmark_summary.md"
    summary_md_path.write_text("\n".join(md_lines), encoding="utf-8")

    # -------------------------------------------------------------------------
    # Metadata File
    # -------------------------------------------------------------------------
    meta_info = {
        "config": asdict(config),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "qiskit": qiskit.__version__,
        "qiskit_aer": qiskit_aer.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "networkx": nx.__version__,
        "pandas": pd.__version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seeds_used": sorted(list(seeds_used)),
        "disclaimer": "No quantum advantage is claimed; all quantum runs are noiseless classical simulations.",
    }
    meta_path = out_path / "benchmark_meta.json"
    meta_path.write_text(json.dumps(meta_info, indent=2), encoding="utf-8")

    return df_runs
