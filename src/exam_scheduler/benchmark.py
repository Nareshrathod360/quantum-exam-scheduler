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
from exam_scheduler.plots import generate_all_plots
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


def _format_sig_figs(val: float, n: int = 3) -> str:
    """Format float to n significant digits without scientific notation where possible."""
    if pd.isna(val):
        return "-"
    if val == 0:
        return "0"
    import math
    mag = math.floor(math.log10(abs(val)))
    decimals = n - 1 - mag
    if decimals > 0:
        return f"{val:.{decimals}f}"
    else:
        return f"{round(val, decimals):.0f}"






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
                "feasibility_yield": np.nan,
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
                "feasibility_yield": np.nan,
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
    # Generate Reports and Save Metadata
    # -------------------------------------------------------------------------
    write_reports(df_runs, out_path)

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


def write_reports(df: pd.DataFrame, out_dir: Any) -> pd.DataFrame:
    """Write benchmark_summary.csv, benchmark_summary.md, and all 4 figures from a runs DataFrame."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    group_cols = ["experiment", "instance", "method", "reps", "penalty"]

    for keys, group in df.groupby(group_cols, dropna=False):
        exp, inst_name, meth, reps_val, pen_val = keys
        valid_subset = group[group["is_valid"] == True]

        runs_count = len(group)
        n_valid_runs = int((group["is_valid"] == True).sum())
        valid_run_rate = float((group["is_valid"] == True).mean())

        # Feasibility yield only applies to sampling methods (random-sampling, qaoa-*)
        meth_str = str(meth)
        is_det = (
            meth_str in ("exact", "greedy-largest_first", "greedy-DSATUR")
            or meth_str.startswith("greedy-")
            or meth_str == "exact"
        )
        if is_det:
            mean_yield = np.nan
            med_yield = np.nan
        else:
            mean_yield = float(group["feasibility_yield"].mean()) if group["feasibility_yield"].notna().any() else np.nan
            med_yield = float(group["feasibility_yield"].median()) if group["feasibility_yield"].notna().any() else np.nan

        # Optimum found rate: objective matches optimal_objective within 1e-9
        opt_matches = group.apply(
            lambda r: bool(r["is_valid"]) and pd.notna(r["optimal_objective"]) and pd.notna(r["objective"]) and abs(r["objective"] - r["optimal_objective"]) <= 1e-9,
            axis=1,
        )
        opt_found_rate = float(opt_matches.mean())

        mean_gap = float(valid_subset["optimality_gap"].mean()) if not valid_subset.empty and valid_subset["optimality_gap"].notna().any() else np.nan
        mean_slots = float(valid_subset["slots_used"].mean()) if not valid_subset.empty and valid_subset["slots_used"].notna().any() else np.nan
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
            "n_valid_runs": n_valid_runs,
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
    # Format Display DataFrame for Markdown Table
    # -------------------------------------------------------------------------
    disp = pd.DataFrame()
    disp["experiment"] = summary_df["experiment"].astype(str)
    disp["instance"] = summary_df["instance"].astype(str)
    disp["method"] = summary_df["method"].astype(str)
    disp["reps"] = summary_df["reps"].apply(lambda v: str(int(v)) if pd.notna(v) else "-")
    disp["penalty"] = summary_df["penalty"].apply(lambda v: f"{v:.1f}" if pd.notna(v) else "-")
    disp["runs"] = summary_df["runs"].apply(lambda v: str(int(v)))
    disp["n_valid_runs"] = summary_df["n_valid_runs"].apply(lambda v: str(int(v)))
    disp["valid_run_rate"] = summary_df["valid_run_rate"].apply(lambda v: f"{v * 100:.1f}%" if pd.notna(v) else "-")
    disp["mean_feasibility_yield"] = summary_df["mean_feasibility_yield"].apply(lambda v: f"{v * 100:.2f}%" if pd.notna(v) else "-")
    disp["median_feasibility_yield"] = summary_df["median_feasibility_yield"].apply(lambda v: f"{v * 100:.2f}%" if pd.notna(v) else "-")
    disp["optimum_found_rate"] = summary_df["optimum_found_rate"].apply(lambda v: f"{v * 100:.1f}%" if pd.notna(v) else "-")
    disp["mean_optimality_gap"] = summary_df["mean_optimality_gap"].apply(lambda v: f"{v * 100:.1f}%" if pd.notna(v) else "-")
    disp["mean_slots_used"] = summary_df["mean_slots_used"].apply(
        lambda v: (f"{v:.1f}" if v == round(v, 1) else f"{v:.2f}") if pd.notna(v) else "-"
    )
    disp["median_runtime_s"] = summary_df["median_runtime_s"].apply(lambda v: _format_sig_figs(v, 3))
    disp["min_runtime_s"] = summary_df["min_runtime_s"].apply(lambda v: _format_sig_figs(v, 3))
    disp["max_runtime_s"] = summary_df["max_runtime_s"].apply(lambda v: _format_sig_figs(v, 3))

    # Headline random guess
    e1_rows = df[df["experiment"] == "E1_headline"]
    if not e1_rows.empty:
        inst_name = str(e1_rows["instance"].iloc[0])
        k = int(e1_rows["n_slots"].iloc[0])
        if inst_name == "sample":
            h_inst = sample_instance()
        elif inst_name.startswith("ring_"):
            h_inst = ring_instance(int(inst_name.split("_")[1]), seed=42)
        else:
            h_inst = sample_instance()
    else:
        h_inst = sample_instance()
        k = 3

    h_mask = valid_state_mask(h_inst, k)
    f_count = int(h_mask.sum())
    n_vars = h_inst.n_courses * k
    total_states = 1 << n_vars
    pct_rand = (f_count / total_states) * 100.0

    md_lines = [
        "# Quantum Exam Scheduler Benchmark Summary",
        "",
        f"**Random-guess feasibility = {f_count}/{total_states} ({pct_rand:.4f}%)**",
        "",
        "## Performance Comparison Table",
        "",
        _dataframe_to_markdown(disp),
        "",
        "### Experiment Notes",
        "- **E1 (Headline):** Compares exact search, greedy heuristics, classical random sampling, and QAOA on the benchmark problem to evaluate solution quality and feasibility yield.",
        "- **E2 (Penalty sweep):** Analyzes how the QUBO constraint penalty parameter affects ground-state validity and QAOA sampling success.",
        "- **E3 (Scaling):** Assesses solver feasibility yield and runtime scaling as the number of courses (qubit count) increases.",
    ]

    # E3 random guess probabilities
    e3_rows = df[df["experiment"] == "E3_scaling"]
    if not e3_rows.empty:
        md_lines.append("")
        md_lines.append("### Random-Guess Feasibility by Problem Size (E3)")
        seen_sizes = set()
        for _, row in e3_rows.iterrows():
            size_key = (str(row["instance"]), int(row["n_courses"]), int(row["n_slots"]))
            if size_key in seen_sizes:
                continue
            seen_sizes.add(size_key)
            i_name, n_c, n_s = size_key
            if i_name.startswith("ring_"):
                s_inst = ring_instance(int(i_name.split("_")[1]), seed=42)
            elif i_name == "sample":
                s_inst = sample_instance()
            else:
                s_inst = ring_instance(n_c, seed=42)
            mask_s = valid_state_mask(s_inst, n_s)
            c_s = int(mask_s.sum())
            q_s = s_inst.n_courses * n_s
            t_s = 1 << q_s
            p_s = (c_s / t_s) * 100.0
            md_lines.append(f"- **{i_name}** ({n_c} courses, {n_s} slots, {q_s} qubits): {c_s}/{t_s} ({p_s:.4f}%)")

    md_lines.append("")
    summary_md_path = out_path / "benchmark_summary.md"
    summary_md_path.write_text("\n".join(md_lines), encoding="utf-8")

    # Generate all four figures
    generate_all_plots(df, out_path)

    return summary_df

