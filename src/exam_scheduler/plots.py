"""Visualization and plotting module for benchmark evaluations."""

from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_feasibility(df: pd.DataFrame, out_path: Any) -> None:
    """Plot E1 feasibility yield per sampling method with random-guess baseline."""
    out_file = Path(out_path)
    fig, ax = plt.subplots(figsize=(7, 5))

    e1_df = df[df["experiment"] == "E1_headline"]
    methods = [m for m in ["random-sampling", "qaoa-p1", "qaoa-p2"] if m in e1_df["method"].unique()]

    if methods and not e1_df.empty:
        data_to_plot = []
        labels = []
        for m in methods:
            m_data = e1_df[e1_df["method"] == m]["feasibility_yield"].dropna()
            data_to_plot.append(m_data)
            labels.append(m)

        if data_to_plot:
            ax.boxplot(data_to_plot, tick_labels=labels, showmeans=True)
            for i, d in enumerate(data_to_plot, 1):
                # Add jittered points
                x = np.random.normal(i, 0.04, size=len(d))
                ax.plot(x, d, "o", alpha=0.6, markersize=5)

        # Baseline probability line
        rand_prob = e1_df["random_guess_probability"].dropna()
        if not rand_prob.empty:
            baseline = rand_prob.iloc[0]
            ax.axhline(baseline, color="red", linestyle="--", label=f"Random Guess ({baseline*100:.4f}%)")
            ax.legend(loc="upper left")

    ax.set_ylabel("Feasibility Yield")
    ax.set_title("E1: Feasibility Yield Comparison Across Solvers")
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_objective(df: pd.DataFrame, out_path: Any) -> None:
    """Plot E1 objective value of valid schedules with optimum reference line."""
    out_file = Path(out_path)
    fig, ax = plt.subplots(figsize=(7, 5))

    e1_valid = df[(df["experiment"] == "E1_headline") & (df["is_valid"] == True)]
    methods = [m for m in ["exact", "greedy-largest_first", "greedy-DSATUR", "random-sampling", "qaoa-p1", "qaoa-p2"] if m in e1_valid["method"].unique()]

    if methods and not e1_valid.empty:
        tick_labels = []
        for idx, m in enumerate(methods, 1):
            m_data = e1_valid[e1_valid["method"] == m]["objective"].dropna()
            n_valid = len(m_data)
            tick_labels.append(f"{m}\n(N={n_valid})")
            x = np.random.normal(idx, 0.04, size=len(m_data))
            ax.plot(x, m_data, "o", alpha=0.7, markersize=6)
            if not m_data.empty:
                ax.plot(idx, m_data.mean(), "s", color="black", markersize=8)

        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels(tick_labels, rotation=15, ha="right")

        # Reference line for exact optimum
        opt_vals = e1_valid["optimal_objective"].dropna()
        if not opt_vals.empty:
            opt = opt_vals.iloc[0]
            ax.axhline(opt, color="green", linestyle="--", label=f"Exact Optimum ({opt:.2f})")
            ax.legend(loc="upper right")

    ax.set_ylabel("Schedule Objective (Lower is Better)")
    ax.set_title("E1: Objective Distribution (Valid Schedules Only)")
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_penalty(df: pd.DataFrame, out_path: Any) -> None:
    """Plot E2 feasibility yield and exact probability across penalty multipliers."""
    out_file = Path(out_path)
    fig, ax = plt.subplots(figsize=(7, 5))

    e2_df = df[df["experiment"] == "E2_penalty_sweep"]
    if not e2_df.empty:
        summary = e2_df.groupby("penalty").agg({
            "feasibility_yield": "mean",
            "exact_feasible_probability": "mean",
            "ground_state_valid": "first",
        }).reset_index().sort_values("penalty")

        penalties = summary["penalty"].values
        mean_yields = summary["feasibility_yield"].values
        mean_probs = summary["exact_feasible_probability"].values
        gs_valid = summary["ground_state_valid"].values

        ax.plot(penalties, mean_yields, "o-", label="Mean Feasibility Yield (Sampled)")
        ax.plot(penalties, mean_probs, "s--", label="Exact Feasible Probability (Statevector)")

        # Mark penalties where ground state is invalid
        for p, y, valid in zip(penalties, mean_yields, gs_valid):
            if not bool(valid):
                ax.scatter([p], [y], color="red", s=100, zorder=5, label="Ground State Invalid" if p == penalties[0] else None)

        ax.legend(loc="best")

    ax.set_xlabel("Penalty Multiplier (A = B)")
    ax.set_ylabel("Probability / Yield")
    ax.set_title("E2: Penalty Sweep Impact on Feasibility")
    ax.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_scaling(df: pd.DataFrame, out_path: Any) -> None:
    """Plot E3 scaling comparison: yield vs qubits, and median runtime vs qubits."""
    out_file = Path(out_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    e3_df = df[df["experiment"] == "E3_scaling"]

    if not e3_df.empty:
        # Panel 1: Feasibility yield vs Qubits
        for m in ["random-sampling", "qaoa-p1"]:
            sub = e3_df[e3_df["method"] == m]
            if not sub.empty:
                agg = sub.groupby("n_qubits")["feasibility_yield"].mean().reset_index()
                ax1.plot(agg["n_qubits"], agg["feasibility_yield"], "o-", label=m)

        rand_line = e3_df.groupby("n_qubits")["random_guess_probability"].first().reset_index()
        if not rand_line.empty:
            ax1.plot(rand_line["n_qubits"], rand_line["random_guess_probability"], "r--", label="Random Guess Probability")

        ax1.set_xlabel("Number of Qubits (N)")
        ax1.set_ylabel("Feasibility Yield")
        ax1.set_title("Yield vs Qubit Count")
        ax1.legend(loc="best")
        ax1.grid(True, linestyle=":", alpha=0.6)

        # Panel 2: Median Runtime vs Qubits (log y)
        all_methods = e3_df["method"].unique()
        for m in all_methods:
            sub = e3_df[e3_df["method"] == m]
            if not sub.empty:
                agg = sub.groupby("n_qubits")["runtime_s"].median().reset_index()
                ax2.plot(agg["n_qubits"], agg["runtime_s"], "o-", label=m)

        ax2.set_yscale("log")
        ax2.set_xlabel("Number of Qubits (N)")
        ax2.set_ylabel("Median Runtime (seconds, log scale)")
        ax2.set_title("Runtime Scaling vs Qubit Count")
        ax2.legend(loc="best")
        ax2.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def generate_all_plots(df: pd.DataFrame, out_dir: Any) -> None:
    """Generate and save all 4 benchmark figure visualizations to out_dir."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    plot_feasibility(df, out_path / "fig_feasibility.png")
    plot_objective(df, out_path / "fig_objective.png")
    plot_penalty(df, out_path / "fig_penalty.png")
    plot_scaling(df, out_path / "fig_scaling.png")
