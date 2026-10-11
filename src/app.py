"""Streamlit Interactive Dashboard for Quantum Exam Scheduler.

Provides an interactive GUI for problem configuration, conflict graph visualization,
schedule solving via classical heuristics, exact search, and QAOA,
quantum circuit diagnostics, and benchmark results analysis.
"""

from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from exam_scheduler.app_helpers import (
    build_instance,
    load_benchmark_results,
    qubit_budget,
    runs_to_dataframe,
    timetable_dataframe,
    top_counts_dataframe,
)
from exam_scheduler.classical import exact_schedule, greedy_schedule, random_sampling_schedule
from exam_scheduler.conflict_graph import build_conflict_graph, draw_conflict_graph
from exam_scheduler.data import ExamInstance, load_instance, sample_instance
from exam_scheduler.qaoa_solver import qaoa_schedule
from exam_scheduler.results import SolverResult

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Quantum Exam Scheduler", layout="wide")
st.title("Quantum Exam Scheduler")

# -----------------------------------------------------------------------------
# Sidebar: Instance Selection & Inputs
# -----------------------------------------------------------------------------
st.sidebar.header("Problem Instance")

data_kind = st.sidebar.radio(
    "Data Source",
    options=["Built-in sample (5 courses)", "Ring instance", "Random instance", "Upload CSV"],
    key="data_kind",
)

instance: Optional[ExamInstance] = None

if data_kind == "Built-in sample (5 courses)":
    instance = build_instance("sample")

elif data_kind == "Ring instance":
    ring_n = st.sidebar.slider("Number of courses", min_value=3, max_value=8, value=5, key="ring_n")
    data_seed = int(st.sidebar.number_input("Random Seed", value=42, key="data_seed"))
    instance = build_instance("ring", n_courses=ring_n, seed=data_seed)

elif data_kind == "Random instance":
    rand_courses = st.sidebar.slider("Number of courses", min_value=3, max_value=8, value=5, key="rand_courses")
    rand_students = st.sidebar.slider("Number of students", min_value=20, max_value=200, value=60, key="rand_students")
    data_seed = int(st.sidebar.number_input("Random Seed", value=42, key="data_seed"))
    instance = build_instance("random", n_courses=rand_courses, n_students=rand_students, seed=data_seed)

elif data_kind == "Upload CSV":
    up_courses = st.sidebar.file_uploader("Courses CSV (course_code, title)", type=["csv"], key="up_courses")
    up_enrol = st.sidebar.file_uploader("Enrollments CSV (student_id, course_code)", type=["csv"], key="up_enrol")
    if up_courses is not None and up_enrol is not None:
        try:
            courses_df = pd.read_csv(up_courses)
            enrol_df = pd.read_csv(up_enrol)
            codes = tuple(courses_df["course_code"].astype(str))
            titles = tuple(courses_df["title"].astype(str)) if "title" in courses_df.columns else tuple(f"Course {c}" for c in codes)
            enrol_df["student_id"] = enrol_df["student_id"].astype(str)
            enrol_df["course_code"] = enrol_df["course_code"].astype(str)
            instance = ExamInstance(codes, titles, enrol_df)
        except ValueError as e:
            st.error(f"Invalid uploaded instance: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Error loading CSV files: {e}")
            st.stop()
    else:
        st.info("Please upload both courses and enrollments CSV files to proceed.")
        st.stop()

# -----------------------------------------------------------------------------
# Sidebar: Scheduling Parameters
# -----------------------------------------------------------------------------
st.sidebar.header("Schedule Settings")

n_slots = st.sidebar.slider("Slots Budget (K)", min_value=1, max_value=6, value=3, key="n_slots")

algo_options = [
    "Greedy (largest first)",
    "Greedy (DSATUR)",
    "Exact search",
    "Random sampling",
    "QAOA",
]
selected_algos = st.sidebar.multiselect(
    "Algorithms to run",
    options=algo_options,
    default=["Greedy (largest first)", "Exact search", "Random sampling", "QAOA"],
    key="algorithms",
)

st.sidebar.subheader("Solver Hyperparameters")
penalty = float(st.sidebar.slider("QUBO Penalty (A = B)", min_value=0.5, max_value=5.0, value=1.5, step=0.1, key="penalty"))
shots = int(st.sidebar.select_slider("Measurement Shots", options=[256, 1024, 4096, 8192], value=4096, key="shots"))
seed = int(st.sidebar.number_input("Solver Random Seed", value=42, key="seed"))

reps = int(st.sidebar.radio("QAOA Layers (p)", options=[1, 2], index=0, key="reps"))
restarts = int(st.sidebar.slider("QAOA Restarts", min_value=1, max_value=5, value=2, key="restarts"))
maxiter = int(st.sidebar.slider("COBYLA Max Iterations", min_value=20, max_value=300, value=100, key="maxiter"))

# -----------------------------------------------------------------------------
# Sidebar: Live Qubit Budget & Memory Guard
# -----------------------------------------------------------------------------
st.sidebar.subheader("Resource Budget")
budget = qubit_budget(instance, n_slots)
n_qubits = budget["n_qubits"]
state_mib = budget["state_bytes"] / (1024 * 1024)

st.sidebar.markdown(f"**Qubits (N = n x K):** `{n_qubits}`")
st.sidebar.markdown(f"**Statevector Memory:** `{state_mib:.2f} MiB`")

allow_slow = False
if reps == 2:
    st.sidebar.info("p=2 layers with multiple restarts can take several minutes to simulate.")

if budget["qaoa_blocked"]:
    st.sidebar.warning(f"QAOA blocked: N={n_qubits} > 18 qubits exceeds practical classical statevector memory.")
elif budget["qaoa_slow"]:
    st.sidebar.warning(f"N={n_qubits} qubits (15 < N <= 18) simulation will be slow.")
    allow_slow = st.sidebar.checkbox("I understand this can take very long", value=False, key="allow_slow")

if not budget["exact_allowed"]:
    st.sidebar.warning(f"Exact search space K^n={n_slots**instance.n_courses:,} exceeds 2,000,000; exact search will be skipped.")

run_clicked = st.sidebar.button("Run Solvers", key="run")

# -----------------------------------------------------------------------------
# Input Signature and Session State Handling
# -----------------------------------------------------------------------------
current_signature = (
    data_kind,
    instance.course_codes,
    len(instance.enrollments),
    n_slots,
    tuple(selected_algos),
    penalty,
    shots,
    seed,
    reps,
    restarts,
    maxiter,
    allow_slow,
)

if run_clicked:
    runs_results: Dict[str, SolverResult] = {}
    stored_qaoa_outcome = None

    # 1. Greedy Largest First
    if "Greedy (largest first)" in selected_algos:
        with st.spinner("Running Greedy (largest first)..."):
            res_glf = greedy_schedule(instance, n_slots=n_slots, strategy="largest_first")
            runs_results["Greedy (largest first)"] = res_glf

    # 2. Greedy DSATUR
    if "Greedy (DSATUR)" in selected_algos:
        with st.spinner("Running Greedy (DSATUR)..."):
            res_gds = greedy_schedule(instance, n_slots=n_slots, strategy="DSATUR")
            runs_results["Greedy (DSATUR)"] = res_gds

    # 3. Exact Search
    if "Exact search" in selected_algos:
        if budget["exact_allowed"]:
            with st.spinner("Running Exact Search..."):
                res_exact = exact_schedule(instance, n_slots=n_slots)
                runs_results["Exact search"] = res_exact
        else:
            st.warning("Skipped Exact search: search space exceeds 2,000,000.")

    # 4. Random Sampling
    if "Random sampling" in selected_algos:
        with st.spinner("Running Random Sampling baseline..."):
            res_rand = random_sampling_schedule(instance, n_slots=n_slots, shots=shots, seed=seed)
            runs_results["Random sampling"] = res_rand

    # 5. QAOA
    if "QAOA" in selected_algos:
        if budget["qaoa_blocked"]:
            st.warning("QAOA skipped: problem size exceeds 18 qubits limit.")
        elif budget["qaoa_slow"] and not allow_slow:
            st.warning("QAOA skipped: please check 'I understand this can take very long' to proceed.")
        else:
            st.info("QAOA training ... this can take a minute or more")
            with st.spinner("Optimizing QAOA circuit parameters..."):
                q_outcome = qaoa_schedule(
                    instance,
                    n_slots=n_slots,
                    reps=reps,
                    penalty=penalty,
                    shots=shots,
                    n_restarts=restarts,
                    maxiter=maxiter,
                    seed=seed,
                )
                runs_results[f"QAOA (p={reps})"] = q_outcome.result
                stored_qaoa_outcome = q_outcome

    st.session_state["results"] = runs_results
    st.session_state["qaoa_outcome"] = stored_qaoa_outcome
    st.session_state["signature"] = current_signature

has_stored_results = "results" in st.session_state
inputs_match = has_stored_results and (st.session_state.get("signature") == current_signature)

if has_stored_results and not inputs_match:
    st.warning("Inputs changed - click Run to refresh the results")

# -----------------------------------------------------------------------------
# Main Dashboard Tabs
# -----------------------------------------------------------------------------
tab_data, tab_graph, tab_results, tab_qaoa, tab_bench = st.tabs(
    ["Data", "Conflict graph", "Results", "QAOA analysis", "Benchmark"]
)

# -----------------------------------------------------------------------------
# TAB 1: Data
# -----------------------------------------------------------------------------
with tab_data:
    st.header("Exam Instance Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Courses", instance.n_courses)
    col2.metric("Unique Students", instance.n_students)
    col3.metric("Total Enrollments", len(instance.enrollments))

    st.subheader("Course Directory")
    st.dataframe(instance.course_table(), width="stretch")

    st.subheader("Shared-Students Conflict Matrix")
    shared_mat = instance.shared_students()
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(shared_mat, cmap="Blues")
    fig.colorbar(cax, ax=ax)
    ax.set_xticks(range(instance.n_courses))
    ax.set_yticks(range(instance.n_courses))
    ax.set_xticklabels(instance.course_codes)
    ax.set_yticklabels(instance.course_codes)
    for i in range(instance.n_courses):
        for j in range(instance.n_courses):
            val = shared_mat[i, j]
            color = "white" if val > shared_mat.max() / 2 else "black"
            ax.text(j, i, str(int(val)), ha="center", va="center", color=color)
    ax.set_title("Shared Student Counts Between Courses")
    st.pyplot(fig)
    plt.close(fig)

    st.caption("Synthetic data for demonstration and benchmarking purposes.")

# -----------------------------------------------------------------------------
# TAB 2: Conflict Graph
# -----------------------------------------------------------------------------
with tab_graph:
    st.header("Course Conflict Graph")
    graph = build_conflict_graph(instance)

    slot_of: Optional[Dict[int, int]] = None
    if inputs_match:
        results = st.session_state["results"]
        valid_methods = [m for m, r in results.items() if r.is_valid and r.assignment is not None]
        if valid_methods:
            color_choice = st.selectbox("colour by method", ["Uncoloured"] + valid_methods, key="color_choice")
            if color_choice != "Uncoloured":
                assignment = results[color_choice].assignment
                if assignment:
                    slot_of = {c: assignment[c] for c in range(instance.n_courses) if assignment[c] is not None}

    fig, ax = plt.subplots(figsize=(7, 6))
    draw_conflict_graph(graph, slot_of=slot_of, ax=ax, title="Conflict Graph (Shared Enrollments)")
    st.pyplot(fig)
    plt.close(fig)

# -----------------------------------------------------------------------------
# TAB 3: Results
# -----------------------------------------------------------------------------
with tab_results:
    st.header("Solver Execution Results")
    if inputs_match:
        results = st.session_state["results"]

        # Determine optimum from Exact search if present and valid
        optimum = None
        if "Exact search" in results and results["Exact search"].is_valid:
            optimum = results["Exact search"].objective

        runs_df = runs_to_dataframe(results, optimum=optimum)
        st.subheader("Performance Comparison")
        st.dataframe(runs_df, width="stretch")

        if optimum is None:
            st.caption("Optimum unknown ('-'). Include 'Exact search' among algorithms to compute exact optimality gaps.")

        # Show Schedule Inspector
        st.subheader("Inspect Generated Schedule")
        method_names = list(results.keys())
        if method_names:
            inspect_method = st.selectbox("Show schedule of", method_names, key="inspect_method")
            res_sel = results[inspect_method]

            if res_sel.is_valid and res_sel.assignment is not None:
                st.success(f"Schedule for {inspect_method} is VALID (Objective: {res_sel.objective:.4f})")
                tt_df = timetable_dataframe(instance, res_sel.assignment, n_slots)
                st.dataframe(tt_df, width="stretch")

                ct = instance.course_table().copy()
                ct["Assigned Slot"] = [res_sel.assignment[c] for c in range(instance.n_courses)]
                st.dataframe(ct, width="stretch")
            else:
                st.error(f"Schedule for {inspect_method} is INVALID")
                messages = res_sel.extra.get("messages", []) if res_sel.extra else []
                if messages:
                    for m in messages:
                        st.error(m)
                else:
                    st.error("No valid schedule was found.")

        # Objective Bar Chart for Valid Results Only
        valid_objs = {
            m: r.objective for m, r in results.items() if r.is_valid and r.objective is not None
        }
        if valid_objs:
            st.subheader("Objective Value by Solver (Valid Schedules Only, Lower is Better)")
            chart_df = pd.DataFrame(list(valid_objs.items()), columns=["Method", "Objective"]).set_index("Method")
            st.bar_chart(chart_df)

    else:
        st.info("Select algorithms in the sidebar and click 'Run Solvers' to generate schedules.")

# -----------------------------------------------------------------------------
# TAB 4: QAOA Analysis
# -----------------------------------------------------------------------------
with tab_qaoa:
    st.header("QAOA Quantum Analysis")
    if inputs_match and st.session_state.get("qaoa_outcome") is not None:
        q_outcome = st.session_state["qaoa_outcome"]

        # Circuit statistics metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Qubits", q_outcome.circuit_stats["num_qubits"])
        m2.metric("Circuit Depth", q_outcome.circuit_stats["depth"])
        m3.metric("CX Gates", q_outcome.circuit_stats["cx"])
        m4.metric("Parameters", q_outcome.circuit_stats["num_parameters"])

        st.subheader("Feasibility Yield & Sampling Probabilities")
        p1, p2, p3 = st.columns(3)
        p1.metric("Exact Feasible Prob", f"{q_outcome.exact_feasible_probability * 100:.2f}%")
        p2.metric("Shot Feasibility Yield", f"{q_outcome.feasibility_yield * 100:.2f}%")
        p3.metric("Random-Guess Prob", f"{q_outcome.random_guess_probability * 100:.4f}%")

        # Top bitstrings table and stacked bar chart
        st.subheader("Top Measured Bitstrings")
        top_df = top_counts_dataframe(instance, n_slots, q_outcome.counts, top=15)
        st.dataframe(top_df, width="stretch")

        bar_chart_data = pd.DataFrame({
            "Bitstring": top_df["bitstring"],
            "Valid Share (%)": top_df.apply(lambda r: r["share_pct"] if r["valid"] else 0.0, axis=1),
            "Invalid Share (%)": top_df.apply(lambda r: r["share_pct"] if not r["valid"] else 0.0, axis=1),
        }).set_index("Bitstring")
        st.bar_chart(bar_chart_data)

        # Convergence chart across restarts
        st.subheader("Optimization Convergence (<H_C> per Evaluation)")
        history_dict = {}
        for r_info in q_outcome.restarts:
            r_num = r_info["restart"]
            label = f"Restart {r_num}" + (" (Best)" if r_num == q_outcome.best_restart else "")
            hist = r_info.get("history", [])
            if hist:
                history_dict[label] = pd.Series(hist)
        if history_dict:
            hist_df = pd.DataFrame(history_dict)
            st.line_chart(hist_df)

        with st.expander("Optimized Parameters and Circuit Metadata"):
            st.write("**Parameter Names:**", q_outcome.param_names)
            st.write("**Optimal Angles:**", [float(x) for x in q_outcome.best_params])
            st.json(q_outcome.circuit_stats)

        st.caption("Noiseless simulation on a classical computer. No quantum advantage is claimed.")
    else:
        st.info("Run QAOA from the sidebar to inspect variational quantum circuit diagnostics.")

# -----------------------------------------------------------------------------
# TAB 5: Benchmark
# -----------------------------------------------------------------------------
with tab_bench:
    st.header("Precomputed Benchmark Suite")
    bench_data = load_benchmark_results("results")

    if bench_data["summary"] is not None:
        st.subheader("Benchmark Summary Table")
        st.dataframe(bench_data["summary"], width="stretch")

        if bench_data["figures"]:
            st.subheader("Benchmark Visualizations")
            col_a, col_b = st.columns(2)
            with col_a:
                if "feasibility" in bench_data["figures"]:
                    st.image(bench_data["figures"]["feasibility"], caption="E1 Feasibility Yield")
                if "penalty" in bench_data["figures"]:
                    st.image(bench_data["figures"]["penalty"], caption="E2 Penalty Sweep")
            with col_b:
                if "objective" in bench_data["figures"]:
                    st.image(bench_data["figures"]["objective"], caption="E1 Objective")
                if "scaling" in bench_data["figures"]:
                    st.image(bench_data["figures"]["scaling"], caption="E3 Scaling")

        if bench_data["meta"] is not None:
            st.subheader("Environment & Metadata")
            st.json(bench_data["meta"])

    else:
        st.info("Benchmark results not found. Run: python src/main.py benchmark --out results")

    st.caption(
        "How to read this: Feasibility yield is the share of measured shots that decode to a "
        "validator-approved schedule; random-guess probability is the share of ALL bitstrings that are valid; "
        "exact/greedy are deterministic so yield does not apply."
    )

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.divider()
st.caption(
    "Limitations: Simulator only, small instances (<= 15 qubits recommended), "
    "penalty-based QUBO, results vary with seed."
)
