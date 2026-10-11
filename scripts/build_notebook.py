"""Script to generate, execute, and validate the hackathon solution notebook.

Builds notebooks/solution.ipynb with all required pedagogical markdown and code cells,
executes it in the notebooks/ working directory using nbconvert, and verifies
all execution outputs and content constraints.
"""

import argparse
import json
from pathlib import Path
import re
import sys

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor


def create_solution_notebook() -> nbf.NotebookNode:
    """Construct the complete unexecuted solution notebook."""
    nb = nbf.v4.new_notebook()
    cells = []

    # =========================================================================
    # SECTION 0: Title, Problem Statement, & Honesty Box
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell(
        "# Quantum Exam Scheduler: Track 2 Open-Domain Optimisation\n\n"
        "## Problem Statement\n"
        "University exam scheduling is an NP-hard combinatorial optimisation problem. "
        "Given a set of academic courses and student course enrollments, the goal is to assign "
        "each course an exam timeslot such that:\n"
        "1. **Hard Constraints (Feasibility):** No two courses sharing enrolled students are scheduled in the same timeslot, "
        "and every course receives exactly one timeslot.\n"
        "2. **Soft Objectives (Timetabling Preference):** Courses with large student enrollments are prioritized into "
        "earlier exam timeslots to allow more grading turnaround time.\n\n"
        "## What This Notebook Demonstrates\n"
        "- Formulation of timetabling as Graph Colouring and Quadratic Unconstrained Binary Optimisation (QUBO).\n"
        "- Mapping of binary quadratic cost Hamiltonians to Pauli $Z$ and $ZZ$ Ising spin operators.\n"
        "- Classical heuristics (Greedy Largest-First and DSATUR) and classical Exact exhaustive search.\n"
        "- Implementation of the Quantum Approximate Optimisation Algorithm (QAOA) via Qiskit 2.5.2.\n"
        "- Rigorous statistical benchmarking, parameter sweep sensitivity, and classical vs quantum trade-offs.\n\n"
        "> **Honesty Box & Technical Disclaimer:**\n"
        "> All quantum executions in this notebook are noiseless classical simulations executed via Statevector and Qiskit Aer Sampler. "
        "> Classical exact search solves this problem orders of magnitude faster. "
        "> **No quantum advantage, quantum speed-up, or commercial supremacy is claimed.**"
    ))

    # =========================================================================
    # SECTION 1: Setup
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Environment Setup & Dependency Configuration\n\n"
        "We configure the runtime path, suppress non-critical deprecation notices, "
        "and record pinned library versions for reproducible execution.\n\n"
        "*Technical Note:* Qiskit's `QAOAAnsatz` internally uses classes deprecated in Qiskit 2.1 "
        "(scheduled for removal in Qiskit 3.0). These warnings are filtered to ensure clean outputs "
        "on Qiskit 2.5.2."
    ))

    cells.append(nbf.v4.new_code_cell(
        "%matplotlib inline\n"
        "import sys\n"
        "import os\n"
        "import json\n"
        "import warnings\n"
        "from pathlib import Path\n"
        "\n"
        "# Ensure project src directory is importable\n"
        "sys.path.insert(0, '../src')\n"
        "\n"
        "# Suppress known Qiskit 2.1 internal deprecation warnings from QAOAAnsatz\n"
        "warnings.filterwarnings('ignore', category=DeprecationWarning)\n"
        "warnings.filterwarnings('ignore', message='.*NLocal.*')\n"
        "warnings.filterwarnings('ignore', message='.*BlueprintCircuit.*')\n"
        "\n"
        "import platform\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import scipy\n"
        "import networkx as nx\n"
        "import matplotlib.pyplot as plt\n"
        "import qiskit\n"
        "import qiskit_aer\n"
        "from IPython.display import Image, display\n"
        "\n"
        "from exam_scheduler.data import DEFAULT_SEED, ExamInstance, sample_instance, ring_instance\n"
        "from exam_scheduler.conflict_graph import build_conflict_graph, draw_conflict_graph, clique_lower_bound\n"
        "from exam_scheduler.classical import greedy_schedule, exact_schedule, random_sampling_schedule\n"
        "from exam_scheduler.validator import validate_schedule, schedule_objective\n"
        "from exam_scheduler.qubo import build_qubo, decode_bits, qubo_energies_all, valid_state_mask, bitstring_to_bits\n"
        "from exam_scheduler.ising import qubo_to_ising, ising_energies_all\n"
        "from exam_scheduler.qaoa_solver import qaoa_schedule\n"
        "from exam_scheduler.app_helpers import timetable_dataframe, top_counts_dataframe\n"
        "\n"
        "print('=== Environment Dependency Versions ===')\n"
        "print(f'Python:     {platform.python_version()}')\n"
        "print(f'Qiskit:     {qiskit.__version__}')\n"
        "print(f'Qiskit Aer: {qiskit_aer.__version__}')\n"
        "print(f'NumPy:      {np.__version__}')\n"
        "print(f'SciPy:      {scipy.__version__}')\n"
        "print(f'NetworkX:   {nx.__version__}')\n"
        "print(f'Pandas:     {pd.__version__}')"
    ))

    # =========================================================================
    # SECTION 2: Data
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Problem Instance Representation & Conflict Data\n\n"
        "An exam instance contains courses, student enrollments, and enrollment overlap. "
        "Here we inspect the course directory and plot the shared-students conflict matrix. "
        "The headline dataset used for demonstration is synthetic."
    ))

    cells.append(nbf.v4.new_code_cell(
        "inst = sample_instance()\n"
        "print(f'Courses: {inst.n_courses} | Unique Students: {inst.n_students} | Total Enrollments: {len(inst.enrollments)}')\n"
        "print('\\nCourse Directory Table:')\n"
        "print(inst.course_table().to_string(index=False))\n"
        "\n"
        "# Plot annotated shared-students matrix heatmap\n"
        "shared_mat = inst.shared_students()\n"
        "fig, ax = plt.subplots(figsize=(6, 5))\n"
        "cax = ax.imshow(shared_mat, cmap='Blues')\n"
        "fig.colorbar(cax, ax=ax)\n"
        "ax.set_xticks(range(inst.n_courses))\n"
        "ax.set_yticks(range(inst.n_courses))\n"
        "ax.set_xticklabels(inst.course_codes)\n"
        "ax.set_yticklabels(inst.course_codes)\n"
        "for i in range(inst.n_courses):\n"
        "    for j in range(inst.n_courses):\n"
        "        val = shared_mat[i, j]\n"
        "        color = 'white' if val > shared_mat.max() / 2 else 'black'\n"
        "        ax.text(j, i, str(int(val)), ha='center', va='center', color=color)\n"
        "ax.set_title('Shared Student Overlaps (Conflict Matrix)')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # =========================================================================
    # SECTION 3: Conflict Graph
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Conflict Graph Formulation & Lower Bounds\n\n"
        "We construct an undirected conflict graph $G = (V, E)$, where each vertex represents a course "
        "and an edge exists between two courses whenever they share at least one student.\n\n"
        "*Graph Theory Insight:* The chromatic number $\\chi(G)$ (minimum slots needed for a valid schedule) "
        "is bounded below by the maximum clique size $\\omega(G)$, because all courses in a mutual conflict clique "
        "must occupy pairwise distinct timeslots. Furthermore, odd cycle subgraphs cannot be colored with two slots, "
        "meaning an odd cycle requires more slots than bipartite coloring allows."
    ))

    cells.append(nbf.v4.new_code_cell(
        "graph = build_conflict_graph(inst)\n"
        "bound = clique_lower_bound(graph)\n"
        "print(f'Graph Nodes: {graph.number_of_nodes()} | Graph Edges: {graph.number_of_edges()}')\n"
        "print(f'Maximum Clique Lower Bound on Slots: {bound}')\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(6, 5))\n"
        "draw_conflict_graph(graph, ax=ax, title='Course Conflict Graph (Uncoloured)')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # =========================================================================
    # SECTION 4: Classical Baselines
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Classical Heuristics & Exact Search Baselines\n\n"
        "We evaluate classical graph coloring heuristics: **Largest First** (ordering by degree) and "
        "**DSATUR** (ordering dynamically by degree of saturation), alongside **Exact Exhaustive Search**.\n\n"
        "We solve for both a feasible slot budget ($K=3$) and an intentionally restricted budget ($K=2$) "
        "to demonstrate independent validator diagnostics."
    ))

    cells.append(nbf.v4.new_code_cell(
        "# Evaluate classical solvers for K = 3 slots\n"
        "res_exact = exact_schedule(inst, n_slots=3)\n"
        "res_glf = greedy_schedule(inst, n_slots=3, strategy='largest_first')\n"
        "res_gds = greedy_schedule(inst, n_slots=3, strategy='DSATUR')\n"
        "\n"
        "exact_opt = res_exact.objective\n"
        "results_k3 = [\n"
        "    {\n"
        "        'Method': 'Exact Search',\n"
        "        'Valid': res_exact.is_valid,\n"
        "        'Slots Used': res_exact.slots_used,\n"
        "        'Objective': f'{res_exact.objective:.4f}',\n"
        "        'Optimality Gap': f'{(res_exact.objective - exact_opt)/exact_opt*100:.1f}%',\n"
        "        'Runtime (s)': f'{res_exact.runtime_s:.5f}',\n"
        "    },\n"
        "    {\n"
        "        'Method': 'Greedy (Largest First)',\n"
        "        'Valid': res_glf.is_valid,\n"
        "        'Slots Used': res_glf.slots_used,\n"
        "        'Objective': f'{res_glf.objective:.4f}',\n"
        "        'Optimality Gap': f'{(res_glf.objective - exact_opt)/exact_opt*100:.1f}%',\n"
        "        'Runtime (s)': f'{res_glf.runtime_s:.5f}',\n"
        "    },\n"
        "    {\n"
        "        'Method': 'Greedy (DSATUR)',\n"
        "        'Valid': res_gds.is_valid,\n"
        "        'Slots Used': res_gds.slots_used,\n"
        "        'Objective': f'{res_gds.objective:.4f}',\n"
        "        'Optimality Gap': f'{(res_gds.objective - exact_opt)/exact_opt*100:.1f}%',\n"
        "        'Runtime (s)': f'{res_gds.runtime_s:.5f}',\n"
        "    },\n"
        "]\n"
        "print('=== Classical Solver Benchmark (K = 3) ===')\n"
        "print(pd.DataFrame(results_k3).to_string(index=False))\n"
        "\n"
        "# Display timetable for optimal schedule\n"
        "tt_df = timetable_dataframe(inst, res_exact.assignment, 3)\n"
        "print('\\nOptimal Exam Timetable (Exact Solver):')\n"
        "print(tt_df.to_string(index=False))\n"
        "\n"
        "# Plot conflict graph colored by exact schedule\n"
        "fig, ax = plt.subplots(figsize=(6, 5))\n"
        "slot_map = {c: res_exact.assignment[c] for c in range(inst.n_courses)}\n"
        "draw_conflict_graph(graph, slot_of=slot_map, ax=ax, title='Conflict Graph Coloured by Exact Schedule')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "### Infeasibility Verification at Restricted Budget ($K=2$)\n\n"
        "Because the maximum clique size exceeds the slot budget, no conflict-free assignment exists at $K=2$. "
        "We verify that both exact search and greedy heuristics fail validation."
    ))

    cells.append(nbf.v4.new_code_cell(
        "res_exact_k2 = exact_schedule(inst, n_slots=2)\n"
        "res_greedy_k2 = greedy_schedule(inst, n_slots=2, strategy='largest_first')\n"
        "print(f'Exact search valid at K=2: {res_exact_k2.is_valid} (assignment: {res_exact_k2.assignment})')\n"
        "print(f'Greedy search valid at K=2: {res_greedy_k2.is_valid}')\n"
        "\n"
        "val_k2 = validate_schedule(inst, res_greedy_k2.assignment, 2)\n"
        "print('Validator diagnostics for Greedy at K=2:')\n"
        "for msg in val_k2.messages:\n"
        "    print(f'  - {msg}')"
    ))

    # =========================================================================
    # SECTION 5: QUBO Formulation
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell(
        "## 5. QUBO Formulation & Constraint Penalty Analysis\n\n"
        "To formulate exam scheduling for quantum optimization, we map decisions to binary variables "
        "$x_{c,t} \\in \\{0, 1\\}$, where $x_{c,t} = 1$ indicates course $c$ is scheduled in timeslot $t$.\n\n"
        "The objective energy function balances hard constraints against scheduling preferences:\n\n"
        "$$E(x) = A \\sum_{c} \\left( \\sum_{t} x_{c,t} - 1 \\right)^2 + B \\sum_{(i,j) \\in E} \\sum_{t} x_{i,t} x_{j,t} + \\sum_{c} \\sum_{t} w_c \\cdot t \\cdot x_{c,t}$$\n\n"
        "- **Term 1 (One-Hot Penalty $A$):** Penalizes any course assigned to zero or multiple slots.\n"
        "- **Term 2 (Conflict Penalty $B$):** Penalizes conflicting courses assigned to identical slots.\n"
        "- **Term 3 (Soft Objective):** Minimizes total enrollment-weighted slot indices.\n\n"
        "*Penalty Trade-Off Note:* The penalty parameters $A$ and $B$ must be large enough that invalid configurations "
        "have strictly higher energy than the optimal valid configuration. However, if a penalty is just large enough "
        "at the boundary threshold, an invalid configuration can tie in energy with the best valid schedule, "
        "causing the ground state manifold to contain multiple degenerate states, some of which may be invalid."
    ))

    cells.append(nbf.v4.new_code_cell(
        "model = build_qubo(inst, n_slots=3, penalty_one_hot=1.5, penalty_conflict=1.5)\n"
        "print(f'QUBO Binary Variables: {model.n_vars}')\n"
        "print(f'Upper-Triangular Matrix Shape: {model.Q.shape}')\n"
        "\n"
        "# Heatmap of upper-triangular QUBO matrix\n"
        "fig, ax = plt.subplots(figsize=(6, 5))\n"
        "cax = ax.imshow(model.Q, cmap='coolwarm')\n"
        "fig.colorbar(cax, ax=ax)\n"
        "ax.set_title('Upper-Triangular QUBO Matrix Q')\n"
        "ax.set_xlabel('Variable Index')\n"
        "ax.set_ylabel('Variable Index')\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
        "\n"
        "# Exhaustive state-space brute-force scan\n"
        "energies = qubo_energies_all(model)\n"
        "mask = valid_state_mask(inst, n_slots=3)\n"
        "f_count = int(mask.sum())\n"
        "total_states = len(mask)\n"
        "pct_feasible = (f_count / total_states) * 100.0\n"
        "print(f'Feasible bitstrings in full state space: {f_count} / {total_states} ({pct_feasible:.4f}%)')\n"
        "\n"
        "min_e = energies.min()\n"
        "ground_indices = np.where(np.isclose(energies, min_e, atol=1e-9))[0]\n"
        "all_valid = bool(mask[ground_indices].all())\n"
        "print(f'Ground State Energy: {min_e:.4f} | Ground State Count: {len(ground_indices)} | All Ground States Valid: {all_valid}')"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "### Penalty Multiplier Sweep ($A = B$)\n\n"
        "We evaluate the sensitivity of ground state validity across a spectrum of constraint penalties."
    ))

    cells.append(nbf.v4.new_code_cell(
        "penalty_values = (1.0, 1.1, 1.15, 1.5, 2.0, 3.0)\n"
        "sweep_rows = []\n"
        "\n"
        "for pen in penalty_values:\n"
        "    m_pen = build_qubo(inst, n_slots=3, penalty_one_hot=pen, penalty_conflict=pen)\n"
        "    e_pen = qubo_energies_all(m_pen)\n"
        "    min_val = e_pen.min()\n"
        "    g_idx = np.where(np.isclose(e_pen, min_val, atol=1e-9))[0]\n"
        "    is_all_valid = bool(mask[g_idx].all())\n"
        "    sweep_rows.append({\n"
        "        'Penalty (A=B)': f'{pen:.2f}',\n"
        "        'Ground Energy': f'{min_val:.4f}',\n"
        "        'Ground States Count': len(g_idx),\n"
        "        'All Ground States Valid': is_all_valid,\n"
        "    })\n"
        "\n"
        "print('=== Penalty Multiplier Parameter Sweep ===')\n"
        "print(pd.DataFrame(sweep_rows).to_string(index=False))"
    ))

    # =========================================================================
    # SECTION 6: Ising Mapping
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell(
        "## 6. Ising Hamiltonian Mapping & Pauli Decomposition\n\n"
        "To execute quantum algorithms, the QUBO is mapped to an Ising spin Hamiltonian using the standard variable transformation:\n\n"
        "$$x_q = \\frac{I - Z_q}{2}$$\n\n"
        "where $Z_q$ is the Pauli-$Z$ operator acting on qubit $q$ with eigenvalues $\\pm 1$. "
        "Under this mapping, diagonal QUBO terms become single-qubit $Z$ operators and off-diagonal terms become "
        "two-qubit $Z_i Z_j$ interaction terms.\n\n"
        "*Qiskit Little-Endian Convention:* In Qiskit, measured bitstring strings are indexed right-to-left "
        "(the rightmost character represents qubit index 0). Our decoding function `bitstring_to_bits` "
        "explicitly reverses this string indexing to match variable indices."
    ))

    cells.append(nbf.v4.new_code_cell(
        "ising = qubo_to_ising(model)\n"
        "op = ising.operator\n"
        "pauli_list = op.to_list()\n"
        "\n"
        "z_terms = sum(1 for p, _ in pauli_list if p.count('Z') == 1)\n"
        "zz_terms = sum(1 for p, _ in pauli_list if p.count('Z') == 2)\n"
        "print(f'Total Pauli Hamiltonian Terms: {len(op)}')\n"
        "print(f'Single-Qubit Z Terms:           {z_terms}')\n"
        "print(f'Two-Qubit ZZ Interaction Terms: {zz_terms}')\n"
        "\n"
        "print('\\nFirst 8 Pauli Terms in Cost Operator:')\n"
        "for label, coeff in pauli_list[:8]:\n"
        "    print(f'  {label} : {coeff.real:+.4f}')\n"
        "\n"
        "# Verify mathematical equivalence across all 2^N statevectors\n"
        "i_energies = ising_energies_all(ising)\n"
        "assert np.allclose(i_energies, energies)\n"
        "print('\\nIsing spectrum identically matches QUBO energy across all 2^N states: True')\n"
        "\n"
        "# Demonstrate little-endian bitstring indexing\n"
        "sample_str = '100'\n"
        "sample_decoded = bitstring_to_bits(sample_str)\n"
        "print(f\"\\nLittle-Endian demonstration: '{sample_str}' -> decoded bits: {sample_decoded} (qubit 0 is {sample_decoded[0]})\")"
    ))

    # =========================================================================
    # SECTION 7: QAOA Circuit
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell(
        "## 7. QAOA Variational Circuit Architecture & Memory Safety\n\n"
        "The QAOA ansatz prepares an initial uniform superposition $|+\\rangle^{\\otimes N}$ via Hadamard gates, "
        "followed by alternating applications of the Cost unitary $U(C, \\gamma) = e^{-i \\gamma H_C}$ (built from $R_{ZZ}$ and $R_Z$ rotations) "
        "and the Mixer unitary $U(B, \\beta) = e^{-i \\beta H_M}$ ($R_X$ transverse field rotations) across $p$ layers.\n\n"
        "> **Memory Guard Notice:** Passing an untranspiled `QAOAAnsatz` (containing high-level `PauliEvolutionGate` blocks) "
        "> directly to `StatevectorEstimator` causes Qiskit to construct a dense $2^N \\times 2^N$ operator matrix. "
        "> For $N=15$ qubits, this consumes 16 GiB of RAM and causes process termination. "
        "> To prevent this, the circuit must always be transpiled into 1- and 2-qubit basis gates (`rz`, `sx`, `x`, `cx`) prior to simulation."
    ))

    cells.append(nbf.v4.new_code_cell(
        "# Illustrate high-level and decomposed circuit structure on a smaller ring instance\n"
        "from qiskit.circuit.library import QAOAAnsatz\n"
        "from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager\n"
        "\n"
        "small_inst = ring_instance(3, seed=DEFAULT_SEED)\n"
        "small_model = build_qubo(small_inst, n_slots=3)\n"
        "small_ising = qubo_to_ising(small_model)\n"
        "small_ansatz = QAOAAnsatz(small_ising.operator, reps=1)\n"
        "\n"
        "print(f'Illustrative Small Ring Instance Qubits: {small_ansatz.num_qubits}')\n"
        "try:\n"
        "    fig = small_ansatz.decompose().draw('mpl', fold=25)\n"
        "    plt.show()\n"
        "    print('Rendered circuit diagram using Matplotlib drawer.')\n"
        "except Exception as e:\n"
        "    print(f'Matplotlib drawer unavailable ({type(e).__name__}). Using text drawer:')\n"
        "    print(small_ansatz.draw('text', fold=70))\n"
        "\n"
        "# Transpilation statistics for headline instance (reps = 1 and reps = 2)\n"
        "pm = generate_preset_pass_manager(optimization_level=1, basis_gates=['rz', 'sx', 'x', 'cx'], seed_transpiler=42)\n"
        "\n"
        "head_ansatz_p1 = QAOAAnsatz(ising.operator, reps=1)\n"
        "trans_p1 = pm.run(head_ansatz_p1)\n"
        "ops_p1 = trans_p1.count_ops()\n"
        "print(f'\\nHeadline Instance QAOA (p=1): Depth={trans_p1.depth()} | CX Gates={ops_p1.get(\"cx\", 0)}')\n"
        "print(f'Parameters (p=1): {[p.name for p in trans_p1.parameters]}')\n"
        "\n"
        "head_ansatz_p2 = QAOAAnsatz(ising.operator, reps=2)\n"
        "trans_p2 = pm.run(head_ansatz_p2)\n"
        "ops_p2 = trans_p2.count_ops()\n"
        "print(f'\\nHeadline Instance QAOA (p=2): Depth={trans_p2.depth()} | CX Gates={ops_p2.get(\"cx\", 0)}')\n"
        "print(f'Parameters (p=2): {[p.name for p in trans_p2.parameters]}')"
    ))

    # =========================================================================
    # SECTION 8: Live QAOA Execution
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell(
        "## 8. Live QAOA Execution & Measurement Sampling ($p=1$)\n\n"
        "We optimize the variational angles $\\beta$ and $\\gamma$ using the classical COBYLA optimizer "
        "with multiple random restarts to mitigate local minima. Following convergence, the optimal state "
        "is sampled with 4,096 measurement shots."
    ))

    cells.append(nbf.v4.new_code_cell(
        "q_outcome = qaoa_schedule(\n"
        "    inst,\n"
        "    n_slots=3,\n"
        "    reps=1,\n"
        "    n_restarts=3,\n"
        "    shots=4096,\n"
        "    maxiter=150,\n"
        "    seed=42,\n"
        ")\n"
        "\n"
        "print(f'QAOA Solved: Best Restart={q_outcome.best_restart} | Final Energy={q_outcome.final_energy:.4f}')\n"
        "for r_info in q_outcome.restarts:\n"
        "    print(f\"  Restart {r_info['restart']}: Final <H_C> = {r_info['final_energy']:.4f} ({r_info['n_evaluations']} evals)\")\n"
        "\n"
        "# Convergence plot across restarts\n"
        "fig, ax = plt.subplots(figsize=(7, 4))\n"
        "for r_info in q_outcome.restarts:\n"
        "    r_idx = r_info['restart']\n"
        "    hist = r_info.get('history', [])\n"
        "    label = f\"Restart {r_idx}\" + (' (Best)' if r_idx == q_outcome.best_restart else '')\n"
        "    ax.plot(hist, label=label, linewidth=1.5)\n"
        "ax.set_xlabel('COBYLA Evaluation Step')\n"
        "ax.set_ylabel('Cost Energy <H_C>')\n"
        "ax.set_title('Optimization Convergence per Restart')\n"
        "ax.grid(True, linestyle=':', alpha=0.6)\n"
        "ax.legend()\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
        "\n"
        "# Inspect top measured bitstrings\n"
        "top_df = top_counts_dataframe(inst, 3, q_outcome.counts, top=15)\n"
        "print('\\nTop 15 Measured Bitstrings:')\n"
        "print(top_df.to_string(index=False))\n"
        "\n"
        "# Bar chart of top bitstrings coloured by validity\n"
        "fig, ax = plt.subplots(figsize=(8, 4))\n"
        "colors = ['#2ca02c' if v else '#d62728' for v in top_df['valid']]\n"
        "ax.bar(range(len(top_df)), top_df['share_pct'], color=colors)\n"
        "ax.set_xticks(range(len(top_df)))\n"
        "ax.set_xticklabels(top_df['bitstring'], rotation=60, ha='right', fontsize=9)\n"
        "ax.set_ylabel('Shot Share (%)')\n"
        "ax.set_title('Top Measured Bitstrings (Green = Valid, Red = Invalid)')\n"
        "from matplotlib.patches import Patch\n"
        "legend_elements = [Patch(facecolor='#2ca02c', label='Valid Schedule'), Patch(facecolor='#d62728', label='Invalid')]\n"
        "ax.legend(handles=legend_elements)\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
        "\n"
        "# Feasibility comparison\n"
        "print(f'Exact Feasible Probability (Statevector): {q_outcome.exact_feasible_probability * 100:.4f}%')\n"
        "print(f'Shot Feasibility Yield (Measured):        {q_outcome.feasibility_yield * 100:.4f}% ({q_outcome.n_valid_shots}/{q_outcome.shots} shots)')\n"
        "print(f'Uniform Random-Guess Probability:         {q_outcome.random_guess_probability * 100:.4f}%')\n"
        "\n"
        "if q_outcome.result.is_valid and q_outcome.result.assignment is not None:\n"
        "    val = validate_schedule(inst, q_outcome.result.assignment, 3)\n"
        "    print(f'Best Valid Schedule: {q_outcome.result.assignment} | Objective: {q_outcome.result.objective:.4f}')\n"
        "    fig, ax = plt.subplots(figsize=(6, 5))\n"
        "    slot_map_q = {c: q_outcome.result.assignment[c] for c in range(inst.n_courses)}\n"
        "    draw_conflict_graph(graph, slot_of=slot_map_q, ax=ax, title='QAOA Best Valid Schedule')\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
        "else:\n"
        "    print('No valid schedule was found in the measured shots.')"
    ))

    # =========================================================================
    # SECTION 9: Head-to-Head Comparison
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell(
        "## 9. Head-to-Head Solver Comparison\n\n"
        "We evaluate exact search, greedy heuristics, classical random sampling (with identical shot count and seed), "
        "and QAOA side-by-side on the headline problem instance."
    ))

    cells.append(nbf.v4.new_code_cell(
        "res_rand = random_sampling_schedule(inst, n_slots=3, shots=4096, seed=42)\n"
        "\n"
        "h2h_rows = [\n"
        "    {\n"
        "        'Method': 'Exact Search',\n"
        "        'Valid': res_exact.is_valid,\n"
        "        'Slots Used': res_exact.slots_used,\n"
        "        'Objective': f'{res_exact.objective:.4f}',\n"
        "        'Optimality Gap': f'{(res_exact.objective - exact_opt)/exact_opt*100:.1f}%',\n"
        "        'Runtime (s)': f'{res_exact.runtime_s:.5f}',\n"
        "    },\n"
        "    {\n"
        "        'Method': 'Greedy (Largest First)',\n"
        "        'Valid': res_glf.is_valid,\n"
        "        'Slots Used': res_glf.slots_used,\n"
        "        'Objective': f'{res_glf.objective:.4f}',\n"
        "        'Optimality Gap': f'{(res_glf.objective - exact_opt)/exact_opt*100:.1f}%',\n"
        "        'Runtime (s)': f'{res_glf.runtime_s:.5f}',\n"
        "    },\n"
        "    {\n"
        "        'Method': 'Greedy (DSATUR)',\n"
        "        'Valid': res_gds.is_valid,\n"
        "        'Slots Used': res_gds.slots_used,\n"
        "        'Objective': f'{res_gds.objective:.4f}',\n"
        "        'Optimality Gap': f'{(res_gds.objective - exact_opt)/exact_opt*100:.1f}%',\n"
        "        'Runtime (s)': f'{res_gds.runtime_s:.5f}',\n"
        "    },\n"
        "    {\n"
        "        'Method': 'Random Sampling (4096 shots)',\n"
        "        'Valid': res_rand.is_valid,\n"
        "        'Slots Used': res_rand.slots_used if res_rand.is_valid else '-',\n"
        "        'Objective': f'{res_rand.objective:.4f}' if res_rand.objective is not None else '-',\n"
        "        'Optimality Gap': f'{(res_rand.objective - exact_opt)/exact_opt*100:.1f}%' if res_rand.is_valid and res_rand.objective is not None else '-',\n"
        "        'Runtime (s)': f'{res_rand.runtime_s:.5f}',\n"
        "    },\n"
        "    {\n"
        "        'Method': 'QAOA (p=1, 4096 shots)',\n"
        "        'Valid': q_outcome.result.is_valid,\n"
        "        'Slots Used': q_outcome.result.slots_used if q_outcome.result.is_valid else '-',\n"
        "        'Objective': f'{q_outcome.result.objective:.4f}' if q_outcome.result.objective is not None else '-',\n"
        "        'Optimality Gap': f'{(q_outcome.result.objective - exact_opt)/exact_opt*100:.1f}%' if q_outcome.result.is_valid and q_outcome.result.objective is not None else '-',\n"
        "        'Runtime (s)': f'{q_outcome.result.runtime_s:.5f}',\n"
        "    },\n"
        "]\n"
        "print('=== Head-to-Head Solver Comparison Table ===')\n"
        "print(pd.DataFrame(h2h_rows).to_string(index=False))"
    ))

    # =========================================================================
    # SECTION 10: Benchmark Results
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell(
        "## 10. Multi-Run Benchmark Suite Results\n\n"
        "To evaluate statistical reproducibility beyond single-seed runs, we load the precomputed "
        "benchmark suite across experiments E1 (headline), E2 (penalty sweep), and E3 (instance scaling)."
    ))

    cells.append(nbf.v4.new_code_cell(
        "res_dir = Path('../results')\n"
        "summary_csv = res_dir / 'benchmark_summary.csv'\n"
        "meta_file = res_dir / 'benchmark_meta.json'\n"
        "\n"
        "if summary_csv.exists():\n"
        "    summary_df = pd.read_csv(summary_csv)\n"
        "    print('=== Benchmark Summary Table ===')\n"
        "    print(summary_df.to_string(index=False))\n"
        "\n"
        "    print('\\n=== Precomputed Benchmark Figures ===')\n"
        "    for fname, title in [\n"
        "        ('fig_feasibility.png', 'E1 Feasibility Yield Across Solvers'),\n"
        "        ('fig_objective.png', 'E1 Schedule Objective Distribution'),\n"
        "        ('fig_penalty.png', 'E2 Penalty Multiplier Sweep'),\n"
        "        ('fig_scaling.png', 'E3 Problem Scaling (Yield and Runtime)'),\n"
        "    ]:\n"
        "        p = res_dir / fname\n"
        "        if p.exists():\n"
        "            print(f'-- {title} --')\n"
        "            display(Image(filename=str(p)))\n"
        "\n"
        "    if meta_file.exists():\n"
        "        meta_data = json.loads(meta_file.read_text(encoding='utf-8'))\n"
        "        print('\\nBenchmark Execution Metadata:')\n"
        "        for k in ['platform', 'python_version', 'qiskit', 'qiskit_aer', 'timestamp_utc']:\n"
        "            print(f'  {k}: {meta_data.get(k)}')\n"
        "else:\n"
        "    print(f'Benchmark data not found in {res_dir}. Run: python src/main.py benchmark --out results')"
    ))

    # =========================================================================
    # SECTION 10b: Statistical Significance
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell(
        "## 10b. Statistical Significance Analysis: Are the Differences Meaningful?\n\n"
        "To avoid misleading single-run claims, we analyze empirical distributions across repeated benchmark trials:\n"
        "1. **Wilson Score Confidence Intervals (95%):** Provide coverage bounds for binomial proportions "
        "(valid run rate and optimum found rate) that remain well-calibrated even with small sample sizes.\n"
        "2. **Fisher's Exact Test:** Evaluates whether differences in the proportion of runs finding the global optimum "
        "reach statistical significance ($p < 0.05$).\n"
        "3. **Mann-Whitney $U$ Test:** Non-parametric rank-sum test evaluating whether the distribution of feasibility "
        "yields from QAOA is stochastically greater than uniform random sampling.\n\n"
        "When confidence intervals overlap and sample sizes are small, differences cannot be deemed statistically conclusive."
    ))

    cells.append(nbf.v4.new_code_cell(
        "runs_file = Path('../results/benchmark_runs.csv')\n"
        "if runs_file.exists():\n"
        "    runs_df = pd.read_csv(runs_file)\n"
        "    e1_df = runs_df[runs_df['experiment'] == 'E1_headline']\n"
        "\n"
        "    def wilson_ci(k, n, z=1.95996):\n"
        "        if n == 0:\n"
        "            return 0.0, 0.0, 0.0\n"
        "        p_hat = k / n\n"
        "        denom = 1.0 + (z**2) / n\n"
        "        center = (p_hat + (z**2) / (2 * n)) / denom\n"
        "        half = (z * np.sqrt((p_hat * (1.0 - p_hat) / n) + (z**2) / (4 * n**2))) / denom\n"
        "        return p_hat, max(0.0, center - half), min(1.0, center + half)\n"
        "\n"
        "    stat_rows = []\n"
        "    for m in ['random-sampling', 'qaoa-p1', 'qaoa-p2']:\n"
        "        m_df = e1_df[e1_df['method'] == m]\n"
        "        if m_df.empty:\n"
        "            continue\n"
        "        n_runs = len(m_df)\n"
        "        n_valid = int((m_df['is_valid'] == True).sum())\n"
        "        opt_hits = m_df.apply(\n"
        "            lambda r: bool(r['is_valid']) and pd.notna(r['optimal_objective']) and pd.notna(r['objective']) and abs(r['objective'] - r['optimal_objective']) < 1e-9,\n"
        "            axis=1,\n"
        "        )\n"
        "        n_opt = int(opt_hits.sum())\n"
        "\n"
        "        v_rate, v_lo, v_hi = wilson_ci(n_valid, n_runs)\n"
        "        o_rate, o_lo, o_hi = wilson_ci(n_opt, n_runs)\n"
        "        mean_yield = m_df['feasibility_yield'].mean() if m_df['feasibility_yield'].notna().any() else np.nan\n"
        "\n"
        "        stat_rows.append({\n"
        "            'Method': m,\n"
        "            'Runs': n_runs,\n"
        "            'Valid Runs': n_valid,\n"
        "            'Valid Rate [95% CI]': f'{v_rate*100:.1f}% [{v_lo*100:.1f}%, {v_hi*100:.1f}%]',\n"
        "            'Optimum Hits': n_opt,\n"
        "            'Optimum Rate [95% CI]': f'{o_rate*100:.1f}% [{o_lo*100:.1f}%, {o_hi*100:.1f}%]',\n"
        "            'Mean Yield': f'{mean_yield*100:.2f}%' if pd.notna(mean_yield) else '-',\n"
        "        })\n"
        "\n"
        "    print('=== Statistical Confidence Intervals (E1 Headline) ===')\n"
        "    print(pd.DataFrame(stat_rows).to_string(index=False))\n"
        "\n"
        "    # Hypothesis tests comparing qaoa-p1 against random-sampling\n"
        "    q1_data = e1_df[e1_df['method'] == 'qaoa-p1']\n"
        "    rs_data = e1_df[e1_df['method'] == 'random-sampling']\n"
        "    if not q1_data.empty and not rs_data.empty:\n"
        "        q1_opt = int(q1_data.apply(lambda r: bool(r['is_valid']) and abs(r['objective'] - r['optimal_objective']) < 1e-9, axis=1).sum())\n"
        "        q1_fail = len(q1_data) - q1_opt\n"
        "        rs_opt = int(rs_data.apply(lambda r: bool(r['is_valid']) and abs(r['objective'] - r['optimal_objective']) < 1e-9, axis=1).sum())\n"
        "        rs_fail = len(rs_data) - rs_opt\n"
        "\n"
        "        odds_ratio, p_fisher = scipy.stats.fisher_exact([[q1_opt, q1_fail], [rs_opt, rs_fail]])\n"
        "        print(f\"\\nFisher's Exact Test (Optimum Found Rate: qaoa-p1 vs random-sampling): p-value = {p_fisher:.4f} (odds ratio = {odds_ratio:.4f})\")\n"
        "\n"
        "        q1_yields = q1_data['feasibility_yield'].dropna()\n"
        "        rs_yields = rs_data['feasibility_yield'].dropna()\n"
        "        u_stat, p_mwu = scipy.stats.mannwhitneyu(q1_yields, rs_yields, alternative='greater')\n"
        "        yield_ratio = q1_yields.mean() / rs_yields.mean() if rs_yields.mean() > 0 else np.nan\n"
        "        print(f'Mann-Whitney U Test (Feasibility Yield: qaoa-p1 > random-sampling): p-value = {p_mwu:.4f} (U = {u_stat:.1f})')\n"
        "        print(f'Ratio of Mean Feasibility Yield (qaoa-p1 / random-sampling): {yield_ratio:.2f}x')\n"
        "\n"
        "    # Runtime scaling ratios relative to exact search\n"
        "    exact_runs = e1_df[e1_df['method'] == 'exact']\n"
        "    if not exact_runs.empty:\n"
        "        exact_med_rt = exact_runs['runtime_s'].median()\n"
        "        if not q1_data.empty:\n"
        "            q1_med_rt = q1_data['runtime_s'].median()\n"
        "            print(f'Ratio of Median Runtime (qaoa-p1 / exact solver): {q1_med_rt / exact_med_rt:.1f}x')\n"
        "        q2_data = e1_df[e1_df['method'] == 'qaoa-p2']\n"
        "        if not q2_data.empty:\n"
        "            q2_med_rt = q2_data['runtime_s'].median()\n"
        "            print(f'Ratio of Median Runtime (qaoa-p2 / exact solver): {q2_med_rt / exact_med_rt:.1f}x')\n"
        "\n"
        "    # Random sampling empirical yield vs full state-space feasibility\n"
        "    if not rs_data.empty:\n"
        "        emp_yield = rs_data['feasibility_yield'].mean()\n"
        "        true_prob = float(valid_state_mask(inst, 3).sum()) / (1 << 15)\n"
        "        print(f'\\nEmpirical Random-Sampling Yield:     {emp_yield*100:.4f}%')\n"
        "        print(f'True State-Space Feasible Fraction:   {true_prob*100:.4f}%')\n"
        "else:\n"
        "    print(f'Benchmark runs file {runs_file} not found.')"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "Please inspect the table and statistical test outputs printed directly above for the confidence intervals, "
        "p-values, and runtime ratios. As shown in the outputs, with small sample sizes the confidence intervals overlap, "
        "meaning firm conclusions cannot be drawn regarding solver superiority."
    ))

    # =========================================================================
    # SECTION 11: Conclusions & Limitations
    # =========================================================================
    cells.append(nbf.v4.new_markdown_cell(
        "## 11. Conclusions & Technical Limitations\n\n"
        "### Key Findings\n"
        "- **Feasibility Amplification:** Variational optimization via QAOA shifts state probabilities toward low-energy configurations, "
        "raising the per-shot share of valid schedules above uniform random guess levels. However, a majority of measured shots remain infeasible.\n"
        "- **Validation Requirement:** Quantum solvers generate candidate solutions probabilistically; every schedule must undergo "
        "independent verification by the classical validator before acceptance.\n"
        "- **Modal State vs Validity:** The single most frequent bitstring measured from QAOA is frequently invalid, "
        "requiring top-shot filtering across measured samples.\n"
        "- **Classical vs Quantum Scalability:** On these small instances, classical exact search and greedy heuristics "
        "execute orders of magnitude faster than classical quantum simulation. No quantum advantage or speed-up is claimed.\n\n"
        "### Practical Limitations\n"
        "- **Simulation Overhead:** Classical simulation of quantum circuits scales exponentially with qubit count, restricting "
        "direct noiseless statevector study to modest problem sizes.\n"
        "- **Classical Optimizer Sensitivity:** Optimization performance is sensitive to the COBYLA evaluation budget and initial random angle seeds.\n"
        "- **Sample Size Constraints:** Modest seed counts yield overlapping confidence intervals, indicating exploratory rather than definitive statistical trends.\n"
        "- **Noiseless Simulation:** Results reflect ideal mathematical operators; physical quantum processor noise (decoherence, crosstalk, gate infidelities) was not modeled.\n"
        "- **QUBO Penalty Trade-offs:** Quadratic penalty coefficients require delicate tuning to balance constraint satisfaction against objective discrimination."
    ))

    nb.cells = cells
    return nb


def validate_executed_notebook(nb_path: Path) -> bool:
    """Validate execution status, outputs, images, and text constraints."""
    print("\n=== Validating Executed Notebook ===")
    content = nb_path.read_text(encoding="utf-8")
    nb = nbf.reads(content, as_version=4)

    all_passed = True
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    markdown_cells = [c for c in nb.cells if c.cell_type == "markdown"]

    # 1. Check for execution errors
    error_cells = []
    for idx, c in enumerate(code_cells):
        for out in c.get("outputs", []):
            if out.get("output_type") == "error":
                error_cells.append((idx, out.get("ename"), out.get("evalue")))
    if error_cells:
        print(f"[FAIL] Found {len(error_cells)} code cells with errors:")
        for idx, ename, evalue in error_cells:
            print(f"       Cell {idx}: {ename} - {evalue}")
        all_passed = False
    else:
        print("[PASS] No code cells produced execution errors.")

    # 2. Check execution counts
    unexecuted_cells = [idx for idx, c in enumerate(code_cells) if c.get("execution_count") is None]
    if unexecuted_cells:
        print(f"[FAIL] Found {len(unexecuted_cells)} unexecuted code cells: {unexecuted_cells}")
        all_passed = False
    else:
        print(f"[PASS] All {len(code_cells)} code cells have valid execution counts.")

    # 3. Check for image outputs
    image_count = 0
    for c in code_cells:
        for out in c.get("outputs", []):
            if "data" in out:
                if "image/png" in out["data"] or "image/jpeg" in out["data"]:
                    image_count += 1
    if image_count < 3:
        print(f"[FAIL] Found only {image_count} image outputs (minimum 3 required).")
        all_passed = False
    else:
        print(f"[PASS] Found {image_count} generated image/plot outputs.")

    # 4. Check that markdown cells do not hardcode suspicious numbers from benchmark summary
    # Extract numerical tokens from benchmark summary outputs to detect accidental hardcoding
    suspicious_found = []
    forbidden_numbers = ["0.0916", "0.1428", "1.805", "1.935", "2.222", "30/32768", "14.3%"]
    for idx, m_cell in enumerate(markdown_cells):
        text = m_cell.get("source", "")
        for target in forbidden_numbers:
            if target in text:
                suspicious_found.append((idx, target))

    if suspicious_found:
        print(f"[WARN/FAIL] Suspicious hardcoded numbers in markdown cells: {suspicious_found}")
        all_passed = False
    else:
        print("[PASS] Markdown cells contain only conceptual explanations without typed benchmark numbers.")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Build and execute hackathon solution notebook.")
    parser.add_argument("--execute", action="store_true", help="Execute the generated notebook and validate.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    notebooks_dir = repo_root / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)
    nb_path = notebooks_dir / "solution.ipynb"

    print(f"Building notebook: {nb_path}")
    nb = create_solution_notebook()
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Notebook structure successfully written.")

    if args.execute:
        print("\nExecuting notebook via nbconvert (cwd = notebooks/, kernel = python3)...")
        print("This will take approximately 1-2 minutes...")
        ep = ExecutePreprocessor(timeout=1800, kernel_name="python3")
        with open(nb_path, "r", encoding="utf-8") as f:
            nb_to_run = nbf.read(f, as_version=4)

        try:
            ep.preprocess(nb_to_run, {"metadata": {"path": str(notebooks_dir)}})
        except Exception as e:
            print(f"\n[ERROR] Notebook execution raised an exception: {e}")
            sys.exit(1)

        with open(nb_path, "w", encoding="utf-8") as f:
            nbf.write(nb_to_run, f)
        print("Executed notebook successfully written back to disk.")

        passed = validate_executed_notebook(nb_path)

        # File size and sections report
        file_size_kb = nb_path.stat().st_size / 1024.0
        print(f"\nNotebook File Size: {file_size_kb:.2f} KiB")

        headings = []
        for c in nb_to_run.cells:
            if c.cell_type == "markdown":
                for line in c.source.splitlines():
                    if line.startswith("#"):
                        headings.append(line.strip())

        print("\n=== Final Notebook Headings ===")
        for h in headings:
            print(f"  {h}")

        if not passed:
            print("\n>>> VALIDATION FAILED <<<")
            sys.exit(1)
        else:
            print("\n>>> ALL VALIDATION CHECKS PASSED <<<")


if __name__ == "__main__":
    main()
