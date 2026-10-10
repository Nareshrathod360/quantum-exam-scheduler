# Track 2 Reference Guide: Open-Domain Optimization & Real-World Solvers
**Qiskit Fall Fest 2026 — IIIT Kottayam**

Welcome to **Track 2**! Your goal is to identify a compelling real-world dilemma, translate it into a mathematical optimization problem, and solve it using **Qiskit** (via the Quantum Approximate Optimization Algorithm — QAOA, or variational quantum circuits).

---

## 1. The 4-Step Quantum Optimization Pipeline

Any combinatorial optimization problem can be tackled using quantum computing through these four steps:

```
┌──────────────────────────────────────────────┐
│  1. Problem Formulation (Binary Variables)   │  x_i ∈ {0, 1}
└──────────────────────┬───────────────────────┘
                       ▼
┌──────────────────────────────────────────────┐
│  2. Cost Function & Constraints Formulation  │  C(x) = Objective(x) + λ · Penalty(x)
└──────────────────────┬───────────────────────┘
                       ▼
┌──────────────────────────────────────────────┐
│  3. Map to Ising Hamiltonian                 │  x_i → (I - Z_i) / 2
└──────────────────────┬───────────────────────┘
                       ▼
┌──────────────────────────────────────────────┐
│  4. Solve with QAOA & Compare Baseline       │  QAOAAnsatz + Optimizer vs. Classical
└──────────────────────────────────────────────┘
```

### Step 1: Define Your Binary Decision Variables
Identify what choices must be made. For example:
- In project selection: $x_i = 1$ if project $i$ is funded, $0$ otherwise.
- In exam scheduling: $x_{i, t} = 1$ if exam $i$ is held at time slot $t$.
- In graph partitioning: $x_i = 1$ if node $i$ is in Group A, $0$ if in Group B.

### Step 2: Formulate Objective & Penalty Terms
Express your goal as a function to minimize (or maximize) over bitstrings $x = (x_0, x_1, \dots, x_{n-1})$. If you have constraints (e.g., "select exactly $K$ items"), add them as quadratic penalties:
$$C(x) = \text{Cost}(x) + \lambda \cdot \left(\sum_{i} x_i - K\right)^2$$
*(where $\lambda > 0$ is a penalty multiplier that punishes infeasible solutions).*

### Step 3: Map to Pauli-$Z$ Operators (Ising Model)
Quantum computers solve optimization by mapping binary variables $x_i \in \{0, 1\}$ to Pauli-$Z$ spin operators $Z_i \in \{+1, -1\}$ via:
$$x_i = \frac{I - Z_i}{2}$$
When expanded, pairwise terms $x_i x_j$ turn into $Z_i Z_j$ two-qubit interactions, resulting in a diagonal Cost Hamiltonian:
$$H_C = \sum_{i < j} J_{ij} Z_i Z_j + \sum_i h_i Z_i + \text{offset} \cdot I$$
In Qiskit, this is constructed using `SparsePauliOp.from_list([("ZZ", J), ...])`.

### Step 4: Run QAOA & Evaluate
Using `QAOAAnsatz(cost_operator=H_C, reps=p)`:
- Choose depth $p$ (start with $p=1$ or $p=2$).
- Use a classical optimizer (like COBYLA) with `StatevectorSampler` or `StatevectorEstimator`.
- Measure the most probable sampled bitstrings.

---

## 2. Real-World Problem Ideas & Formulations

You are encouraged to choose your own unique dilemma! Here are popular real-world problems:

### Idea 1: Smart Campus Exam Scheduling (Graph Coloring / Conflict Graph)
* **Scenario:** University courses share enrolled students and cannot have exams at the same time.
* **Formulation:** Represent courses as nodes and student conflicts as edges. Partition the graph into time slots to minimize schedule collisions.

### Idea 2: Green Energy Project Portfolio Selection (Knapsack with Budget)
* **Scenario:** An organization has 5 candidate solar/wind/biomass projects, each with expected return $r_i$ and carbon offset $c_i$, under a total budget $K$.
* **Formulation:** Maximize return and carbon offset while penalizing solutions where $\sum x_i \neq K$.

### Idea 3: Delivery Drone Routing & Hub Placement (Max-Cut / Facility Location)
* **Scenario:** Dividing delivery zones between two regional drone hubs to minimize inter-hub travel while balancing delivery load.
* **Formulation:** Max-Cut or graph bipartition on a weighted graph of customer locations.

### Idea 4: Hospital Shift Scheduling (Exact Cover)
* **Scenario:** Assigning nurses or doctors across critical shifts ensuring no shift is left uncovered and no worker exceeds duty hours.
* **Formulation:** Quadratic Unconstrained Binary Optimization (QUBO) with coverage penalties.

---

## 3. How to Benchmark Against Classical Baselines

A crucial part of your project and presentation is demonstrating how the quantum solver compares to classical methods:
1. **Brute-Force Optimum (for small instances $N \le 10$):**
   * Enumerate all $2^N$ bitstrings classically and find the exact optimal cost $C_{\text{best}}$.
2. **Approximation Ratio:**
   $$\text{Approximation Ratio} = \frac{C_{\text{QAOA}}}{C_{\text{optimal}}}$$
   Study how the approximation ratio improves as you increase QAOA circuit depth $p=1, 2, 3$.
3. **Feasibility Yield:**
   * What percentage of shots returned by QAOA satisfy your budget/hard constraints? How does increasing penalty strength $\lambda$ change this?

---

## 4. Starter Notebook & References

* **Starter Notebook:** See [`track2_optimization_QAOA.ipynb`](./track2_optimization_QAOA.ipynb) in this folder.  
  * Originally developed for BasQ Qiskit Fall Fest 2026 by Benjamin Tirado.
  * Walks through the complete, working, end-to-end workflow of solving Max-Cut on a 5-node graph using modern Qiskit (`QAOAAnsatz`, `SparsePauliOp`, `StatevectorEstimator`, `StatevectorSampler`, COBYLA, and brute-force evaluation).
  * You can adapt this exact notebook structure to your chosen real-world problem and cost Hamiltonian!

### Recommended External References:
* [IBM Quantum Documentation: End-to-End QAOA Tutorial](https://quantum.cloud.ibm.com/docs/tutorials/quantum-approximate-optimization-algorithm)
* [Qiskit Optimization Documentation](https://qiskit-community.github.io/qiskit-optimization/)
* [Rustworkx Documentation (for graph creation and drawing)](https://www.rustworkx.org/)
* [Farhi, Goldstone, Gutmann (2014): A Quantum Approximate Optimization Algorithm](https://arxiv.org/abs/1411.4028)
