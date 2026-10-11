"""QUBO formulation and energy evaluation module for exam scheduling."""

from dataclasses import dataclass
from typing import List, Optional, Sequence
import numpy as np

from exam_scheduler.conflict_graph import build_conflict_graph, conflict_edges
from exam_scheduler.data import ExamInstance


@dataclass(frozen=True)
class QuboModel:
    """Represents a Quadratic Unconstrained Binary Optimization model for exam scheduling."""

    Q: np.ndarray  # Upper-triangular matrix of shape (N, N)
    constant: float
    n_courses: int
    n_slots: int
    penalty_one_hot: float
    penalty_conflict: float

    @property
    def n_vars(self) -> int:
        """Total number of binary decision variables N = n_courses * n_slots."""
        return self.n_courses * self.n_slots

    def var_index(self, c: int, t: int) -> int:
        """Map (course, slot) pair to flattened variable index q = c * K + t."""
        return c * self.n_slots + t


def build_qubo(
    instance: ExamInstance,
    n_slots: int,
    penalty_one_hot: float = 1.5,
    penalty_conflict: float = 1.5,
) -> QuboModel:
    """Build the upper-triangular QUBO matrix and constant offset for the exam instance."""
    if n_slots < 1:
        raise ValueError(f"n_slots must be >= 1, got {n_slots}.")
    if penalty_one_hot <= 0 or penalty_conflict <= 0:
        raise ValueError("Penalties must be strictly positive.")

    n = instance.n_courses
    K = n_slots
    N = n * K
    Q = np.zeros((N, N), dtype=float)

    A = float(penalty_one_hot)
    B = float(penalty_conflict)
    weights = instance.weights()
    constant = float(A * n)

    # 1. Diagonal terms: linear one-hot penalty + objective cost
    # (sum_t x_ct - 1)^2 expands to 1 - sum_t x_ct + 2 * sum_{t<u} x_ct * x_cu
    for c in range(n):
        for t in range(K):
            q = c * K + t
            Q[q, q] = -A + weights[c] * t

    # 2. Pairwise one-hot terms: 2*A for same course in different slots (t < u)
    for c in range(n):
        for t in range(K):
            for u in range(t + 1, K):
                q1 = c * K + t
                q2 = c * K + u
                Q[q1, q2] = 2.0 * A

    # 3. Conflict terms: B for conflicting courses sharing slot t
    graph = build_conflict_graph(instance)
    for i, j in conflict_edges(graph):
        for t in range(K):
            a = i * K + t
            b = j * K + t
            # Since i < j, a < b, so this is strictly upper-triangular
            Q[a, b] = B

    return QuboModel(
        Q=Q,
        constant=constant,
        n_courses=n,
        n_slots=K,
        penalty_one_hot=A,
        penalty_conflict=B,
    )


def qubo_energy(model: QuboModel, bits: Sequence[int]) -> float:
    """Evaluate QUBO energy of a bit vector using the upper-triangular matrix form."""
    x = np.asarray(bits, dtype=float)
    return float(model.constant + x @ model.Q @ x)


def direct_energy(
    instance: ExamInstance,
    n_slots: int,
    bits: Sequence[int],
    penalty_one_hot: float,
    penalty_conflict: float,
) -> float:
    """Evaluate QUBO energy directly from problem definitions without matrix representation."""
    x = np.asarray(bits, dtype=float)
    n = instance.n_courses
    K = n_slots
    weights = instance.weights()
    graph = build_conflict_graph(instance)
    edges = conflict_edges(graph)

    # One-hot penalty: A * sum_c (sum_t x[c,t] - 1)^2
    one_hot_term = 0.0
    for c in range(n):
        sum_t = sum(x[c * K + t] for t in range(K))
        one_hot_term += (sum_t - 1.0) ** 2

    # Conflict penalty: B * sum_{(i,j) in edges} sum_t x[i,t] * x[j,t]
    conflict_term = 0.0
    for i, j in edges:
        for t in range(K):
            conflict_term += x[i * K + t] * x[j * K + t]

    # Objective: sum_c sum_t w_c * t * x[c,t]
    objective_term = 0.0
    for c in range(n):
        for t in range(K):
            objective_term += weights[c] * t * x[c * K + t]

    return float(
        penalty_one_hot * one_hot_term
        + penalty_conflict * conflict_term
        + objective_term
    )


def qubo_energies_all(model: QuboModel) -> np.ndarray:
    """Compute QUBO energies for all 2^N states simultaneously using vectorized numpy."""
    if model.n_vars > 18:
        raise ValueError(
            f"Number of variables ({model.n_vars}) exceeds 18 (too large for all-states vectorization)."
        )

    num_states = 1 << model.n_vars
    s = np.arange(num_states, dtype=np.int64)
    q_indices = np.arange(model.n_vars, dtype=np.int64)
    # Bit q of integer state s is (s >> q) & 1
    B = ((s[:, None] >> q_indices) & 1).astype(np.float64)

    # E(x) = constant + sum_j ((B @ Q) * B)_j
    energies = model.constant + np.sum((B @ model.Q) * B, axis=1)
    return energies


def decode_bits(model: QuboModel, bits: Sequence[int]) -> List[Optional[int]]:
    """Decode bit vector to schedule assignment: slot t if exactly one 1, else None."""
    assignment: List[Optional[int]] = []
    K = model.n_slots
    for c in range(model.n_courses):
        course_bits = [bits[c * K + t] for t in range(K)]
        ones = [t for t, val in enumerate(course_bits) if val == 1]
        if len(ones) == 1:
            assignment.append(ones[0])
        else:
            assignment.append(None)
    return assignment


def bits_from_assignment(
    model: QuboModel, assignment: Sequence[Optional[int]]
) -> List[int]:
    """Convert a schedule assignment to an N-element 0/1 bit vector."""
    bits = [0] * model.n_vars
    K = model.n_slots
    for c in range(model.n_courses):
        slot = assignment[c]
        if slot is not None and 0 <= slot < K:
            bits[c * K + slot] = 1
    return bits


def bitstring_to_bits(bitstring: str) -> List[int]:
    """Convert Qiskit little-endian bitstring to bit vector (rightmost char = qubit 0)."""
    return [int(ch) for ch in reversed(bitstring.strip())]
