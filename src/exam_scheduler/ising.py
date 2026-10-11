"""Ising Hamiltonian mapping from QUBO formulations using modern Qiskit 2.5.2."""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from qiskit.quantum_info import SparsePauliOp

from exam_scheduler.qubo import QuboModel


@dataclass(frozen=True)
class IsingModel:
    """Represents an Ising Hamiltonian cost operator H = sum h_i Z_i + sum J_ij Z_i Z_j + offset * I."""

    operator: SparsePauliOp
    h: np.ndarray
    J: Dict[Tuple[int, int], float]
    offset: float
    n_qubits: int


def qubo_to_ising(model: QuboModel) -> IsingModel:
    """Convert an upper-triangular QUBO model into an Ising Hamiltonian using x = (I - Z)/2."""
    N = model.n_vars
    Q = model.Q

    # J_ij = Q_ij / 4 for i < j
    J: Dict[Tuple[int, int], float] = {}
    for i in range(N):
        for j in range(i + 1, N):
            val = float(Q[i, j] / 4.0)
            if abs(val) > 1e-12:
                J[(i, j)] = val

    # h_i = -Q_ii / 2 - (1/4) * sum_{j != i} Q_ij (counting Q_ij whether i is row or col)
    h = np.zeros(N, dtype=float)
    for i in range(N):
        cross_sum = sum(Q[min(i, j), max(i, j)] for j in range(N) if j != i)
        h[i] = float(-Q[i, i] / 2.0 - 0.25 * cross_sum)

    # offset = constant + sum_i Q_ii/2 + sum_{i<j} Q_ij/4
    linear_sum = sum(Q[i, i] for i in range(N))
    pairwise_sum = sum(Q[i, j] for i in range(N) for j in range(i + 1, N))
    offset = float(model.constant + 0.5 * linear_sum + 0.25 * pairwise_sum)

    # Construct SparsePauliOp terms
    pauli_list = []
    for i in range(N):
        if abs(h[i]) > 1e-12:
            pauli_list.append(("Z", [i], h[i]))

    for (i, j), coeff in J.items():
        if abs(coeff) > 1e-12:
            pauli_list.append(("ZZ", [i, j], coeff))

    if abs(offset) > 1e-12:
        pauli_list.append(("", [], offset))

    operator = SparsePauliOp.from_sparse_list(pauli_list, num_qubits=N).simplify()

    return IsingModel(
        operator=operator,
        h=h,
        J=J,
        offset=offset,
        n_qubits=N,
    )


def ising_energies_all(ising: IsingModel) -> np.ndarray:
    """Extract diagonal energy eigenvalues of the Ising Hamiltonian across all 2^N computational states."""
    if ising.n_qubits > 18:
        raise ValueError(
            f"Number of qubits ({ising.n_qubits}) exceeds 18 (too large for all-states diagonal extraction)."
        )

    # NEVER build a dense matrix; use sparse=True and read diagonal
    return ising.operator.to_matrix(sparse=True).diagonal().real
