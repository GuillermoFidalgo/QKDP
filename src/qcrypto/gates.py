from __future__ import annotations

from typing import Dict

import numpy as np
import numpy.typing as npt

Pauli: Dict[str, npt.NDArray[np.complex128]] = {
    "x": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}

H_gate: npt.NDArray[np.complex128] = (1 / np.sqrt(2)) * np.array(
    [[1, 1], [1, -1]], dtype=np.complex128
)


def Phase_shift(phase: float) -> npt.NDArray[np.complex128]:
    """Generates a phase shift gate for a given phase.

    Args:
        phase (float): The phase angle in radians.

    Returns:
        NDArray: A 2x2 numpy array representing the phase shift gate.
    """
    phase_shift_gate = np.array([1, 0], [0, np.e ** (1j * phase)])
    return phase_shift_gate


def tensor_power(
    gate: npt.NDArray[np.complex128], N: int
) -> npt.NDArray[np.complex128]:
    """Computes the tensor power of a 2x2 gate matrix.

    Args:
        gate (NDArray): A 2x2 numpy array representing a quantum gate.
        N (int): The power to which the gate matrix is to be raised, tensor-wise.

    Returns:
        NDArray: A numpy array representing the N-th tensor power of the gate.
    """
    result = gate
    for _ in range(N - 1):
        result = np.kron(result, gate)
    return result
