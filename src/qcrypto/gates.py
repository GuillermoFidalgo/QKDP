import numpy as np

Pauli_gates = [
    np.array([[0, 1], [1, 0]], dtype=np.complex_),
    np.array([[0, -1j], [1j, 0]], dtype=np.complex_),
    np.array([[1, 0], [0, -1]], dtype=np.complex_),
]

H_gate = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex_)


def tensor_power(gate, N):
    result = gate
    for _ in range(N - 1):
        result = np.kron(result, gate)
    return result


def Gate(state, gate):
    N = int(np.log2(len(state)))
    gate = tensor_power(gate, N)
    new_state = np.dot(gate, state)
    return new_state


def Hadamard(state):
    N = int(np.log2(len(state)))
    gate = tensor_power(H_gate, N)
    new_state = np.dot(gate, state)
    return new_state


def Pauli(state, i=0):
    # N = int(np.log2(len(state)))
    # gate = Pauli_gates[i]
    # new_state = np.dot(gate, state)
    # return new_state
    pass


def Rot(state, thetax=0, thetay=0, thetaz=0):
    #     Rx = np.array([np.cos(thetax/2), -1j * np.sin(thetax/2)], [-1j * np.sin(thetax/2), np.cos(thetax/2)])
    pass
