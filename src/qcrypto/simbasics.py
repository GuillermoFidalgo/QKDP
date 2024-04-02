import numpy as np
import dataclasses
from qcrypto.gates import *
from typing import List, Union
from abc import ABC, abstractmethod


class QState(ABC):

    @abstractmethod
    def measure(self, qubit_idx=None):
        pass

    @abstractmethod
    def measure_all(self, order="simult"):
        pass

    @abstractmethod
    def apply_gate(self, gate):
        pass

    @abstractmethod
    def _calculate_measurement_probs(self, qubit_idx):
        pass

    @abstractmethod
    def _update_state_post_measurement(self, qubit_idx, outcome):
        pass

    @abstractmethod
    def _normalize_state(self):
        pass


@dataclasses.dataclass
class QstateUnEnt(QState):
    num_qubits: int
    state: np.ndarray = None
    init_method: str = "zeros"

    def __post_init__(self):
        if self.state is None:
            if self.init_method == "zeros":
                self.state = np.zeros((self.num_qubits, 2), dtype=np.complex_)
                self.state[:, 0] = 1
            elif self.init_method == "random":
                self.state = np.random.random(
                    (self.num_qubits, 2)
                ) + 1j * np.random.random((self.num_qubits, 2))
                self._normalize_state()
            else:
                raise ValueError("Invalid initialization method.")
        else:
            if len(self.state) != 2**self.num_qubits:
                raise ValueError(
                    "State vector size not appropriate for the number of qubits."
                )
            self._normalize_state()
            self.num_qubits = len(self.state)
        self.state = np.asarray(self.state, dtype=np.complex_)

    def measure(self, qubit_idx=None):
        if qubit_idx is not None:
            probs_0, probs_1 = self._calculate_measurement_probs(qubit_idx)
            outcome = np.random.choice([0, 1], p=[probs_0, probs_1])
            self._update_state_post_measurement(qubit_idx, outcome)
            return outcome
        else:
            raise ValueError("qubit_idx not specified")

    def measure_all(self, order="simult"):
        outcome = []
        for qubit_idx in range(self.num_qubits):
            outcome.append(self.measure(qubit_idx=qubit_idx))
        return np.array(outcome)

    def apply_gate(self, gate, qubit_idx=None):
        if qubit_idx is not None:
            self.state[qubit_idx] = np.dot(gate, self.state[qubit_idx])
        else:
            reshaped_states = self.state.reshape(self.state.shape[0], 2, 1)
            new_states = np.dot(gate, reshaped_states)
            self.state = new_states.reshape(self.state.shape)

    def _calculate_measurement_probs(self, qubit_idx):
        prob_0 = np.abs(self.state[qubit_idx, 0]) ** 2
        prob_1 = np.abs(self.state[qubit_idx, 1]) ** 2
        return prob_0, prob_1

    def _update_state_post_measurement(self, qubit_idx, outcome):
        if outcome == 0:
            self.state[qubit_idx] = np.array([1, 0], dtype=np.complex_)
        else:
            self.state[qubit_idx] = np.array([0, 1], dtype=np.complex_)

    def _normalize_state(self):
        norms = np.linalg.norm(self.state, axis=1)
        norms = norms.reshape(-1, 1)
        self.state = self.state / norms


@dataclasses.dataclass
class QstateEnt(QState):
    """
    Representes the state of a set of N qubits which might be entangled.
    """

    num_qubits: int
    state: np.ndarray = None
    init_method: str = "zeros"

    def __post_init__(self):
        if self.state is None:
            if self.init_method == "zeros":
                self.state = np.zeros(2**self.num_qubits, dtype=np.complex_)
                self.state[0] = 1
            elif self.init_method == "random":
                self.state = np.random.rand(2**self.num_qubits) + 1j * np.random.rand(
                    2**self.num_qubits
                )
                self._normalize_state()
            else:
                raise ValueError("Invalid initialization method.")
        else:
            if len(self.state) != 2**self.num_qubits:
                raise ValueError(
                    "State vector size not appropriate for the number of qubits."
                )
            self._normalize_state()
            self.num_qubits = len(self.state)
        self.state = np.asarray(self.state, dtype=np.complex_)

    def measure(self, qubit_idx=None):
        if qubit_idx is not None:
            probs_0, probs_1 = self._calculate_measurement_probs(qubit_idx)
            outcome = np.random.choice([0, 1], p=[probs_0, probs_1])
            self._update_state_post_measurement(qubit_idx, outcome)
            return outcome
        else:
            raise ValueError("qubit_idx not specified")

    def measure_all(self, order="simult"):
        if order == "simult":
            outcome = np.random.choice(
                np.arange(2**self.num_qubits), p=np.abs(self.state) ** 2
            )
            self.state.fill(0 + 0j)
            self.state[outcome] = 1 + 0j
            binoutcome = np.array(
                list("0" * (self.num_qubits - len(bin(outcome))) + bin(outcome)),
                dtype=int,
            )
            return binoutcome
        elif order == "sequential":
            outcome = []
            for i in range(self.num_qubits):
                outcome.append(self.measure(qubit_idx=i))
            self._update_state_post_measurement
            return np.array(outcome)
        else:
            raise ValueError("Order specified not valid.")

    def _calculate_measurement_probs(self, qubit_idx):
        prob_0 = 0
        prob_1 = 0
        for idx, prob_amp in enumerate(self.state):
            if (idx >> qubit_idx) & 1 == 0:
                prob_0 += np.abs(prob_amp) ** 2
            else:
                prob_1 += np.abs(prob_amp) ** 2
        return prob_0, prob_1

    def _update_state_post_measurement(self, qubit_idx, outcome):
        new_state = []
        for idx, amplitude in enumerate(self.state):
            if ((idx >> qubit_idx) & 1) == outcome:
                new_state.append(amplitude)
            else:
                new_state.append(0)

        self.state = np.array(new_state)
        self._normalize_state()

    def _normalize_state(self):
        self.state /= np.linalg.norm(self.state)

    def apply_gate(self, gate):
        N = int(np.log2(len(self.state)))
        gate = tensor_power(gate, N)
        self.state = np.dot(gate, self.state)

    def __str__(self):
        return str(self.state)

    def __repr__(self):
        return str(self.state)


@dataclasses.dataclass
class Agent:
    num_priv_qubits: int = None
    priv_qstates: Union[QstateEnt, QstateUnEnt] = None
    pblc_qstates: Union[QstateEnt, QstateUnEnt] = None
    priv_key: List = dataclasses.field(default_factory=lambda: [])
    pblc_key: List = dataclasses.field(default_factory=lambda: [])
    init_method: str = "random"
    priv_qbittype: str = None
    pblc_qbittype: str = None

    def __post_init__(self):
        if self.priv_qstates is None and self.priv_qbittype == "entangled":
            self.priv_qstates = QstateEnt(
                num_qubits=self.num_priv_qubits, init_method=self.init_method
            )
        elif self.priv_qstates is None and self.priv_qbittype == "unentangled":
            self.priv_qstates = QstateUnEnt(
                num_qubits=self.num_priv_qubits, init_method=self.init_method
            )

    def measure(self, qstate_type, qubit_idx=None):
        if qstate_type == "private" and self.priv_qstates is not None:
            return self.priv_qstates.measure(qubit_idx=qubit_idx)
        elif qstate_type == "public" and self.pblc_qstates is not None:
            return self.pblc_qstates.measure(qubit_idx=qubit_idx)
        else:
            raise ValueError("Invalid Qstate type or no Qstate object.")

    def apply_gate(self, gate, qstate_type):
        if qstate_type == "private" and self.priv_qstates is not None:
            self.priv_qstates.apply_gate(gate)
        elif qstate_type == "public" and self.pblc_qstates is not None:
            self.pblc_qstates.apply_gate(gate)
        else:
            raise ValueError("Invalid Qstate type or no Qstate object.")

    def measure_all(self, qstate_type, order="simult"):
        if qstate_type == "private" and self.priv_qstates is not None:
            return self.priv_qstates.measure_all(order=order)
        elif qstate_type == "public" and self.pblc_qstates is not None:
            return self.pblc_qstates.measure_all(order=order)
        else:
            raise ValueError("Invalid Qstate type or no Qstate object.")

    def get_key(self, qstate_type):
        return self.measure(qstate_type=qstate_type)


####
@dataclasses.dataclass
class Qubit:
    """
    Data class which represents the qubit and its quantum state.
    Quantum state is given by:
    |\psi> = \cos(\frac12 \theta) |0> + e^{i\phi}  \sin(\frac12 \theta) |1>
    """

    def __init__(self, theta, base, phi=0):
        self.base = base
        self.theta = theta  # Meaningless once projected. FIX
        self.phi = phi  # Same with this. FIX
        self.state = np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])

    def __str__(self):
        return "{} |0> + {} |1>".format(self.state[0], self.state[1])

    def __repr__(self):
        return "{} |0> + {} |1>".format(self.state[0], self.state[1])

    def normalize(self):
        """
        Nomalizes the qubit's quantum state
        """
        norm = np.sqrt(np.dot(np.conjugate(self.state), self.state))
        self.state /= norm

    def get_probs(self):
        """
        Gives the probabilities of getting 0 or 1 when measuring the qubit
        """

        prob0 = (np.conjugate(self.state[0]) * self.state[0]).real
        prob1 = (np.conjugate(self.state[1]) * self.state[1]).real
        return prob0, prob1

    def copy(self):
        return Qubit(theta=self.theta, base=self.base, phi=self.phi)


# class Agent:
#     """
#     Class representing the players in a quantum cryptography simulation (e.g. Alice, Eve or Bob)
#     """

#     def __init__(self, numqubits=None, basis_selection=None, key=None, message=None):
#         """
#         Args:
#             numqubits: Number of qubits to be used
#             basis_selection: Set of bases to choose from when generating the set of numqubits qubits.
#                 If not given, will default to 0 and pi/2.
#         """
#         self.qubits = []
#         self.measurements = []
#         self.key = []

#         if numqubits is not None or numqubits != 0:
#             if basis_selection is None:
#                 self.basis_selection = [0, np.pi / 4]
#             else:
#                 self.basis_selection = basis_selection

#             qubitbases = np.random.choice(self.basis_selection, numqubits)

#             for basis in qubitbases:
#                 # theta = 0 -> qubit = 0 state
#                 # theta = pi -> qubit = 1 state
#                 qubit = Qubit(np.random.choice([0, np.pi]), basis, phi=0)
#                 self.qubits.append(qubit)

#     def project(self, qubit, base):
#         """
#         Projects the given qubit unto the agent's measuring basis
#         """
#         newqubit = qubit
#         base_delta = base - newqubit.base

#         # Projection oprator
#         proj_mtrx = np.array(
#             [
#                 [np.cos(base_delta), -1 * np.sin(base_delta)],
#                 [np.sin(base_delta), np.cos(base_delta)],
#             ]
#         )

#         newqubit.state = np.dot(proj_mtrx, newqubit.state)
#         newqubit.normalize()

#         newqubit.base = base

#         return newqubit

#     def measure(self, qubit, base, rtrn_result=True):
#         """
#         Gives the result of a sample measurement of the qubit.
#         """
#         self.project(qubit, base)
#         measurement_result = np.random.choice((0, 1), p=qubit.get_probs())
#         self.measurements.append(measurement_result)
#         if rtrn_result:
#             return measurement_result

#     def send_quantum(self, recipient, recipient_bases):
#         # Recipient obtains the qubits from self and measures using given basis
#         for qubit, base in zip(self.qubits, recipient_bases):
#             recipient.measure(qubit.copy(), base)

#         # Recipient construct qubits based on these measurements and on their own basis selection
#         recipient.qubits = []
#         for base, measurement in zip(recipient_bases, recipient.measurements):
#             received_qubit = Qubit(measurement * np.pi, base)  # measurement = 0 or 1
#             recipient.qubits.append(received_qubit)
#         recipient.qubits = np.array(recipient.qubits)

#     def get_key(self, bases):
#         for qubit, base in zip(self.qubits, bases):
#             measurement = self.measure(qubit, base)
#             self.key.append(measurement)
#         return self.key

#     def send_classic(self, receiverAgent, bits):
#         pass

#     def genkey(self, numqubits, numcheckbits):
#         """
#         Generates an array of qubits
#         """
#         pass
