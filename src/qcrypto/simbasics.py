import numpy as np
import numpy.typing as npt
import dataclasses
from qcrypto.gates import tensor_power
from typing import Union, Dict
from abc import ABC, abstractmethod


class QState(ABC):

    @abstractmethod
    def measure(self):
        pass

    @abstractmethod
    def measure_all(self):
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
    _state: np.ndarray = None
    num_qubits: int = 10
    init_method: str = "zeros"

    def __post_init__(self):
        if self._state is None:
            if self.init_method == "zeros":
                self._state = np.zeros((self.num_qubits, 2), dtype=np.complex_)
                self._state[:, 0] = 1
            elif self.init_method == "random":
                self._state = np.random.random(
                    (self.num_qubits, 2)
                ) + 1j * np.random.random((self.num_qubits, 2))
                self._normalize_state()
            else:
                raise ValueError("Invalid initialization method.")
        else:
            if self._state.shape != (self.num_qubits, 2):
                raise ValueError(
                    "State vector shape not appropriate for the number of qubits."
                )
            self._normalize_state()
            self.num_qubits = len(self._state)
        self._state = np.asarray(self._state, dtype=np.complex_)

    def measure(self, qubit_idx: int = None) -> int:
        """
        Simulates the measurement of a single qubit. As a result, the state of said qubit is collapsed depending on the result.

        Args:
            qubit_idx (int): Index of the qubit to be measured

        Returns:
            Outcome of the measurement. Also collapses the state of the qubit.

        """
        if qubit_idx is not None:
            probs_0, probs_1 = self._calculate_measurement_probs(qubit_idx)
            outcome = np.random.choice([0, 1], p=[probs_0, probs_1])
            self._update_state_post_measurement(qubit_idx, outcome)
            return outcome
        else:
            raise ValueError("qubit_idx not specified")

    def measure_all(self, *kwargs) -> npt.NDArray[int]:
        """
        Measures all of the qubits in sequential order.

        Args:
            n/a

        Returns:
            Numpy array containing the outcome of all of the measurments.

        """
        outcome = []
        for qubit_idx in range(self.num_qubits):
            outcome.append(self.measure(qubit_idx=qubit_idx))
        return np.array(outcome)

    def apply_gate(
        self, gate: npt.NDArray[np.complex_], qubit_idx: Union[int, list] = None
    ):
        """
        Applies a given gate to a subset of qubits, modifying the quantum state.

        Args:
            gate (np.array): Gate to be applied. Represented as a numpy matrix
            qubit_idx (int, list): Index/Indices of qubit/qubits which will be transformed by gate

        Returns:
            None
        """
        if qubit_idx is not None:
            self._state[qubit_idx] = np.dot(gate, self._state[qubit_idx])
        else:
            reshaped_states = self._state.reshape(self._state.shape[0], 2, 1)
            new_states = np.dot(gate, reshaped_states)
            self._state = new_states.reshape(self._state.shape)
        self._normalize_state()

    def _calculate_measurement_probs(self, qubit_idx: int) -> tuple[int, ...]:
        """
        Computes the probability of measuring qubit_idx to be in state 0 or 1 in whatever base its in.

        Args:
            qubit_idx (int): Index of qubit

        Returns:
            Probabilities of obtaining 0 and 1 if qubit were to be measured
        """

        prob_0 = np.abs(self._state[qubit_idx, 0]) ** 2
        prob_1 = np.abs(self._state[qubit_idx, 1]) ** 2
        return prob_0, prob_1

    def _update_state_post_measurement(self, qubit_idx: int, outcome: int):
        """
        Updates the quantum state of a qubit by projecting it unto a given outcome state.

        Args:
            qubit_idx (int): Index of qubit
            outcome (int): Outcome unto which the qubit will be projected

        Returns:
            None
        """

        if outcome == 0:
            self._state[qubit_idx] = np.array([1, 0], dtype=np.complex_)
        else:
            self._state[qubit_idx] = np.array([0, 1], dtype=np.complex_)

    def _normalize_state(self):
        """
        Normalizes the quantum state.

        Args:
            n/a

        Returns:
            None
        """

        norms = np.linalg.norm(self._state, axis=1)
        norms = norms.reshape(-1, 1)
        self._state = self._state / norms


@dataclasses.dataclass
class QstateEnt(QState):
    """
    Representes the state of a set of N qubits which might be entangled.
    """

    _state: np.ndarray = None
    num_qubits: int = 10
    init_method: str = "zeros"

    def __post_init__(self):
        if self._state is None:
            self._auto_init(self.init_method)
        else:
            if len(self._state) != 2**self.num_qubits:
                raise ValueError(
                    "State vector size not appropriate for the number of qubits specified."
                )
            self._normalize_state()
            self._state = np.asarray(self._state, dtype=np.complex_)

    def _auto_init(self, init_method: str):
        """
        Initializes the quantum state of the system depending on the initialziation method chosen by the user.

        Args:
            init_method (str): Initialziation method

        Returns:
            None

        """
        if self.init_method not in ["zeros", "random"]:
            raise ValueError("Invalid initialization method.")
        if self.init_method == "zeros":
            self._state = np.zeros(2**self.num_qubits, dtype=np.complex_)
            self._state[0] = 1
        elif self.init_method == "random":
            self._state = np.random.rand(2**self.num_qubits) + 1j * np.random.rand(
                2**self.num_qubits
            )
            self._normalize_state()

    def measure(self, qubit_idx: int):
        """
        Measure ths qubit_idx'th qubit, calculating the probability of and, with these, returning 1 or 0.

        Args:
            qubit_idx (int): Index identifying the qubit to be measured

        Returns:
            Outcome of the measurement, either 0 or 1
        """

        probs_0, probs_1 = self._calculate_measurement_probs(qubit_idx)
        outcome = np.random.choice([0, 1], p=[probs_0, probs_1])
        self._update_state_post_measurement(qubit_idx, outcome)
        return outcome

    def measure_all(self, order):
        """
        Measures all of the qubits

        Args:
            order (str): Specifies the order in which the qubits will be measured.
                "simult" = all qubits measured simultaneously
                "sequential" = qubits measured in sequential order (first 0, second 1, etc.)

        Returns:
            Outcome of the measurements done. Array of 0's and 1's equal in length to the number of qubits in the system.
        """
        if order == "simult":
            outcome = np.random.choice(
                np.arange(len(self._state)), p=self._calculate_measurement_probs()
            )
            self._state.fill(0 + 0j)
            self._state[outcome] = 1 + 0j
            outcome_arr = np.array(
                list(
                    "0" * (self.num_qubits - len(bin(outcome)[2:])) + bin(outcome)[2:]
                ),
                dtype=int,
            )
            return outcome_arr
        elif order == "sequential":
            outcome = []
            for i in range(self.num_qubits):
                outcome.append(self.measure(qubit_idx=i))
            self._update_state_post_measurement
            outcome_arr = np.array(outcome)
            return outcome_arr
        else:
            raise ValueError("Order specified not valid.")

    def _calculate_measurement_probs(self, qubit_idx: int = None):
        """
        From the probability amplitude, computes the probability that a measurement of a given qubit will give 0 or 1.

        Args:
            qubits_idx (int): Index identifying the qubit to be measured

        Returns:
            Probability of measuring qubit in position qubit_idx to be measured to be 0 or to be 1
        """
        if qubit_idx is None:
            outcome_probs = np.abs(self._state) ** 2
            return outcome_probs
        else:
            if qubit_idx >= self.num_qubits or qubit_idx < 0:
                raise ValueError(
                    "Invalid qubit index. Make sure it is between 1 and num_qubits - 1"
                )

            prob_0 = 0
            prob_1 = 0
            for idx, prob_amp in enumerate(self._state):
                if (idx >> qubit_idx) & 1 == 0:
                    prob_0 += np.abs(prob_amp) ** 2
                else:
                    prob_1 += np.abs(prob_amp) ** 2
            return prob_0, prob_1

    def _update_state_post_measurement(self, qubit_idx, outcome):
        """
        Updates the quantum state post-measurement, effectively collapsing the wave function. based on the result obtained.

        Args:
            qubit_idx (int): Index identifying qubit which was measured
            outcome (int): Result of the measurement. Either 0 or 1

        Returns:
            None

        """
        new_state = []
        for idx, amplitude in enumerate(self._state):
            if ((idx >> qubit_idx) & 1) == outcome:
                new_state.append(amplitude)
            else:
                new_state.append(0)

        self._state = np.array(new_state)
        self._normalize_state()

    def _normalize_state(self):
        """
        Updates the state of the system by normalizing its states.

        Args:
            n/a

        Retuns:
            n/a
        """
        self._state /= np.linalg.norm(self._state)

    def apply_gate(self, gate: npt.NDArray[np.complex_]):
        """
        Applies quantum gate to the system of qubits.

        Args:
            gate (np.NDarray): Gate to be applied to the to the system

        Returns:
            n/a

        """
        N = int(np.log2(len(self._state)))
        gate = tensor_power(gate, N)
        self._state = np.dot(gate, self._state)
        self._normalize_state()

    def __str__(self):
        return str(self._state)

    def __repr__(self):
        return str(self._state)


@dataclasses.dataclass
class Agent:
    num_priv_qubits: int = None
    qstates: Dict[str, npt.NDArray[np.complex_]] = dataclasses.field(
        default_factory=lambda: {"private": None, "public": None}
    )
    keys: Dict[str, npt.NDArray[np.int_]] = dataclasses.field(
        default_factory=lambda: {"private": None, "public": None}
    )
    priv_qstates: Union[QstateEnt, QstateUnEnt] = None
    init_method: str = "random"
    priv_qbittype: str = None

    def __post_init__(self):
        self.qstates = {"private": None, "public": None}

        if self.priv_qstates is None and self.priv_qbittype == "entangled":
            self.set_qstate(
                QstateEnt(
                    num_qubits=self.num_priv_qubits, init_method=self.init_method
                ),
                "private",
            )
        elif self.priv_qstates is None and self.priv_qbittype == "unentangled":
            self.set_qstate(
                QstateUnEnt(
                    num_qubits=self.num_priv_qubits, init_method=self.init_method
                ),
                "private",
            )

    def set_qstate(
        self, qstate: Union[QstateEnt, QstateUnEnt], qstate_type: str
    ) -> None:
        """
        Sets a given qstate as either a private or public qubit of the Agent

        Args:
            qstate (QstateEnt or QstateUnEnt): Quantum state of a private or public system of qubits
            qstate_type (str): Whether the given qstate is to be private or public

        Returns:
            None
        """
        if not isinstance(qstate, QstateUnEnt) and not isinstance(qstate, QstateEnt):
            raise ValueError("Wrong type given for system state.")

        self.qstates[qstate_type] = qstate

    def measure(self, qstate_type, qubit_idx=None):
        if qstate_type not in self.qstates.keys():
            raise ValueError("Not valid qstate type.")

        outcome = self.qstates[qstate_type].measure(qubit_idx=qubit_idx)
        return outcome

    def measure_all(self, qstate_type, order=None):
        if qstate_type not in self.qstates.keys():
            raise ValueError("Invalid qstate type")

        outcome = self.qstates[qstate_type].measure_all(order)
        return outcome

    def apply_gate(self, gate, qstate_type):
        if qstate_type not in self.qstates.keys():
            raise ValueError("Invalid qstate type")

        self.qstates[qstate_type].apply_gate(gate)

    # def measure_all(self, qstate_type, order="simult"):
    #     if qstate_type not in self.qstates.keys():
    #         raise ValueError("Invalid qstate type")

    #     self.qstates[qstate_type].measure_all(order=order)

    def get_key(self, qstate_type, order=None):
        outcome = self.measure_all(qstate_type=qstate_type, order=order)
        self.keys[qstate_type] = outcome
        return outcome
