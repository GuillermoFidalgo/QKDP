from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt

from qcrypto.gates import tensor_power


class QState(ABC):
    @abstractmethod
    def measure(self, qubit_idx: int) -> int: ...

    @abstractmethod
    def measure_all(self, *args: Any) -> npt.NDArray[np.int_]: ...

    @abstractmethod
    def apply_gate(
        self,
        gate: npt.NDArray[np.complex128],
        qubit_idx: int | npt.NDArray[np.int_] | list[int] | None = None,
    ) -> None: ...

    @abstractmethod
    def _calculate_measurement_probs(self, qubit_idx: int) -> tuple[float, float]: ...

    @abstractmethod
    def _update_state_post_measurement(self, qubit_idx: int, outcome: int) -> None: ...

    @abstractmethod
    def _normalize_state(self) -> None: ...


@dataclasses.dataclass
class QstateUnEnt(QState):
    _state: npt.NDArray[np.complex128] | None = None
    num_qubits: int = 10
    init_method: str = "zeros"
    rng: np.random.Generator = np.random.default_rng

    def __post_init__(self) -> None:
        if self._state is None:
            if self.init_method == "zeros":
                self._state = np.zeros((self.num_qubits, 2), dtype=np.complex128)
                self._state[:, 0] = 1
            elif self.init_method == "random":
                self._state = self.rng().random(
                    (self.num_qubits, 2)
                ) + 1j * self.rng().random((self.num_qubits, 2))
                self._normalize_state()
            else:
                msg = f"{self.init_method = } Invalid initialization method."
                raise ValueError(msg)
        else:
            if self._state.shape != (self.num_qubits, 2):
                msg = f"Got {self._state.shape = }. State vector shape not appropriate for the number of qubits."
                raise ValueError(msg)
            self._normalize_state()
            self.num_qubits = len(self._state)
        self._state = np.asarray(self._state, dtype=np.complex128)

    def measure(self, qubit_idx: int) -> int:
        """
        Simulates the measurement of a single qubit. As a result, the state of said qubit is collapsed depending on the result.

        Args:
            qubit_idx (int): Index of the qubit to be measured

        Returns:
            Outcome of the measurement. Also collapses the state of the qubit.

        """

        probs_0, probs_1 = self._calculate_measurement_probs(qubit_idx)
        outcome: int = self.rng().choice([0, 1], p=[probs_0, probs_1])
        self._update_state_post_measurement(qubit_idx, outcome)
        return outcome

    def measure_all(self, *args: Any) -> npt.NDArray[np.int_]:
        """
        Measures all of the qubits in sequential order.

        Args:
            n/a

        Returns:
            Numpy array containing the outcome of all of the measurements.

        """
        outcome = []
        for qubit_idx in range(self.num_qubits):
            outcome.append(self.measure(qubit_idx=qubit_idx))
        return np.array(outcome)

    def apply_gate(
        self,
        gate: npt.NDArray[np.complex128],
        qubit_idx: int | npt.NDArray[np.int_] | list[int] | None = None,
    ) -> None:
        """
        Applies a given gate to a subset of qubits, modifying the quantum state.

        Args:
            gate (np.array): Gate to be applied. Represented as a numpy matrix
            qubit_idx (int, list): Index/Indices of qubit/qubits which will be transformed by gate

        Returns:
            None
        """
        if self._state is None:
            msg = "Error applying gate. State has not been initialized."
            raise ValueError(msg)

        if qubit_idx is not None:
            if isinstance(qubit_idx, (list, np.ndarray)):
                for idx in qubit_idx:
                    self._state[idx] = np.dot(gate, self._state[idx])
            else:
                self._state[qubit_idx] = np.dot(gate, self._state[qubit_idx])
        else:
            self._state = np.dot(self._state, gate.T)
        self._normalize_state()

    def _calculate_measurement_probs(self, qubit_idx: int) -> tuple[float, float]:
        """
        Computes the probability of measuring qubit_idx to be in state 0 or 1 in whatever base its in.

        Args:
            qubit_idx (int): Index of qubit

        Returns:
            Probabilities of obtaining 0 and 1 if qubit were to be measured
        """

        if self._state is None:
            msg = "Unable to compute measurement probabilities. State has not been initialized."
            raise ValueError(msg)

        prob_0 = np.abs(self._state[qubit_idx, 0]) ** 2
        prob_1 = np.abs(self._state[qubit_idx, 1]) ** 2
        return prob_0, prob_1

    def _update_state_post_measurement(self, qubit_idx: int, outcome: int) -> None:
        """
        Updates the quantum state of a qubit by projecting it unto a given outcome state.

        Args:
            qubit_idx (int): Index of qubit
            outcome (int): Outcome unto which the qubit will be projected

        Returns:
            None
        """

        if self._state is None:
            msg = "Unable to update state. State has not been initialized."
            raise ValueError(msg)

        if outcome == 0:
            self._state[qubit_idx] = np.array([1, 0], dtype=np.complex128)
        else:
            self._state[qubit_idx] = np.array([0, 1], dtype=np.complex128)

    def _normalize_state(self) -> None:
        """
        Normalizes the quantum state.

        Args:
            n/a

        Returns:
            None
        """

        if self._state is None:
            msg = "Error normalizing state. State has not been initialized."
            raise ValueError(msg)

        norms = np.linalg.norm(self._state, axis=1)
        norms = norms.reshape(-1, 1)
        self._state = self._state / norms

    def __str__(self):
        return str(self._state)

    def __repr__(self):
        return str(self._state)


@dataclasses.dataclass
class QstateEnt(QState):
    """
    Represents the state of a set of N qubits which might be entangled.
    """

    _state: npt.NDArray[np.complex128] | None = None
    num_qubits: int = 10
    init_method: str = "zeros"
    rng: np.random.Generator = np.random.default_rng

    def __post_init__(self) -> None:

        if self._state is None:
            self._auto_init()
        else:
            if len(self._state) != 2**self.num_qubits:
                msg = f"Got {len(self._state) =}. State vector size not appropriate for the number of qubits specified. Expected {2**self.num_qubits = }"
                raise ValueError(msg)
            self._normalize_state()
            self._state = np.asarray(self._state, dtype=np.complex128)

    def _auto_init(self) -> None:
        """
        Initializes the quantum state of the system depending on the initialziation method chosen by the user.

        Args:
            init_method (str): Initialziation method

        Returns:
            None

        """

        if self.init_method not in ["zeros", "random"]:
            msg = f"""
                    Invalid initialization method. Got {self.init_method}
                    expected one of either ['zeros', 'random']
                    """
            raise ValueError(msg)
        if self.init_method == "zeros":
            self._state = np.zeros(2**self.num_qubits, dtype=np.complex128)
            self._state[0] = 1
        elif self.init_method == "random":
            self._state = self.rng().rand(2**self.num_qubits) + 1j * self.rng().rand(
                2**self.num_qubits
            )
            self._normalize_state()

    def measure(self, qubit_idx: int) -> int:
        """
        Measure the qubit_idx'th qubit, calculating the probability of each outcome and returning said outcome.

        Args:
            qubit_idx (int): Index identifying the qubit to be measured

        Returns:
            Outcome of the measurement, either 0 or 1
        """

        probs_0, probs_1 = self._calculate_measurement_probs(qubit_idx)
        outcome: int = self.rng().choice([0, 1], p=[probs_0, probs_1])
        self._update_state_post_measurement(qubit_idx, outcome)
        return outcome

    def measure_all(self, order: str) -> npt.NDArray[np.int_]:
        """
        Measures all of the qubits

        Args:
            order (str): Specifies the order in which the qubits will be measured.
                "simult" = all qubits measured simultaneously
                "sequential" = qubits measured in sequential order (first 0, second 1, etc.)

        Returns:
            Outcome of the measurements done. Array of 0's and 1's equal in length to the number of qubits in the system.
        """

        if self._state is None:
            msg = "Error performing measurements. State is not initialized."
            raise ValueError(msg)

        if order == "simult":
            outcome = self.rng().choice(
                np.arange(len(self._state)), p=self._calculate_measurement_probs()
            )
            self._state.fill(0 + 0j)
            self._state[outcome] = 1 + 0j

            outcome_arr = np.array(
                list(
                    "0" * (self.num_qubits - len(bin(outcome)[2:])) + bin(outcome)[2:]
                ),
                dtype=np.int_,
            )
        elif order == "sequential":
            outcome = []
            for i in range(self.num_qubits):
                outcome.append(self.measure(qubit_idx=i))
            outcome_arr = np.array(outcome)

        else:
            msg = f"Got {order =}. Expected one of either 'simult' or 'sequential'. Order specified not valid."
            raise ValueError(msg)

        return outcome_arr

    def _calculate_measurement_probs(
        self, qubit_idx: int | None = None
    ) -> npt.NDArray[float] | tuple[float, float]:
        """
        From the probability amplitude, computes the probability that a measurement of a given qubit will give 0 or 1.

        Args:
            qubits_idx (int): Index identifying the qubit to be measured

        Returns:
            Probability of measuring qubit in position qubit_idx to be measured to be 0 or to be 1
        """

        if self._state is None:
            msg = "Error calculating measurement probabilities. State has not been initialized."
            raise ValueError(msg)

        if qubit_idx is None:
            outcome_probs = np.abs(self._state) ** 2
        else:
            if qubit_idx >= self.num_qubits or qubit_idx < 0:
                msg = (
                    "Invalid qubit index. Make sure it is between 1 and num_qubits - 1"
                )
                raise ValueError(msg)

            prob_0 = 0
            prob_1 = 0
            for idx, prob_amp in enumerate(self._state):
                if (idx >> qubit_idx) & 1 == 0:
                    prob_0 += np.abs(prob_amp) ** 2
                else:
                    prob_1 += np.abs(prob_amp) ** 2
            outcome_probs = (prob_0, prob_1)

        return outcome_probs

    def _update_state_post_measurement(self, qubit_idx: int, outcome: int) -> None:
        """
        Updates the quantum state post-measurement, effectively collapsing the wave function. based on the result obtained.

        Args:
            qubit_idx (int): Index identifying qubit which was measured
            outcome (int): Result of the measurement. Either 0 or 1

        Returns:
            None
        """

        if self._state is None:
            msg = "Error updating state. State has not been initialized."
            raise ValueError(msg)

        new_state = []
        for idx, amplitude in enumerate(self._state):
            if ((idx >> qubit_idx) & 1) == outcome:
                new_state.append(amplitude)
            else:
                new_state.append(0)

        self._state = np.array(new_state)
        self._normalize_state()

    def _normalize_state(self) -> None:
        """
        Updates the state of the system by normalizing its states.

        Args:
            n/a

        Returns:
            n/a
        """
        self._state /= np.linalg.norm(self._state)

    def apply_gate(self, gate: npt.NDArray[np.complex128], _: Any = None) -> None:
        """
        Applies quantum gate to the system of qubits.

        Args:
            gate (np.NDarray): Gate to be applied to the to the system

        Returns:
            n/a

        """

        if self._state is None:
            msg = "Error applying gate. State has not been initialized."
            raise ValueError(msg)

        N = int(np.log2(len(self._state)))
        gate = tensor_power(gate, N)
        self._state = np.dot(gate, self._state)
        self._normalize_state()

    def __str__(self) -> str:
        return str(self._state)

    def __repr__(self) -> str:
        return str(self._state)


@dataclasses.dataclass
class Agent:
    num_priv_qubits: Any = None
    qstates: dict[str, QState | Any] = dataclasses.field(
        default_factory=lambda: {"private": None, "public": None}
    )
    keys: dict[str, npt.NDArray[np.int_] | None] = dataclasses.field(
        default_factory=lambda: {"private": None, "public": None}
    )
    priv_qstates: QstateEnt | QstateUnEnt | None = None
    init_method: str = "random"
    priv_qbittype: str | None = None

    def __post_init__(self) -> None:
        """
        Initializes the priavate `qstate`.

        Args:
            n/a

        Returns:
            None
        """

        if self.priv_qstates is None:
            if self.priv_qbittype == "entangled":
                self.set_qstate(
                    QstateEnt(
                        num_qubits=self.num_priv_qubits, init_method=self.init_method
                    ),
                    "private",
                )
            elif self.priv_qbittype == "unentangled":
                self.set_qstate(
                    QstateUnEnt(
                        num_qubits=self.num_priv_qubits, init_method=self.init_method
                    ),
                    "private",
                )
        elif isinstance(self.priv_qstates, (QState)):
            self.set_qstate(qstate=self.priv_qstates, qstate_type="private")

    def set_qstate(self, qstate: QstateEnt | QstateUnEnt, qstate_type: str) -> None:
        """
        Sets a given qstate as either a private or public qubit of the Agent

        Args:
            qstate (QstateEnt or QstateUnEnt): Quantum state of a private or public system of qubits
            qstate_type (str): Whether the given qstate is to be private or public

        Returns:
            None
        """

        # if not isinstance(qstate, QstateUnEnt) and not isinstance(qstate, QstateEnt):
        #     msg = f"Got {type(qstate) = }. Expected either 'QstateUnEnt' | 'QstateEnt'.  Wrong type given for system state."
        #     raise ValueError(msg)

        self.qstates[qstate_type] = qstate

    def measure(self, qstate_type: str, qubit_idx: int) -> int:
        """
        Measure the qubit_idx'th qubit, calculating the probability of each outcome and returning said outcome.

        Args:
            qstate_type (str): Whether the given qstate is to be private or public
            qubit_idx (int): Index identifying the qubit to be measured

        Returns:
            Outcome of the measurement, either 0 or 1

        """
        if qstate_type not in self.qstates:
            msg = "Not valid qstate type."
            raise ValueError(msg)

        if self.qstates.get(qstate_type) is None:
            msg = f"Got {self.qstates[qstate_type] = }. qstate needs to be specified"
            raise ValueError(msg)

        return self.qstates[qstate_type].measure(qubit_idx=qubit_idx)

    def measure_all(self, qstate_type: str, order: Any = None) -> npt.NDArray[np.int_]:

        if self.qstates[qstate_type] is None:
            msg = f"Error measuring {qstate_type} qstate. It has not been initialized."
            raise ValueError(msg)

        if qstate_type not in self.qstates:
            msg = "Invalid qstate type"
            raise ValueError(msg)

        return self.qstates[qstate_type].measure_all(order)

    def apply_gate(self, gate: npt.NDArray[np.complex128], qstate_type: str, qubit_idx: int | None = None) -> None:
        if qstate_type not in self.qstates:
            msg = "Invalid qstate type"
            raise ValueError(msg)

        self.qstates[qstate_type].apply_gate(gate, qubit_idx=qubit_idx)

    def get_key(self, qstate_type: str, order: Any = None) -> npt.NDArray[np.int_]:
        outcome = self.measure_all(qstate_type=qstate_type, order=order)
        self.keys[qstate_type] = outcome
        return outcome
