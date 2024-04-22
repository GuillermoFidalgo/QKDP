from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import numpy.typing as npt

from qcrypto.qstates import QState, QstateEnt, QstateUnEnt


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
        # msg = f"Got {type(qstate) = }. Expected either 'QstateUnEnt' | 'QstateEnt'.  Wrong type given for system state."
        # raise ValueError(msg)
        if (qstate_type == "public") or (qstate_type == "private"):
            self.qstates[qstate_type] = qstate
        else:
            msg = f"Got {qstate_type = }. Expected either 'public' | 'private'.  Wrong string given for qstate type."
            raise ValueError(msg)

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

    def apply_gate(
        self,
        gate: npt.NDArray[np.complex128],
        qstate_type: str,
        qubit_idx: int | None = None,
    ) -> None:
        if qstate_type not in self.qstates:
            msg = "Invalid qstate type"
            raise ValueError(msg)

        if isinstance(self.qstates[qstate_type], (QstateUnEnt, QstateUnEnt)):
            self.qstates[qstate_type].apply_gate(gate, qubit_idx=qubit_idx)

    def get_key(self, qstate_type: str, order: Any = None) -> npt.NDArray[np.int_]:
        outcome = self.measure_all(qstate_type=qstate_type, order=order)
        self.keys[qstate_type] = outcome
        return outcome
