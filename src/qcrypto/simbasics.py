import numpy as np
import dataclasses



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
        self.state = self.state / np.sqrt(np.dot(np.conjugate(self.state), self.state))
        return self.state

    def get_probs(self):
        """
        Gives the probabilities of getting 0 or 1 when measuring the qubit
        """
        prob0 = (np.conjugate(self.state[0]) * self.state[0]).real
        prob1 = (np.conjugate(self.state[1]) * self.state[1]).real
        return prob0, prob1

    def copy(self):
        return Qubit(theta=self.theta, base=self.base, phi=self.phi)


class Agent:
    """
    Class representing the players in a quantum cryptography simulation (e.g. Alice, Eve or Bob)
    """

    def __init__(self, numqubits=None, basis_selection=None, key=None, message=None):
        """
        Args:
            numqubits: Number of qubits to be used
            basis_selection: Set of bases to choose from when generating the set of numqubits qubits.
                If not given, will default to 0 and pi/2.
        """
        self.qubits = np.array([], dtype=object)
        self.measurements = np.array([], dtype=int)
        self.key = np.array([], dtype=int)

        if numqubits is not None or numqubits != 0:
            if basis_selection is None:
                self.basis_selection = [0, np.pi / 4]
            else:
                self.basis_selection = basis_selection

            qubitbases = np.random.choice(self.basis_selection, numqubits)

            for basis in qubitbases:
                # theta = 0 -> qubit = 0 state
                # theta = pi -> qubit = 1 state
                qubit = Qubit(np.random.choice([0, np.pi]), basis, phi=0)
                self.qubits = np.append(self.qubits, qubit)

    def project(self, qubit, base):
        """
        Projects the given qubit unto the agent's measuring basis
        """
        newqubit = qubit
        base_delta = base - newqubit.base

        # Projection oprator
        proj_mtrx = np.array(
            [
                [np.cos(base_delta), -1 * np.sin(base_delta)],
                [np.sin(base_delta), np.cos(base_delta)],
            ]
        )

        newqubit.state = np.dot(proj_mtrx, newqubit.state)
        newqubit.normalize()

        newqubit.base = base

        return newqubit

    def measure(self, qubit, base, rtrn_result=True):
        """
        Gives the result of a sample measurement of the qubit.
        """
        self.project(qubit, base)
        measurement_result = np.random.choice((0, 1), p=qubit.get_probs())
        self.measurements = np.append(self.measurements, measurement_result)
        if rtrn_result:
            return measurement_result

    # def reset_results(self):
    # self.measurements = []

    # def genbasis(self, numqubits):
    # return np.random.choice([0, 1], numqubits)

    def send_quantum(self, recipient, recipient_bases):
        for qubit, base in zip(self.qubits, recipient_bases):
            recipient.measure(qubit.copy(), base)

        recipient.qubits = np.array([], dtype=object)
        for base, measurement in zip(recipient_bases, recipient.measurements):
            received_qubit = Qubit(measurement * np.pi, base)  # measurement = 0 or 1
            recipient.qubits = np.append(recipient.qubits, received_qubit)

    def get_key(self, bases):
        for qubit, base in zip(self.qubits, bases):
            measurement = self.measure(qubit, base)
            self.key = np.append(self.key, measurement)
        return self.key

    def send_classic(self, receiverAgent, bits):
        pass

    def genkey(self, numqubits, numcheckbits):
        """
        Generates an array of qubits
        """
        pass
