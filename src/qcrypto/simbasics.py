import numpy as np
import dataclasses

@dataclasses.dataclass
class Qubit:
    """
    Data class which represents the qubit and its quantum state.
    Quantum state is given by:
    |\psi> = \cos(\frac12 \theta) |0> + e^{i\phi}  \sin(\frac12 \theta) |1>
    """

    def __init__(self, theta, phi, base):
        self.base = base
        self.state = np.array([
            np.cos(theta/2),
            np.exp(1j * phi) * np.sin(theta/2)
        ])

    def __str__(self):
        return "{} |0> + {} |1>".format(self.state[0], self.state[1])
    
    def __repr__(self):
        return "{} |0> + {} |1>".format(self.state[0], self.state[1])

    def normalize(self):
        """
        Nomalizes the qubit's quantum state
        """
        self.state = self.state / np.sqrt(np.dot(np.conjugate(self.state), self.state))

    def get_probs(self):
        """
        Gives the probabilities of getting 0 or 1 when measuring the qubit
        """
        prob0 = (np.conjugate(self.state[0]) * self.state[0]).real
        prob1 = (np.conjugate(self.state[1]) * self.state[1]).real
        return prob0, prob1
    

class Agent:
    """
    Class representing the players in a quantum cryptography simulation (e.g. Alice, Eve or Bob)
    """
    def __init__(self, base=0, phase=0, key=None, message=None):
        self.base = base
        self.phase = phase

    def project(self, qubit):
        """
        Projects the given qubit unto the agent's measuring basis
        """
        newqubit = qubit
        base_delta = self.base - newqubit.base

        # Projection oprator
        proj_mtrx = np.array([
            [np.cos(base_delta),                                -1 * np.exp(-1j * self.phase) * np.sin(base_delta)],
            [np.sin(base_delta) * np.exp(-1j * self.phase),     np.cos(base_delta)                                ]
        ])
        
        newqubit.state = np.dot(proj_mtrx, newqubit.state)
        newqubit.normalize()

        newqubit.base = self.base

        return newqubit

    def measure(self, qubit):
        """
        Gives the result of a sample measurement of the qubit.
        """
        self.project(qubit)
        measurement_result = np.random.choice((0, 1), p=qubit.get_probs())
        return(measurement_result)

    def send_quantum(self, receiverAgent, qubits):
        pass
    
    def send_classic(self, receiverAgent, bits):
        pass
