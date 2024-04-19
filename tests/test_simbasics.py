import unittest
from qcrypto.simbasics import QstateUnEnt, QstateEnt, Agent
from qcrypto.gates import H_gate
import numpy as np


class TestQubit(unittest.TestCase):
    def test_initialization_unent_zeros(self):
        """
        Test case to verify the correct initialization of a QstateUnEnt object with specified state.
        """
        qubit = QstateUnEnt(num_qubits=2, init_method="zeros")
        expected_state = np.complex_(np.array([[1, 0], [1, 0]]))
        np.testing.assert_array_almost_equal(qubit._state, expected_state)

    def test_initialization_ent_zeros(self):
        """
        Test case to verify the correct initialization of a QstateEnt object with specified state.
        """
        qubit = QstateEnt(num_qubits=2, init_method="zeros")
        expected_state = np.complex_(np.array([1, 0, 0, 0]))
        np.testing.assert_array_almost_equal(qubit._state, expected_state)

    def test_normalization_unent(self):
        """
        Test case to verify the normalization of a QstateUnEnt object.
        """
        state = np.array([[1, 2], [3, 4]])
        qubit = QstateUnEnt(num_qubits=2, _state=state.copy())
        norms = np.linalg.norm(state, axis=1)
        norms = norms.reshape(-1, 1)
        expected_states = state / norms
        np.testing.assert_array_almost_equal(qubit._state, expected_states)

    def test_normalization_ent(self):
        """
        Test case to verify the normalization of a QstateEnt object.
        """
        state = np.complex_(np.array([1, 2, 3, 4]))
        qubit = QstateEnt(num_qubits=2, _state=state.copy())
        expected_state = state / np.linalg.norm(state)
        np.testing.assert_array_almost_equal(qubit._state, expected_state)

    def test_probability_calculation_unent(self):
        """
        Test case to verify the probability calculation of a QstateEnt object.
        """
        state0 = np.complex_(np.array([[1, 0]]))
        qubit = QstateUnEnt(num_qubits=1, _state=state0)
        probs = qubit._calculate_measurement_probs(qubit_idx=0)
        self.assertAlmostEqual(probs[0], 1, places=7)
        self.assertAlmostEqual(probs[1], 0, places=7)

        state1 = np.complex_(np.array([[0, 1]]))
        qubit = QstateUnEnt(num_qubits=1, _state=state1)  # Should be |1> state
        probs = qubit._calculate_measurement_probs(qubit_idx=0)
        self.assertAlmostEqual(probs[0], 0, places=7)
        self.assertAlmostEqual(probs[1], 1, places=7)

    def test_probability_calculation_ent(self):
        """
        Test case to verify the probability calculation of a QstateUnEnt object.
        """
        bin_outcome = "1010"
        outcome = int(bin_outcome, 2)  # 10
        num_qubits = 4
        state = np.complex_(np.zeros(2**num_qubits))
        state[outcome] = 1 + 0j
        qubit = QstateEnt(num_qubits=num_qubits, _state=state)
        probs = qubit._calculate_measurement_probs()

        np.testing.assert_equal(np.where(np.isclose(probs, 1))[0][0], outcome)


class TestAgent(unittest.TestCase):
    def test_measurement_unent(self):
        """
        Test case to verify the measurement process of an Agent object with QstateUnEnt.
        """
        num_qubits = 2
        qstate = QstateUnEnt(num_qubits=num_qubits, init_method="zeros")
        agent = Agent(
            priv_qbittype="unentangled", num_priv_qubits=2, init_method="zeros"
        )
        agent_outcome = agent.measure_all(qstate_type="private")
        qstate_outcome = qstate.measure_all()
        np.testing.assert_array_almost_equal(agent_outcome, qstate_outcome)

        agent_outcome0 = agent.measure(qubit_idx=0, qstate_type="private")
        qstate_outcome0 = qstate.measure(qubit_idx=0)
        self.assertAlmostEqual(agent_outcome0, qstate_outcome0)

    def test_measurement_ent(self):
        """
        Test case to verify the measurement process of an Agent object with QstateEnt.
        """
        num_qubits = 2
        qstate = QstateEnt(num_qubits=num_qubits, init_method="zeros")
        agent = Agent(priv_qbittype="entangled", num_priv_qubits=2, init_method="zeros")
        agent_outcome = agent.measure_all(qstate_type="private", order="simult")
        qstate_outcome = qstate.measure_all(order="simult")
        np.testing.assert_array_almost_equal(agent_outcome, qstate_outcome)

        agent_outcome0 = agent.measure(qubit_idx=0, qstate_type="private")
        qstate_outcome0 = qstate.measure(qubit_idx=0)
        self.assertAlmostEqual(agent_outcome0, qstate_outcome0)

    def test_public_qstates_unent(self):
        """
        Test case to verify the sharing of QstateUnEnt object works for Agents.
        """
        Alice = Agent()
        Bob = Agent()

        public_qstate = QstateUnEnt(num_qubits=2, init_method="zeros")

        Alice.set_qstate(qstate=public_qstate, qstate_type="public")
        Bob.set_qstate(qstate=public_qstate, qstate_type="public")

        self.assertIs(Alice.qstates["public"], Bob.qstates["public"])

        Alice.get_key(qstate_type="public")
        Bob.get_key(qstate_type="public")

        np.testing.assert_array_almost_equal(Alice.keys["public"], Bob.keys["public"])

    def test_public_qstate_ent(self):
        """
        Test case to verify the sharing of QstateEnt object works for Agents.
        """
        Alice = Agent()
        Bob = Agent()

        public_qstate = QstateEnt(num_qubits=2, init_method="zeros")

        Alice.set_qstate(qstate=public_qstate, qstate_type="public")
        Bob.set_qstate(qstate=public_qstate, qstate_type="public")

        self.assertIs(Alice.qstates["public"], Bob.qstates["public"])

        Alice.apply_gate(gate=H_gate, qstate_type="public")

        Alice.get_key(qstate_type="public", order="sequential")
        Bob.get_key(qstate_type="public", order="sequential")

        np.testing.assert_array_almost_equal(Alice.keys["public"], Bob.keys["public"])


if __name__ == "__main__":
    unittest.main()
