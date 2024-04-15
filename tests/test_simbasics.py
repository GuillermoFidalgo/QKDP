import unittest
from qcrypto.simbasics import QstateUnEnt, QstateEnt
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

        np.testing.assert_equal(
            np.where(np.isclose(qubit._calculate_measurement_probs(), 1))[0][0], outcome
        )

<<<<<<< HEAD

# class TestAgent(unittest.TestCase):
#     def test_measurement(self):
#         """
#         Test case to verify the measurement process of an Agent object with specified bases.
#         """
#         agent = Agent(numqubits=1, basis_selection=[0])
#         qubit = agent.qubits[0]
#         result_zero_basis = agent.measure(qubit, 0)
#         self.assertIn(result_zero_basis, [0, 1])
=======
>>>>>>> a5eef86 (Began making Agent tets compatible with new implem)

class TestAgent(unittest.TestCase):
    def test_measurement_unent(self):
        """
        Test case to verify the measurement process of an Agent object with specified bases.
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
        Test case to verify the measurement process of an Agent object with specified bases.
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

    # def test_send_quantum_same_basis(self):
    #     """
    #     Test case to verify the sending and measurement of qubits between Alice and Bob
    #     when they use the same basis for sending and measuring.
    #     """
    #     basis_options = [0, np.pi / 4]
    #     chosen_basis = np.random.choice(basis_options)
    #     alice_state = np.random.choice([0, np.pi])
    #     alice_qubit_state = [alice_state]
    #     alice = Agent(numqubits=1, basis_selection=[chosen_basis])
    #     alice.qubits = [
    #         Qubit(theta=state, base=chosen_basis) for state in alice_qubit_state
    #     ]
    #     bob = Agent(numqubits=1, basis_selection=[chosen_basis])
    #     alice.send_quantum(bob, [chosen_basis])
    #     measured_value = bob.measurements[0]
    #     # This should always be True since they are using the same basis
    #     expected_value = 0 if alice_state == 0 else 1
    #     self.assertEqual(measured_value, expected_value)

    # def test_send_quantum_cross_basis(self):
    #     """
    #     Test case to verify the sending and measurement of qubits between Alice and Bob
    #     when they use different bases for sending and measuring.
    #     """
    #     # Alice sends in 0 basis, Bob measures in π/4 basis.
    #     alice = Agent(numqubits=1, basis_selection=[0])
    #     bob_pi_4 = Agent(numqubits=1, basis_selection=[np.pi / 4])
    #     alice.send_quantum(bob_pi_4, [np.pi / 4])
    #     self.assertEqual(len(bob_pi_4.measurements), 1)
    #     # Assert that Bob's measurement is either 0 or 1.
    #     self.assertIn(bob_pi_4.measurements[0], [0, 1])

    #     # Now, we test the opposite: Alice sends in π/4 basis, Bob measures in 0 basis.
    #     alice_pi_4 = Agent(numqubits=1, basis_selection=[np.pi / 4])
    #     bob = Agent(numqubits=1, basis_selection=[0])
    #     alice_pi_4.send_quantum(bob, [0])
    #     self.assertEqual(len(bob.measurements), 1)
    #     # Assert that Bob's measurement is either 0 or 1.
    #     self.assertIn(bob.measurements[0], [0, 1])

    # def test_get_keys(self):
    #     """
    #     Test case to verify the key generation process between Alice and Bob.
    #     """
    #     numqubits = 10
    #     alice_bases = np.random.choice([0, np.pi / 4], size=numqubits)
    #     alice_states = np.random.choice([0, np.pi], size=numqubits)
    #     alice = Agent(numqubits=numqubits)
    #     alice.qubits = np.array(
    #         [
    #             Qubit(theta=state, base=base)
    #             for state, base in zip(alice_states, alice_bases)
    #         ]
    #     )
    #     bob = Agent(numqubits=numqubits)
    #     alice.send_quantum(bob, alice_bases)
    #     alice_key = alice.get_key(alice_bases)
    #     # Bob generates his key after receiving and measuring qubits from Alice
    #     bob_key = bob.get_key(alice_bases)
    #     # Check if Alice's and Bob's keys match
    #     self.assertTrue(np.array_equal(alice_key, bob_key))


if __name__ == "__main__":
    unittest.main()
