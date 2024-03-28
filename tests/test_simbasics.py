import unittest
from qcrypto.simbasics import Qubit, Agent
import numpy as np


class TestQubit(unittest.TestCase):
    def test_initialization(self):
        """
        Test case to verify the correct initialization of a Qubit object with specified theta and phi parameters.
        """
        qubit = Qubit(np.pi / 2, 0)
        expected_state = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)])
        np.testing.assert_array_almost_equal(qubit.state, expected_state)

    def test_normalization(self):
        """
        Test case to verify the normalization of a Qubit object.
        """
        qubit = Qubit(0, 0)
        qubit.state = np.array([1, 2])
        qubit.normalize()
        expected_state = qubit.state / np.linalg.norm(qubit.state)
        np.testing.assert_array_almost_equal(qubit.state, expected_state)

    def test_probability_calculation(self):
        """
        Test case to verify the probability calculation of a Qubit object for |0> and |1> states.
        """
        qubit = Qubit(0, 0)  # Should be |0> state
        probs = qubit.get_probs()
        self.assertAlmostEqual(probs[0], 1, places=7)
        self.assertAlmostEqual(probs[1], 0, places=7)

        qubit = Qubit(np.pi, 0)  # Should be |1> state
        probs = qubit.get_probs()
        self.assertAlmostEqual(probs[0], 0, places=7)
        self.assertAlmostEqual(probs[1], 1, places=7)


class TestAgent(unittest.TestCase):
    def test_measurement(self):
        """
        Test case to verify the measurement process of an Agent object with specified bases.
        """
        agent = Agent(numqubits=1, basis_selection=[0])
        qubit = agent.qubits[0]
        result_zero_basis = agent.measure(qubit, 0)
        self.assertIn(result_zero_basis, [0, 1])

        agent_pi_4 = Agent(numqubits=1, basis_selection=[np.pi / 4])
        qubit_pi_4 = agent_pi_4.qubits[0]
        result_pi_4_basis = agent_pi_4.measure(qubit_pi_4, np.pi / 4)
        self.assertIn(result_pi_4_basis, [0, 1])

    def test_send_quantum_same_basis(self):
        """
        Test case to verify the sending and measurement of qubits between Alice and Bob
        when they use the same basis for sending and measuring.
        """
        basis_options = [0, np.pi / 4]
        chosen_basis = np.random.choice(basis_options)
        alice_state = np.random.choice([0, np.pi])
        alice_qubit_state = [alice_state]
        alice = Agent(numqubits=1, basis_selection=[chosen_basis])
        alice.qubits = [
            Qubit(theta=state, base=chosen_basis) for state in alice_qubit_state
        ]
        bob = Agent(numqubits=1, basis_selection=[chosen_basis])
        alice.send_quantum(bob, [chosen_basis])
        measured_value = bob.measurements[0]
        # This should always be True since they are using the same basis
        expected_value = 0 if alice_state == 0 else 1
        self.assertEqual(measured_value, expected_value)

    def test_send_quantum_cross_basis(self):
        """
        Test case to verify the sending and measurement of qubits between Alice and Bob
        when they use different bases for sending and measuring.
        """
        # Alice sends in 0 basis, Bob measures in π/4 basis.
        alice = Agent(numqubits=1, basis_selection=[0])
        bob_pi_4 = Agent(numqubits=1, basis_selection=[np.pi / 4])
        alice.send_quantum(bob_pi_4, [np.pi / 4])
        self.assertEqual(len(bob_pi_4.measurements), 1)
        # Assert that Bob's measurement is either 0 or 1.
        self.assertIn(bob_pi_4.measurements[0], [0, 1])

        # Now, we test the opposite: Alice sends in π/4 basis, Bob measures in 0 basis.
        alice_pi_4 = Agent(numqubits=1, basis_selection=[np.pi / 4])
        bob = Agent(numqubits=1, basis_selection=[0])
        alice_pi_4.send_quantum(bob, [0])
        self.assertEqual(len(bob.measurements), 1)
        # Assert that Bob's measurement is either 0 or 1.
        self.assertIn(bob.measurements[0], [0, 1])

    def test_get_keys(self):
        """
        Test case to verify the key generation process between Alice and Bob.
        """
        numqubits = 10
        alice_bases = np.random.choice([0, np.pi / 4], size=numqubits)
        alice_states = np.random.choice([0, np.pi], size=numqubits)
        alice = Agent(numqubits=numqubits)
        alice.qubits = np.array(
            [
                Qubit(theta=state, base=base)
                for state, base in zip(alice_states, alice_bases)
            ]
        )
        bob = Agent(numqubits=numqubits)
        alice.send_quantum(bob, alice_bases)
        alice_key = alice.get_key(alice_bases)
        # Bob generates his key after receiving and measuring qubits from Alice
        bob_key = bob.get_key(alice_bases)
        # Check if Alice's and Bob's keys match
        self.assertTrue(np.array_equal(alice_key, bob_key))


if __name__ == "__main__":
    unittest.main()
