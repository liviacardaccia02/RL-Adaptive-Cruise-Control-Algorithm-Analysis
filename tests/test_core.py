import unittest

import numpy as np

from agents.sarsa import SarsaAgent
from agents.sarsa_lambda import SarsaLambdaAgent
from utils.discretizer import Discretizer


class DiscretizerTests(unittest.TestCase):
    def test_all_bucket_indices_are_reachable_and_in_range(self):
        d = Discretizer(min_dist=0, max_dist=10, dist_buckets=4,
                        min_vel=-2, max_vel=2, vel_buckets=3)
        indices = {
            d.get_state_index(distance, velocity)
            for distance in [0, 2.5, 5, 7.5, 10]
            for velocity in [-2, -2/3, 0, 2/3, 2]
        }
        self.assertEqual(d.n_states, 12)
        self.assertTrue(all(0 <= idx < d.n_states for idx in indices))
        self.assertEqual(len(indices), d.n_states)

    def test_out_of_range_values_are_clamped(self):
        d = Discretizer(dist_buckets=4, vel_buckets=3)
        self.assertEqual(d.get_state_index(-100, -100), 0)
        self.assertEqual(d.get_state_index(1000, 1000), d.n_states - 1)


class TerminalUpdateTests(unittest.TestCase):
    def test_sarsa_terminal_target_does_not_bootstrap(self):
        agent = SarsaAgent(n_states=2, n_actions=2, alpha=1.0, gamma=0.9, epsilon=0)
        agent.q_table[1, 1] = 100
        agent.learn(0, 0, reward=2, next_state_idx=1, next_action_idx=1, done=True)
        self.assertAlmostEqual(agent.q_table[0, 0], 2.0)

    def test_sarsa_lambda_terminal_target_does_not_bootstrap(self):
        agent = SarsaLambdaAgent(n_states=2, n_actions=2, alpha=1.0, gamma=0.9, epsilon=0)
        agent.q_table[1, 1] = 100
        agent.update(0, 0, reward=2, next_state=1, next_action=1, done=True)
        self.assertAlmostEqual(agent.q_table[0, 0], 2.0)


if __name__ == "__main__":
    unittest.main()
