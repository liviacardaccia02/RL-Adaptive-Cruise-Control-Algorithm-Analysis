import numpy as np
import os
import pickle

class SarsaAgent:
    """
    SARSA Agent (State-Action-Reward-State-Action).
    On-Policy TD Control algorithm.
    """

    def __init__(self, n_states: int, n_actions: int, alpha: float = 0.1, 
                 gamma: float = 0.99, epsilon: float = 1.0, 
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.01):
        """
        Initialize the SARSA Agent.

        Args:
            n_states (int): Total number of discretized states.
            n_actions (int): Total number of available actions.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Initial exploration rate.
            epsilon_decay (float): Decay rate for epsilon per episode.
            epsilon_min (float): Minimum exploration rate.
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize Q-table with zeros
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state_idx: int) -> int:
        """
        Select an action using Epsilon-Greedy strategy.
        Standardized name matching QLearningAgent.
        """
        # Exploration
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.n_actions)
        
        # Exploitation
        max_value = np.max(self.q_table[state_idx])
        best_actions = np.where(self.q_table[state_idx] == max_value)[0]
        return np.random.choice(best_actions)

    def learn(self, state_idx: int, action_idx: int, reward: float, 
              next_state_idx: int, next_action_idx: int, done: bool = False):
        """
        Update the Q-table using the SARSA update rule.
        
        Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]
        
        Args:
            state_idx (int): Current state.
            action_idx (int): Action taken.
            reward (float): Reward received.
            next_state_idx (int): Next state.
            next_action_idx (int): The actual action taken in the next state (On-Policy).
            done (bool): Whether the episode ended.
        """
        current_q = self.q_table[state_idx, action_idx]
        
        # If it is a terminal state, the future reward is 0
        if done:
            target = reward
        else:
            next_q = self.q_table[next_state_idx, next_action_idx]
            target = reward + (self.gamma * next_q)
        
        # Update Rule
        self.q_table[state_idx, action_idx] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        """Reduces the exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, filepath: str):
        """Saves the Q-table to a file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"SARSA Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Loads a Q-table from a file."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"SARSA Model loaded from {filepath}")
        else:
            print(f"Warning: Model file {filepath} not found.")