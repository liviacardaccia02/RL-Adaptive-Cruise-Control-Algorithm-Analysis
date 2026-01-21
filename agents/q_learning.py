import numpy as np
import os
import pickle

class QLearningAgent:
    """
    Q-Learning Agent for the Adaptive Cruise Control problem.
    Uses a tabular approach to learn the optimal policy.
    """

    def __init__(self, n_states: int, n_actions: int, alpha: float = 0.1, 
                 gamma: float = 0.99, epsilon: float = 1.0, 
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.01):
        """
        Initialize the Q-Learning Agent.

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

        Args:
            state_idx (int): The discretized index of the current state.

        Returns:
            int: The index of the selected action.
        """
        # Exploration: Random action
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.n_actions)
        
        # Exploitation: Best action based on Q-table
        # If multiple actions have the same max value, choose randomly among them
        max_value = np.max(self.q_table[state_idx])
        best_actions = np.where(self.q_table[state_idx] == max_value)[0]
        return np.random.choice(best_actions)

    def learn(self, state_idx: int, action_idx: int, reward: float, 
              next_state_idx: int, done: bool):
        """
        Update the Q-table using the Bellman Optimality Equation.
        
        Q(s,a) = Q(s,a) + alpha * [R + gamma * max(Q(s', a')) - Q(s,a)]

        Args:
            state_idx (int): Current state index.
            action_idx (int): Action taken.
            reward (float): Reward received.
            next_state_idx (int): Next state index.
            done (bool): Whether the episode has ended.
        """
        current_q = self.q_table[state_idx, action_idx]
        
        # If terminal state, there is no future Q value
        if done:
            max_future_q = 0.0
        else:
            max_future_q = np.max(self.q_table[next_state_idx])
        
        # Calculate Target and TD Error
        target = reward + (self.gamma * max_future_q)
        
        # Update Q-value
        self.q_table[state_idx, action_idx] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        """Reduces the exploration rate after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, filepath: str):
        """Saves the Q-table to a file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Loads a Q-table from a file."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Model loaded from {filepath}")
        else:
            print(f"Warning: Model file {filepath} not found. Starting from scratch.")