import numpy as np

class SarsaLambdaAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1, lambd=0.9):
        """
        Args:
            n_states: Number of discrete states (400)
            n_actions: Number of available actions (5)
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
            lambd: Trace decay factor (0 = SARSA, 1 = Monte Carlo)
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.lambd = lambd  # The 'lambda' in Sarsa(lambda)

        # Initialize Q-table with zeros
        self.q_table = np.zeros((n_states, n_actions))

        # Initialize Eligibility Trace table (same shape as Q)
        self.e_table = np.zeros((n_states, n_actions))

    def choose_action(self, state_idx):
        """
        Epsilon-greedy policy:
        With probability epsilon, choose random action.
        Otherwise, choose the best action from Q-table.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions) # Explore
        else:
            # Exploit: choose action with highest Q value
            # If multiple max values, choose randomly among them
            max_q = np.max(self.q_table[state_idx])
            actions_with_max_q = np.where(self.q_table[state_idx] == max_q)[0]
            return np.random.choice(actions_with_max_q)

    def update(self, state, action, reward, next_state, next_action):
        """
        Sarsa(lambda) update rule with Replacing Traces.
        """
        # 1. Calculate TD Error (delta)
        current_q = self.q_table[state, action]
        next_q = self.q_table[next_state, next_action]
        delta = reward + self.gamma * next_q - current_q

        # 2. Update Eligibility Traces (Replacing Traces)
        # First, decay ALL traces
        self.e_table *= (self.gamma * self.lambd)
        
        # Then, set the current state's trace to 1 (Replace, don't accumulate)
        # This is more stable for control problems where states are revisited often
        self.e_table[state, action] = 1.0

        # 3. Update Q-values
        # Q(s,a) <- Q(s,a) + alpha * delta * E(s,a)
        self.q_table += self.alpha * delta * self.e_table

    def reset_traces(self):
        """
        Should be called at the beginning of each new episode.
        Traces from the previous drive shouldn't affect the new one.
        """
        self.e_table.fill(0)

    def get_q_map(self):
        """
        Returns the Q-table in a shape suitable for plotting (States x Actions).
        We might need to reshape this if we want a 2D grid of the environment.
        """
        return self.q_table

    def get_trace_map(self):
        """
        Returns the current Eligibility Trace table.
        Used for the 'Memory Fade' visualization.
        """
        return self.e_table

    def save_model(self, filename="sarsa_lambda_model.npy"):
        """Save the Q-table to a file."""
        np.save(filename, self.q_table)

    def load_model(self, filename="sarsa_lambda_model.npy"):
        """Load a Q-table from a file."""
        self.q_table = np.load(filename)
