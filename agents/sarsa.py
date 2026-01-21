import numpy as np
import random
import pickle
import os
import sys

# Robust import handling
try:
    from environment.acc_env import ACCEnvironment
    from utils.discretizer import Discretizer
    from utils.plotting import plot_training_results_SARSA
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from environment.acc_env import ACCEnvironment
    from utils.discretizer import Discretizer
    from utils.plotting import plot_training_results_SARSA

class SARSAAgent:
    """
    Implementation of the SARSA (State-Action-Reward-State-Action) algorithm.
    Adapted for the Official Group Environment.
    """

    def __init__(self, num_states, num_actions, alpha=0.05, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.9995):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Initialize Q-table
        self.q_table = np.zeros((self.num_states, self.num_actions))

    def choose_action(self, state_idx, greedy=False):
        if not greedy and random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        
        q_values = self.q_table[state_idx]
        max_q = np.max(q_values)
        actions_with_max_q = np.where(q_values == max_q)[0]
        return np.random.choice(actions_with_max_q)

    def update(self, state_idx, action_idx, reward, next_state_idx, next_action_idx):
        current_q = self.q_table[state_idx, action_idx]
        next_q = self.q_table[next_state_idx, next_action_idx]
        target = reward + self.gamma * next_q
        self.q_table[state_idx, action_idx] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        with open(file_path, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"Model loaded from {file_path}")


# ==========================================
# TRAINING LOGIC
# ==========================================

def run_training():
    print("\n=== Starting SARSA Agent Training (Re-Balanced for New Env) ===")
    
    # Configuration
    num_episodes = 5000
    max_steps_per_episode = 500 
    
    # Env Config
    env_config = {
        'time_step': 0.1,
        'target_distance': 30.0, 
        'max_distance': 100.0,
        'min_distance': 5.0,
        'actions': [-3.0, -1.0, 0.0, 1.0, 3.0] 
    }

    # Hyperparameters
    alpha = 0.05
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.9995
    
    # Paths
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    model_path = os.path.join(results_dir, "sarsa_model.pkl")
    plot_path = os.path.join(results_dir, "training_rewards.png")

    # Environment Setup
    env = ACCEnvironment(env_config) 
    
    # Discretizer Setup (Fixed Names)
    discretizer = Discretizer(min_dist=0, max_dist=100, min_vel=-10, max_vel=10)
    
    # Agent Setup
    num_actions = len(env_config['actions'])
    
    agent = SARSAAgent(
        num_states=discretizer.get_num_states(),
        num_actions=num_actions,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay
    )
    
    episode_rewards = []
    avg_rewards = []

    # Training Loop
    for episode in range(num_episodes):
        state = env.reset()
        distance, rel_velocity = state[0], state[1] 
        
        state_idx = discretizer.get_state_index(distance, rel_velocity)
        action_idx = agent.choose_action(state_idx)
        
        total_reward = 0
        done = False
        
        for step in range(max_steps_per_episode):
            next_state, reward, done, info = env.step(action_idx)
            next_distance, next_rel_vel = next_state[0], next_state[1]
            
            # --- REWARD SHAPING (CRITICO: Ribilanciamento) ---
            training_reward = reward
            
            # L'env del capo dà già -0.5 per le azioni estreme.
            # Prima davamo -2.0, portando il totale a -2.5 (troppo alto!).
            # Ora diamo solo -0.5 extra.
            # Totale penalità: -1.0. 
            # Questo permette all'agente di usare l'accelerazione se serve davvero.
            if action_idx == (num_actions - 1): # Hard Accel
                training_reward -= 0.5 
            # -----------------------------------------------
            
            next_state_idx = discretizer.get_state_index(next_distance, next_rel_vel)
            next_action_idx = agent.choose_action(next_state_idx)
            
            agent.update(state_idx, action_idx, training_reward, next_state_idx, next_action_idx)
            
            state_idx = next_state_idx
            action_idx = next_action_idx
            total_reward += reward 
            
            if done:
                break
        
        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        avg_rewards.append(np.mean(episode_rewards[-100:]))
        
        if (episode + 1) % 100 == 0:
            print(f"SARSA Train | Episode {episode + 1}/{num_episodes} | Avg Reward: {avg_rewards[-1]:.2f} | Eps: {agent.epsilon:.4f}")

    agent.save_model(model_path)
    plot_training_results_SARSA(episode_rewards, avg_rewards, plot_path, algorithm_name="SARSA")
    print("SARSA Training finished.")


# ==========================================
# TESTING LOGIC
# ==========================================

def run_testing():
    print("\n=== Starting SARSA Agent Testing ===")
    
    model_path = os.path.join("results", "sarsa_model.pkl")
    if not os.path.exists(model_path):
        print("Model not found. Run training first.")
        return

    # Configuration identical to training
    env_config = {
        'time_step': 0.1,
        'target_distance': 30.0, 
        'max_distance': 100.0,
        'min_distance': 5.0,
        'actions': [-3.0, -1.0, 0.0, 1.0, 3.0]
    }

    env = ACCEnvironment(env_config)
    discretizer = Discretizer(min_dist=0, max_dist=100, min_vel=-10, max_vel=10)
    num_actions = len(env_config['actions'])

    agent = SARSAAgent(
        num_states=discretizer.get_num_states(),
        num_actions=num_actions
    )
    
    agent.load_model(model_path)
    
    n_test_episodes = 100
    crashes = 0
    total_rewards = []
    hard_maneuvers_count = 0
    
    max_steps_limit = 500 
    
    for episode in range(n_test_episodes):
        state = env.reset()
        distance, rel_velocity = state[0], state[1]
        
        state_idx = discretizer.get_state_index(distance, rel_velocity)
        done = False
        ep_reward = 0
        steps = 0
        
        while not done and steps < max_steps_limit:
            action_idx = agent.choose_action(state_idx, greedy=True)
            
            if action_idx == 0 or action_idx == (num_actions - 1):
                hard_maneuvers_count += 1
            
            next_state, reward, done, info = env.step(action_idx)
            next_dist, next_rel_v = next_state[0], next_state[1]
            
            state_idx = discretizer.get_state_index(next_dist, next_rel_v)
            ep_reward += reward
            steps += 1
            
            if done and info.get('msg') == 'CRASH': 
                crashes += 1
        
        total_rewards.append(ep_reward)

    avg_reward = np.mean(total_rewards)
    crash_rate = (crashes / n_test_episodes) * 100
    avg_hard_maneuvers = hard_maneuvers_count / n_test_episodes
    
    print("\n" + "="*40)
    print(f"  SARSA AGENT PERFORMANCE REPORT")
    print("="*40)
    print(f"  Episodes Tested : {n_test_episodes}")
    print(f"  Avg Reward      : {avg_reward:.2f}")
    print(f"  Crash Rate      : {crash_rate:.1f}%")
    print(f"  Hard Maneuvers  : {avg_hard_maneuvers:.1f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    # Forza il training per sovrascrivere il vecchio modello
    run_training()
    run_testing()