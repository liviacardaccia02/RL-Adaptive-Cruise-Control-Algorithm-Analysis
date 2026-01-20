import numpy as np
import matplotlib.pyplot as plt

# Import the modules
from environment.acc_env import ACCEnvironment
from utils.discretizer import Discretizer
from utils.plotting import plot_q_heatmap, plot_memory_fade, plot_learning_curve
from agents.sarsa_lambda import SarsaLambdaAgent

def run_integration_test():
    # --- 1. CONFIGURATION ---
    env_config = {
        'time_step': 0.1,
        'target_distance': 20.0,
        'max_distance': 100.0,
        'min_distance': 5.0,
        'actions': [-5.0, -2.0, 0.0, 2.0, 5.0] # Hard Brake -> Hard Accel
    }

    env = ACCEnvironment(env_config)
    discretizer = Discretizer(min_dist=0, max_dist=100,
                              min_vel=-10, max_vel=10)

    agent = SarsaLambdaAgent(
        n_states=discretizer.get_num_states(),
        n_actions=len(env_config['actions']),
        alpha=0.1,      # Learning Rate
        gamma=0.95,     # Discount Factor
        epsilon=1.0,    # Initial Exploration (will be decayed)
        lambd=0.9       # Eligibility Trace Decay
    )

    # --- 2. TRAINING LOOP ---
    n_episodes = 5000   # Increased for better learning
    max_steps = 200     # Limit episode length (20 seconds) to prevent infinite loops

    # Epsilon Decay Parameters
    epsilon_start = 1.0
    epsilon_end = 0.01
    decay_rate = 0.001

    total_rewards = []

    print(f"--- Starting Training for {n_episodes} Episodes ---")

    for episode in range(n_episodes):
        # A. Epsilon Decay Logic
        # Decay the randomness as the agent gets smarter
        agent.epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                        np.exp(-decay_rate * episode)

        # B. Reset for new episode
        state_raw = env.reset() # Returns [dist, vrel]
        state_idx = discretizer.get_state_index(state_raw[0], state_raw[1])

        agent.reset_traces() # Crucial for Sarsa(lambda)

        action = agent.get_action(state_idx)

        episode_reward = 0
        done = False
        steps = 0

        # C. Step Loop (with Max Steps safety check)
        while not done and steps < max_steps:
            # Take Step
            next_state_raw, reward, done, info = env.step(action)

            # Discretize Next State
            next_state_idx = discretizer.get_state_index(next_state_raw[0], next_state_raw[1])

            # Choose Next Action
            next_action = agent.get_action(next_state_idx)

            # Sarsa(lambda) Update
            agent.update(state_idx, action, reward, next_state_idx, next_action)

            # Move to next step
            state_idx = next_state_idx
            action = next_action
            episode_reward += reward
            steps += 1

        total_rewards.append(episode_reward)

        if (episode + 1) % 500 == 0:
            print(f"Episode {episode+1}/{n_episodes} - Reward: {episode_reward:.2f} - Epsilon: {agent.epsilon:.3f}")

    print("--- Training Complete ---")

    # --- 3. VISUALIZATION 1: LEARNING CURVE ---
    print("Generating Learning Curve...")
    # NOTE: We use the specific argument name we defined in plotting.py
    plot_learning_curve(rewards_sarsa_lambda=total_rewards)

    # --- 4. VISUALIZATION 2: THE BRAIN (HEATMAP) ---
    print("Generating Policy Heatmap...")
    plot_q_heatmap(agent, discretizer)

    # --- 5. VISUALIZATION 3: MEMORY FADE (TRACES) ---
    print("Running Demo Episode for Trace Visualization...")

    state_raw = env.reset()
    state_idx = discretizer.get_state_index(state_raw[0], state_raw[1])
    agent.reset_traces()
    agent.epsilon = 0.0 # Pure exploitation to show learned behavior
    action = agent.get_action(state_idx)

    # Run for up to 50 steps to build up traces
    for _ in range(50):
        next_state_raw, reward, done, info = env.step(action)
        next_state_idx = discretizer.get_state_index(next_state_raw[0], next_state_raw[1])
        next_action = agent.get_action(next_state_idx)

        agent.update(state_idx, action, reward, next_state_idx, next_action)

        if done: break
        state_idx = next_state_idx
        action = next_action

    plot_memory_fade(agent, discretizer)

if __name__ == "__main__":
    run_integration_test()
