import numpy as np
import os
import matplotlib.pyplot as plt

from environment.acc_env import ACCEnvironment
from utils.discretizer import Discretizer
from agents.q_learning import QLearningAgent
from utils.plotting import plot_learning_curve, plot_policy_heatmap, plot_value_heatmap

def run_training():
    # --- 1. CONFIGURATION ---
    OUTPUT_DIR = "visualizations/q_learning"
    
    # Base environment configuration (rewards are already handled internally in acc_env.py)
    env_config = {
        'time_step': 0.1,
        'target_distance': 20.0,
        'max_distance': 100.0,
        'min_distance': 0.0,
        'actions': [-5.0, -2.0, 0.0, 2.0, 5.0] 
    }
    
    env = ACCEnvironment(env_config)
    
    # Discretizer setup
    discretizer = Discretizer(min_dist=0, max_dist=100, dist_buckets=20,
                              min_vel=-10, max_vel=10, vel_buckets=20)

    # Q-Learning Agent Initialization
    # Parameters optimized for stability with the new high penalties
    agent = QLearningAgent(
        n_states=discretizer.get_num_states(),
        n_actions=len(env_config['actions']),
        alpha=0.1,      # Standard learning rate
        gamma=0.99,     # High importance to future (crucial to avoid distant crashes)
        epsilon=1.0,    # Start with maximum exploration
        epsilon_decay=0.9995, # Very slow decay to thoroughly explore before converging
        epsilon_min=0.01
    )

    n_episodes = 10000 
    max_steps = 400     # Maximum episode duration
    total_rewards = []
    
    print(f"--- Starting Q-Learning Training: {n_episodes} Episodes ---")
    print(f"--- Output Directory: {OUTPUT_DIR} ---")

    # --- 2. TRAINING LOOP ---
    for episode in range(n_episodes):
        # Reset environment and initial state   
        state_raw = env.reset()
        state_idx = discretizer.get_state_index(state_raw[0], state_raw[1])
        
        episode_reward = 0
        done = False
        steps = 0

        while not done and steps < max_steps:
            # A. Action Choice (Epsilon-Greedy)
            action_idx = agent.choose_action(state_idx)
            
            # B. Step Execution
            next_state_raw, reward, done, info = env.step(action_idx)
            next_state_idx = discretizer.get_state_index(next_state_raw[0], next_state_raw[1])
            
            # C. Learning (Q-Learning Update)
            agent.learn(state_idx, action_idx, reward, next_state_idx, done)
            
            # D. Advancement
            state_idx = next_state_idx
            episode_reward += reward
            steps += 1
        
        # End of episode: Decay exploration
        agent.decay_epsilon()
        total_rewards.append(episode_reward)

        if (episode + 1) % 1000 == 0:
            avg_rew = np.mean(total_rewards[-100:])
            print(f"Episode {episode+1}/{n_episodes} | Avg Reward (last 100): {avg_rew:.2f} | Epsilon: {agent.epsilon:.3f}")

    print("--- Training Completed ---")

    # --- 3. SAVE RESULTS AND PLOTS ---
    
    # A. Save the trained model (the Q-table)
    model_path = os.path.join(OUTPUT_DIR, "q_table.pkl")
    agent.save_model(model_path)

    # B. Plot Learning Curve
    # Pass a dictionary to label the curve correctly
    plot_learning_curve(
        rewards_dict={'Q-Learning': total_rewards}, 
        save_path=os.path.join(OUTPUT_DIR, "learning_curve.png"),
        window=100,
        title="Q-Learning Training Performance"
    )

    # Set epsilon to 0 to visualize the learned "Optimal Policy" (without random noise)
    agent.epsilon = 0.0

    # C. Plot Policy Heatmap (What the agent does in each state according to learned Q-values)
    plot_policy_heatmap(
        agent, discretizer, 
        title="Q-Learning Optimal Policy",
        save_path=os.path.join(OUTPUT_DIR, "policy_heatmap.png")
    )
    
    # D. Plot Value Heatmap (How confident the agent feels about each state - max Q-value)
    plot_value_heatmap(
        agent, discretizer,
        title="Q-Learning Value Function (Max Q)",
        save_path=os.path.join(OUTPUT_DIR, "value_heatmap.png")
    )

if __name__ == "__main__":
    run_training()