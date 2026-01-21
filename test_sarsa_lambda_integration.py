import numpy as np
import os
import matplotlib.pyplot as plt

from environment.acc_env import ACCEnvironment
from utils.discretizer import Discretizer
from agents.sarsa_lambda import SarsaLambdaAgent
from utils.plotting import plot_learning_curve, plot_policy_heatmap, plot_memory_fade

def run_integration_test():
    # --- 1. Configuration ---
    OUTPUT_DIR = "visualizations/sarsa_lambda"
    
    # Environment Configuration (Physics and Rewards handled internally in acc_env.py)
    env_config = {
        'time_step': 0.1,
        'target_distance': 20.0,
        'max_distance': 100.0,
        'min_distance': 0.0, 
        'actions': [-5.0, -2.0, 0.0, 2.0, 5.0] 
    }
    
    # Training Configuration
    train_config = {
        'n_episodes': 10000,   
        'max_steps': 200,     
        'alpha': 0.1,
        'gamma': 0.95,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'decay_rate': 0.001,
        'lambda': 0.9        
    }

    env = ACCEnvironment(env_config)
    discretizer = Discretizer(min_dist=0, max_dist=100, dist_buckets=20,
                              min_vel=-10, max_vel=10, vel_buckets=20)

    # Initialization of Sarsa(λ) Agent
    agent = SarsaLambdaAgent(
        n_states=discretizer.get_num_states(),
        n_actions=len(env_config['actions']),
        alpha=train_config['alpha'],
        gamma=train_config['gamma'],
        epsilon=train_config['epsilon_start'],
        lambd=train_config['lambda']
    )

    total_rewards = []
    
    print(f"--- Starting Training SARSA(λ): {train_config['n_episodes']} Episodes ---")
    print(f"--- Output Directory: {OUTPUT_DIR} ---")

    # --- 2. TRAINING LOOP ---
    for episode in range(train_config['n_episodes']):
        # A. Decay Epsilon
        agent.epsilon = train_config['epsilon_end'] + \
                        (train_config['epsilon_start'] - train_config['epsilon_end']) * \
                        np.exp(-train_config['decay_rate'] * episode)

        # B. Reset Env and Traces
        state_raw = env.reset()
        state_idx = discretizer.get_state_index(state_raw[0], state_raw[1])
        agent.reset_traces() # Fundamental for Sarsa(λ)

        # SARSA is On-Policy: we must choose the action BEFORE the loop
        action = agent.get_action(state_idx)

        episode_reward = 0
        done = False
        steps = 0

        while not done and steps < train_config['max_steps']:
            # 1. Execute Step
            # Note: Does SarsaLambdaAgent usually work with indices or physical actions?
            # If agent.get_action returns an index, use that.
            # If env accepts float, convert. Assume env accepts indices (as modified for QL)
            # If your env only accepts float, use: env_config['actions'][action]
            next_state_raw, reward, done, info = env.step(action)

            # 2. Discretization
            next_state_idx = discretizer.get_state_index(next_state_raw[0], next_state_raw[1])

            # 3. Choose Next Action (On-Policy)
            next_action = agent.get_action(next_state_idx)

            # 4. Update Sarsa(λ)
            agent.update(state_idx, action, reward, next_state_idx, next_action)

            # 5. Advancement    
            state_idx = next_state_idx
            action = next_action
            episode_reward += reward
            steps += 1

        total_rewards.append(episode_reward)

        if (episode + 1) % 500 == 0:
            avg_rew = np.mean(total_rewards[-100:])
            print(f"Episode {episode+1}/{train_config['n_episodes']} | Avg Reward: {avg_rew:.2f} | Epsilon: {agent.epsilon:.3f}")

    print("--- Training Completed ---")

    # --- 3. SAVING PLOTS ---
    
    # A. Learning Curve
    # FIX: Use dictionary as required by the new plotting.py
    plot_learning_curve(
        rewards_dict={'Sarsa(λ)': total_rewards},
        save_path=os.path.join(OUTPUT_DIR, "learning_curve.png"),
        window=100,
        title="Sarsa(λ) Training Performance"
    )

    # B. Policy Heatmap
    plot_policy_heatmap(
        agent, discretizer,
        title="Sarsa(λ) Learned Policy",
        save_path=os.path.join(OUTPUT_DIR, "agent_policy_heatmap.png")
    )

    # C. Memory Fade (Eligibility Traces) Visualization
    state_raw = env.reset()
    state_idx = discretizer.get_state_index(state_raw[0], state_raw[1])
    agent.reset_traces()
    agent.epsilon = 0.0 # Greedy
    action = agent.get_action(state_idx)

    for _ in range(50):
        next_state_raw, reward, done, info = env.step(action)
        next_state_idx = discretizer.get_state_index(next_state_raw[0], next_state_raw[1])
        next_action = agent.get_action(next_state_idx)
        agent.update(state_idx, action, reward, next_state_idx, next_action)
        if done: break
        state_idx = next_state_idx
        action = next_action

    plot_memory_fade(
        agent, discretizer,
        save_path=os.path.join(OUTPUT_DIR, "memory_fade.png")
    )

if __name__ == "__main__":
    run_integration_test()