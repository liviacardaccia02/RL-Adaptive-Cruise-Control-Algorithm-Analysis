import numpy as np
import os
import matplotlib.pyplot as plt

from environment.acc_env import ACCEnvironment
from utils.discretizer import Discretizer
from utils.plotting import plot_learning_curve, plot_policy_heatmap, plot_value_heatmap, ensure_dir

# Import Agents
from agents.q_learning import QLearningAgent
from agents.sarsa import SarsaAgent
from agents.sarsa_lambda import SarsaLambdaAgent

# --- 1. CENTRALIZED CONFIGURATION ---
CONFIG = {
    # Environment Parameters
    'env': {
        'time_step': 0.1,
        'target_distance': 20.0,
        'max_distance': 100.0,
        'min_distance': 0.0,
        'actions': [-5.0, -2.0, 0.0, 2.0, 5.0]
    },
    # Discretization Parameters
    'discretizer': {
        'min_dist': 0, 'max_dist': 100, 'dist_buckets': 20,
        'min_vel': -10, 'max_vel': 10, 'vel_buckets': 20
    },
    # Training Hyperparameters (Common)
    'training': {
        'n_episodes': 3000,    # Total number of episodes
        'max_steps': 300,      # Max steps per episode (anti-loop)
        'alpha': 0.1,          # Learning Rate
        'gamma': 0.95,         # Discount Factor
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'decay_rate': 0.0015,
        'lambda': 0.9          # Specific for Sarsa(λ)
    },
    'output_dir': 'visualizations/comparison_results'
}

def get_agent(agent_name, n_states, n_actions, cfg):
    """Factory to instantiate the correct agent."""
    train_cfg = cfg['training']
    
    if agent_name == "Q-Learning":
        return QLearningAgent(n_states, n_actions, 
                              alpha=train_cfg['alpha'], gamma=train_cfg['gamma'], 
                              epsilon=train_cfg['epsilon_start'])
    
    elif agent_name == "SARSA":
        return SarsaAgent(n_states, n_actions, 
                          alpha=train_cfg['alpha'], gamma=train_cfg['gamma'], 
                          epsilon=train_cfg['epsilon_start'])
    
    elif agent_name == "SARSA(λ)":
        return SarsaLambdaAgent(n_states, n_actions, 
                                alpha=train_cfg['alpha'], gamma=train_cfg['gamma'], 
                                epsilon=train_cfg['epsilon_start'], lambd=train_cfg['lambda'])
    else:
        raise ValueError(f"Unknown agent: {agent_name}")

def train_single_agent(agent_name, env, discretizer, cfg):
    """Runs the full training loop for a single agent."""
    print(f"\n--- Starting Training: {agent_name} ---")
    
    # Setup
    train_cfg = cfg['training']
    agent = get_agent(agent_name, discretizer.n_states, len(cfg['env']['actions']), cfg)
    rewards_history = []
    
    for episode in range(train_cfg['n_episodes']):
        # Decay Epsilon
        agent.epsilon = train_cfg['epsilon_end'] + \
                        (train_cfg['epsilon_start'] - train_cfg['epsilon_end']) * \
                        np.exp(-train_cfg['decay_rate'] * episode)

        # Reset Episode
        state_raw = env.reset()
        state_idx = discretizer.get_state_index(state_raw[0], state_raw[1])
        
        # Specific handling for SARSA(λ): Reset traces
        if hasattr(agent, 'reset_traces'):
            agent.reset_traces()

        action = agent.choose_action(state_idx)

        ep_reward = 0
        done = False
        steps = 0
        
        while not done and steps < train_cfg['max_steps']:
            # 1. Execute Action
            # Note: agents return indices, env accepts float if actions is not None,
            # but your env accepts the index if actions is defined internally. 
            # Assume env.step accepts INDEX based on analyzed code.
            next_state_raw, reward, done, _ = env.step(action)
            next_state_idx = discretizer.get_state_index(next_state_raw[0], next_state_raw[1])
            
            # 2. Update Agent
            if agent_name == "Q-Learning":
                # Off-Policy: next_action not needed for update
                agent.learn(state_idx, action, reward, next_state_idx, done)
                state_idx = next_state_idx
                action = agent.choose_action(state_idx) # Choose for next step
            elif agent_name in ["SARSA", "SARSA(λ)"]:
                # On-Policy: Needs the actual next action
                next_action = agent.choose_action(next_state_idx)
                
                if agent_name == "SARSA":
                    agent.learn(state_idx, action, reward, next_state_idx, next_action)
                else:
                    agent.update(state_idx, action, reward, next_state_idx, next_action)
                
                state_idx = next_state_idx
                action = next_action

            ep_reward += reward
            steps += 1
        
        rewards_history.append(ep_reward)
        
        if (episode + 1) % 500 == 0:
            avg = np.mean(rewards_history[-100:])
            print(f"Ep {episode+1}/{train_cfg['n_episodes']} | Avg Reward: {avg:.2f} | Eps: {agent.epsilon:.3f}")

    # Saving Heatmap Specific for this Agent
    base_dir = os.path.join(cfg['output_dir'], agent_name.replace(" ", "_").lower())
    ensure_dir(base_dir)
    
    # Policy Heatmap (Greedy)
    agent.epsilon = 0.0 # Force greedy for visualization
    plot_policy_heatmap(agent, discretizer, 
                        title=f"{agent_name} Policy", 
                        save_path=os.path.join(base_dir, "policy.png"))
    
    # Value Heatmap (Optional, if the agent has q_table)
    if hasattr(agent, 'q_table'):
         plot_value_heatmap(agent, discretizer, 
                            title=f"{agent_name} Value Function", 
                            save_path=os.path.join(base_dir, "value_function.png"))

    return rewards_history

def main():
    # 0. General Setup
    # np.random.seed(42) # Uncomment for reproducibility
    ensure_dir(CONFIG['output_dir'])
    
    # Create Environment and Discretizer (Shared)
    env = ACCEnvironment(CONFIG['env'])
    discretizer = Discretizer(**CONFIG['discretizer'])
    
    # 1. Sequential Training of the 3 Agents
    results = {}
    agents_to_train = ["Q-Learning", "SARSA", "SARSA(λ)"]
    
    for name in agents_to_train:
        rewards = train_single_agent(name, env, discretizer, CONFIG)
        results[name] = rewards
    
    # 2. Final Comparative Plot
    print("\n--- Generating Final Comparative Plot ---")
    plot_learning_curve(results, 
                        save_path=os.path.join(CONFIG['output_dir'], "comparison_learning_curve.png"),
                        window=100,
                        title="ACC Algorithms Performance Comparison")

    print(f"Done! Results saved in {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()