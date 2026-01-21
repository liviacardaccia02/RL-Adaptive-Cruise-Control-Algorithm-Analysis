import numpy as np
import os
import matplotlib.pyplot as plt

# Import Project Classes
from environment.acc_env import ACCEnvironment
from utils.discretizer import Discretizer
from utils.plotting import plot_learning_curve, plot_policy_heatmap, plot_value_heatmap, ensure_dir

# Import Agents
from agents.q_learning import QLearningAgent
from agents.sarsa import SarsaAgent
from agents.sarsa_lambda import SarsaLambdaAgent

# --- 1. CONFIGURATION & HYPERPARAMETERS ---
CONFIG = {
    # Environment Parameters
    'env': {
        'time_step': 0.1,
        'target_distance': 20.0,
        'max_distance': 100.0,
        'min_distance': 0.0,
        # TUNED ACTIONS: Finer control for stability
        'actions': [-3.0, -1.0, 0.0, 1.0, 3.0]
    },
    
    # Discretizer Parameters
    'discretizer': {
        'min_dist': 0, 
        'max_dist': 100, 
        'dist_buckets': 50,  # High resolution (2m per bucket)
        'min_vel': -20,      
        'max_vel': 20, 
        'vel_buckets': 25    
    },

    # Global Settings
    'global_training': {
        'n_episodes': 10000, 
        'max_steps': 400
    },
    
    # STATIC Base Output Directory
    'base_output_dir': 'visualizations/comparison_results_tuned',

    # --- AGENT INDIVIDUAL CONFIGURATIONS ---
    'Q-Learning': {
        'alpha': 0.1,      
        'gamma': 0.95,     # Lower gamma is fine for Q-learning here
        'epsilon_start': 1.0, 
        'epsilon_end': 0.01, 
        'decay_rate': 0.0005 
    },
    'SARSA': {
        'alpha': 0.1,      
        'gamma': 0.99,     # Higher gamma for safety
        'epsilon_start': 1.0, 
        'epsilon_end': 0.01, 
        'decay_rate': 0.0005 
    },
    'SARSA(λ)': {
        'alpha': 0.01,     # Much lower because traces accumulate error
        'gamma': 0.99,
        'lambda': 0.9,     # Trace decay factor
        'epsilon_start': 1.0, 
        'epsilon_end': 0.01, 
        'decay_rate': 0.0005 
    }
}

def get_next_run_id(base_dir):
    """
    Determines the next Run ID by checking existing comparison plots.
    Returns 1 if no previous run is found.
    """
    ensure_dir(base_dir)
    # We look for files named 'comparison_learning_curve_N.png'
    files = [f for f in os.listdir(base_dir) if f.startswith("comparison_learning_curve_") and f.endswith(".png")]
    
    ids = []
    for f in files:
        try:
            # Extract number: "comparison_learning_curve_5.png" -> "5"
            part = f.replace("comparison_learning_curve_", "").replace(".png", "")
            ids.append(int(part))
        except ValueError:
            pass
    
    if not ids:
        return 1
    return max(ids) + 1

def get_agent(agent_name, n_states, n_actions, cfg):
    """Factory to instantiate the correct agent with its specific config."""
    agent_cfg = cfg[agent_name]
    
    if agent_name == "Q-Learning":
        return QLearningAgent(n_states, n_actions, 
                              alpha=agent_cfg['alpha'], 
                              gamma=agent_cfg['gamma'], 
                              epsilon=agent_cfg['epsilon_start'])
    
    elif agent_name == "SARSA":
        return SarsaAgent(n_states, n_actions, 
                          alpha=agent_cfg['alpha'], 
                          gamma=agent_cfg['gamma'], 
                          epsilon=agent_cfg['epsilon_start'])
    
    elif agent_name == "SARSA(λ)":
        return SarsaLambdaAgent(n_states, n_actions, 
                                alpha=agent_cfg['alpha'], 
                                gamma=agent_cfg['gamma'], 
                                epsilon=agent_cfg['epsilon_start'], 
                                lambd=agent_cfg['lambda'])
    else:
        raise ValueError(f"Unknown Agent: {agent_name}")

def train_single_agent(agent_name, env, discretizer, cfg, run_id):
    """Executes the training loop for a single agent."""
    print(f"\n--- Starting Training: {agent_name} (Run #{run_id}) ---")
    
    # Load specific configs
    global_cfg = cfg['global_training']
    agent_cfg = cfg[agent_name]
    
    # Instantiate Agent
    agent = get_agent(agent_name, discretizer.n_states, len(cfg['env']['actions']), cfg)
    rewards_history = []
    
    n_episodes = global_cfg['n_episodes']
    max_steps = global_cfg['max_steps']

    for episode in range(n_episodes):
        # 1. Decay Epsilon (Custom per agent)
        agent.epsilon = agent_cfg['epsilon_end'] + \
                        (agent_cfg['epsilon_start'] - agent_cfg['epsilon_end']) * \
                        np.exp(-agent_cfg['decay_rate'] * episode)

        # 2. Reset Episode
        state_raw = env.reset()
        state_idx = discretizer.get_state_index(state_raw[0], state_raw[1])
        
        # Sarsa(λ) Specific: Reset Traces
        if hasattr(agent, 'reset_traces'):
            agent.reset_traces()

        # Choose First Action
        if hasattr(agent, 'choose_action'):
            action = agent.choose_action(state_idx)
        elif hasattr(agent, 'get_action'):
            action = agent.get_action(state_idx)
        else:
            raise AttributeError(f"Agent {agent_name} has no action selection method")

        ep_reward = 0
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            # A. Execute Action
            next_state_raw, reward, done, _ = env.step(action)
            next_state_idx = discretizer.get_state_index(next_state_raw[0], next_state_raw[1])
            
            # B. Update Agent (Handle logic differences)
            if agent_name == "Q-Learning":
                # Off-Policy: Learn then choose next action
                agent.learn(state_idx, action, reward, next_state_idx, done)
                state_idx = next_state_idx
                # Choose next action (Greedy/Epsilon)
                action = agent.choose_action(state_idx)

            elif agent_name in ["SARSA", "SARSA(λ)"]:
                # On-Policy: Choose next action THEN learn
                if hasattr(agent, 'choose_action'):
                    next_action = agent.choose_action(next_state_idx)
                else:
                    next_action = agent.get_action(next_state_idx)
                
                if agent_name == "SARSA":
                    agent.learn(state_idx, action, reward, next_state_idx, next_action)
                else: # SARSA(λ) uses 'update'
                    agent.update(state_idx, action, reward, next_state_idx, next_action)
                
                state_idx = next_state_idx
                action = next_action

            ep_reward += reward
            steps += 1
        
        rewards_history.append(ep_reward)
        
        if (episode + 1) % 500 == 0:
            avg = np.mean(rewards_history[-100:])
            # Formatted print for alignment
            print(f"Episode {episode+1:5d}/{n_episodes} | Avg Reward (last 100): {avg:7.2f} | Epsilon: {agent.epsilon:5.3f}")

    # Save Agent-Specific Plots with Numbered Filenames
    # Structure: base_dir / agent_name / policy_N.png
    agent_dir = os.path.join(cfg['base_output_dir'], agent_name.replace(" ", "_").lower())
    ensure_dir(agent_dir)
    
    # Policy Heatmap (Greedy)
    prev_eps = agent.epsilon
    agent.epsilon = 0.0
    plot_policy_heatmap(agent, discretizer, 
                        title=f"{agent_name} Policy (Run {run_id})", 
                        save_path=os.path.join(agent_dir, f"policy_{run_id}.png"))
    agent.epsilon = prev_eps
    
    # Value Heatmap (if supported)
    if hasattr(agent, 'q_table'):
         plot_value_heatmap(agent, discretizer, 
                            title=f"{agent_name} Value Function (Run {run_id})", 
                            save_path=os.path.join(agent_dir, f"value_function_{run_id}.png"))

    return rewards_history

def main():
    # 0. Setup & Reproducibility
    # np.random.seed(42)  # Uncomment for fixed results
    
    # Determine the Run ID for this session (Synchronized across all agents)
    run_id = get_next_run_id(CONFIG['base_output_dir'])
    print(f"--- Global Run ID: {run_id} ---")
    print(f"--- Saving results to: {CONFIG['base_output_dir']} ---")
    
    # Shared Environment & Discretizer
    env = ACCEnvironment(CONFIG['env'])
    discretizer = Discretizer(**CONFIG['discretizer'])
    
    # Train All Agents
    results = {}
    agents_to_train = ["Q-Learning", "SARSA", "SARSA(λ)"]
    
    for name in agents_to_train:
        # Pass the run_id to number the files consistently
        rewards = train_single_agent(name, env, discretizer, CONFIG, run_id)
        results[name] = rewards
    
    # Comparison Plot (Numbered)
    print("\n--- Generating Comparison Plot ---")
    plot_learning_curve(results, 
                        save_path=os.path.join(CONFIG['base_output_dir'], f"comparison_learning_curve_{run_id}.png"),
                        window=100,
                        title=f"ACC Algorithms Performance Comparison (Run {run_id})")

    print(f"Done! Results saved.")

if __name__ == "__main__":
    main()