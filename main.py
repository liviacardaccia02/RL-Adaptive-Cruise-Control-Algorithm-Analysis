import numpy as np
import os
import matplotlib.pyplot as plt
import csv

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
    'env': {
        'time_step': 0.1,
        'target_distance': 20.0,
        'max_distance': 100.0,
        'min_distance': 0.0,
        'actions': [-3.0, -1.0, 0.0, 1.0, 3.0]
    },
    'discretizer': {
        'min_dist': 0, 'max_dist': 100, 'dist_buckets': 50,
        'min_vel': -20, 'max_vel': 20, 'vel_buckets': 25    
    },
    'global_training': {
        'n_episodes': 10000, 
        'max_steps': 400
    },
    'base_output_dir': 'results/comparison_results_tuned',
    'log_dir': 'results/logs',
    
    'Q-Learning': { 'alpha': 0.1, 'gamma': 0.95, 'epsilon_start': 1.0, 'epsilon_end': 0.01, 'decay_rate': 0.0005 },
    'SARSA': { 'alpha': 0.1, 'gamma': 0.99, 'epsilon_start': 1.0, 'epsilon_end': 0.01, 'decay_rate': 0.0005 },
    'SARSA(λ)': { 'alpha': 0.01, 'gamma': 0.99, 'lambda': 0.9, 'epsilon_start': 1.0, 'epsilon_end': 0.01, 'decay_rate': 0.0005 }
}

def get_next_run_id(base_dir):
    """
    Determines the next run ID based on existing files in the base directory.
    
    Args:
        base_dir (str): Directory where result files are stored.

    Returns:
        int: Next available run ID.
    """
    ensure_dir(base_dir)
    files = [f for f in os.listdir(base_dir) if f.startswith("comparison_learning_curve_") and f.endswith(".png")]
    ids = []
    for f in files:
        try:
            part = f.replace("comparison_learning_curve_", "").replace(".png", "")
            ids.append(int(part))
        except ValueError:
            pass
    return max(ids) + 1 if ids else 1

def get_agent(agent_name, n_states, n_actions, cfg):
    """
    Factory function to create agents based on name.
    
    Args
        agent_name (str): Name of the agent ("Q-Learning", "SARSA", "SARSA(λ)")
        n_states (int): Number of discrete states.
        n_actions (int): Number of discrete actions.
        cfg (dict): Configuration dictionary with hyperparameters.

    Returns:
        Instantiated agent object.
    """
    agent_cfg = cfg[agent_name]
    if agent_name == "Q-Learning":
        return QLearningAgent(n_states, n_actions, alpha=agent_cfg['alpha'], gamma=agent_cfg['gamma'], epsilon=agent_cfg['epsilon_start'])
    elif agent_name == "SARSA":
        return SarsaAgent(n_states, n_actions, alpha=agent_cfg['alpha'], gamma=agent_cfg['gamma'], epsilon=agent_cfg['epsilon_start'])
    elif agent_name == "SARSA(λ)":
        return SarsaLambdaAgent(n_states, n_actions, alpha=agent_cfg['alpha'], gamma=agent_cfg['gamma'], epsilon=agent_cfg['epsilon_start'], lambd=agent_cfg['lambda'])
    else:
        raise ValueError(f"Unknown Agent: {agent_name}")

def train_single_agent(agent_name, env, discretizer, cfg, run_id):
    """
    Trains a single agent and returns rewards history and KPI stats.
    
    Args
        agent_name (str): Name of the agent to train.
        env (ACCEnvironment): The ACC environment instance.
        discretizer (Discretizer): Discretizer instance for state mapping.
        cfg (dict): Configuration dictionary.
        run_id (int): Unique identifier for the training run.

    Returns:
        tuple: (rewards_history (list), kpi_stats (dict))
    """
    print(f"\n--- Starting Training: {agent_name} (Run #{run_id}) ---")
    global_cfg = cfg['global_training']
    agent_cfg = cfg[agent_name]
    
    agent = get_agent(agent_name, discretizer.n_states, len(cfg['env']['actions']), cfg)
    rewards_history = []
    
    crashes = 0
    lost_leaders = 0
    
    n_episodes = global_cfg['n_episodes']
    max_steps = global_cfg['max_steps']

    for episode in range(n_episodes):
        agent.epsilon = agent_cfg['epsilon_end'] + (agent_cfg['epsilon_start'] - agent_cfg['epsilon_end']) * np.exp(-agent_cfg['decay_rate'] * episode)
        state_raw = env.reset()
        state_idx = discretizer.get_state_index(state_raw[0], state_raw[1])
        
        if hasattr(agent, 'reset_traces'): agent.reset_traces()
        
        if hasattr(agent, 'choose_action'): action = agent.choose_action(state_idx)
        elif hasattr(agent, 'get_action'): action = agent.get_action(state_idx)
        else: raise AttributeError(f"Agent {agent_name} has no action selection method")

        ep_reward = 0
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            next_state_raw, reward, done, info = env.step(action)
            next_state_idx = discretizer.get_state_index(next_state_raw[0], next_state_raw[1])
            
            if done:
                msg = info.get('msg', '')
                if msg == "CRASH": crashes += 1
                elif msg == "LOST_LEADER": lost_leaders += 1

            if agent_name == "Q-Learning":
                agent.learn(state_idx, action, reward, next_state_idx, done)
                state_idx = next_state_idx
                action = agent.choose_action(state_idx)
            elif agent_name in ["SARSA", "SARSA(λ)"]:
                if hasattr(agent, 'choose_action'): next_action = agent.choose_action(next_state_idx)
                else: next_action = agent.get_action(next_state_idx)
                
                if agent_name == "SARSA": agent.learn(state_idx, action, reward, next_state_idx, next_action)
                else: agent.update(state_idx, action, reward, next_state_idx, next_action)
                
                state_idx = next_state_idx
                action = next_action

            ep_reward += reward
            steps += 1
        
        rewards_history.append(ep_reward)
        if (episode + 1) % 500 == 0:
            avg = np.mean(rewards_history[-100:])
            print(f"Episode {episode+1:5d}/{n_episodes} | Avg Reward (last 100): {avg:7.2f} | Epsilon: {agent.epsilon:5.3f}")

    # Stats Calculation
    total_avg_reward = np.mean(rewards_history)
    crash_rate = (crashes / n_episodes) * 100
    loss_rate = (lost_leaders / n_episodes) * 100
    survival_rate = 100.0 - crash_rate - loss_rate

    kpi_stats = {
        "Run ID": run_id,
        "Agent": agent_name,
        "Episodes": n_episodes,
        "Avg Reward": round(total_avg_reward, 2),
        "Crash Rate (%)": round(crash_rate, 2),
        "Lost Leader Rate (%)": round(loss_rate, 2),
        "Survival Rate (%)": round(survival_rate, 2)
    }

    # Plotting
    agent_dir = os.path.join(cfg['base_output_dir'], agent_name.replace(" ", "_").lower())
    ensure_dir(agent_dir)
    
    prev_eps = agent.epsilon
    agent.epsilon = 0.0
    plot_policy_heatmap(agent, discretizer, title=f"{agent_name} Policy (Run {run_id})", save_path=os.path.join(agent_dir, f"policy_{run_id}.png"))
    agent.epsilon = prev_eps
    
    if hasattr(agent, 'q_table'):
         plot_value_heatmap(agent, discretizer, title=f"{agent_name} Value Function (Run {run_id})", save_path=os.path.join(agent_dir, f"value_function_{run_id}.png"))

    return rewards_history, kpi_stats

def main():
    run_id = get_next_run_id(CONFIG['base_output_dir'])
    print(f"--- Global Run ID: {run_id} ---")
    
    log_dir = CONFIG['log_dir']
    ensure_dir(log_dir)
    log_file_path = os.path.join(log_dir, "kpi_log.csv")
    
    csv_headers = ["Run ID", "Agent", "Episodes", "Avg Reward", "Crash Rate (%)", "Lost Leader Rate (%)", "Survival Rate (%)"]
    
    file_exists = os.path.isfile(log_file_path)

    with open(log_file_path, mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
        
        if not file_exists:
            writer.writeheader()

        env = ACCEnvironment(CONFIG['env'])
        discretizer = Discretizer(**CONFIG['discretizer'])
        results = {}
        agents_to_train = ["Q-Learning", "SARSA", "SARSA(λ)"]
        
        for name in agents_to_train:
            rewards, stats = train_single_agent(name, env, discretizer, CONFIG, run_id)
            results[name] = rewards
            writer.writerow(stats)
            print(f"Stats logged for {name}")
    
    print(f"--- KPI Logs updated in {log_file_path} ---")

    print("\n--- Generating Comparison Plot ---")
    plot_learning_curve(results, save_path=os.path.join(CONFIG['base_output_dir'], f"comparison_learning_curve_{run_id}.png"), window=100, title=f"ACC Algorithms Performance Comparison")
    print(f"Done!")

if __name__ == "__main__":
    main()