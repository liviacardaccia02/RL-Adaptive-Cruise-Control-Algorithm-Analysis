import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def ensure_dir(file_path):
    """Creates the directory if it doesn't exist."""
    if file_path:
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

def plot_learning_curve(rewards_dict, save_path=None, window=100, title="Learning Curve Comparison"):
    """
    Plots learning curves for one or multiple agents.

    Args:
        rewards_dict (dict): Key=AgentName, Value=List of rewards. 
                             Ex: {'Q-Learning': [r1, r2...], 'SARSA': [r1...]}
        save_path (str, optional): Full path to save the png.
        window (int): Moving average window size.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(12, 6))
    
    # Define a color palette for distinction
    colors = sns.color_palette("husl", len(rewards_dict))

    for i, (label, rewards) in enumerate(rewards_dict.items()):
        if not rewards:
            continue
            
        # 1. Plot Raw Data (Faint)
        plt.plot(rewards, alpha=0.15, color=colors[i], linewidth=1)
        
        # 2. Plot Moving Average (Solid)
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            # Adjust x-axis for moving avg
            x_axis = np.arange(len(rewards) - len(moving_avg), len(rewards))
            plt.plot(x_axis, moving_avg, label=f"{label} (MA {window})", color=colors[i], linewidth=2)
        else:
             plt.plot(rewards, label=f"{label}", color=colors[i], linewidth=2)

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        ensure_dir(save_path)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_policy_heatmap(agent, discretizer, title="Policy Heatmap", save_path=None):
    """
    Visualizes the best action (Policy) for every state.
    """
    n_dist = len(discretizer.dist_bins)
    n_vel = len(discretizer.vel_bins)

    policy_grid = np.zeros((n_vel, n_dist))
    
    for r in range(n_vel): # Velocity (Rows)
        for c in range(n_dist): # Distance (Cols)
            # Reconstruct state index (Must match Discretizer logic!)
            # Discretizer: state_idx = d_idx * n_vel + v_idx
            state_idx = c * n_vel + r
            
            # Get best action index
            if hasattr(agent, 'q_table'):
                best_action = np.argmax(agent.q_table[state_idx])
            else:
                # Fallback for agents without direct Q-table access
                best_action = 0 
            
            policy_grid[r, c] = best_action

    plt.figure(figsize=(10, 8))
    
    # Heatmap
    ax = sns.heatmap(policy_grid, cmap="coolwarm", annot=True, fmt=".0f",
                     xticklabels=np.round(discretizer.dist_bins, 1),
                     yticklabels=np.round(discretizer.vel_bins, 1),
                     cbar_kws={'label': 'Action Index'})

    plt.xlabel("Distance to Leader (m)")
    plt.ylabel("Relative Velocity (m/s)")
    plt.title(title)
    plt.gca().invert_yaxis() # Put positive velocity on top
    plt.tight_layout()
    
    if save_path:
        ensure_dir(save_path)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_value_heatmap(agent, discretizer, title="Value Function Heatmap (Max Q)", save_path=None):
    """
    Visualizes the Value Function (Max Q-value) for every state.
    Useful to see where the agent feels 'safe' (High values) vs 'danger' (Low values).
    """
    n_dist = len(discretizer.dist_bins)
    n_vel = len(discretizer.vel_bins)
    
    value_grid = np.zeros((n_vel, n_dist))
    
    for r in range(n_vel): 
        for c in range(n_dist):
            state_idx = c * n_vel + r
            if hasattr(agent, 'q_table'):
                value_grid[r, c] = np.max(agent.q_table[state_idx])
            
    plt.figure(figsize=(10, 8))
    sns.heatmap(value_grid, cmap="viridis", annot=False,
                xticklabels=np.round(discretizer.dist_bins, 1),
                yticklabels=np.round(discretizer.vel_bins, 1),
                cbar_kws={'label': 'Max Q-Value'})

    plt.xlabel("Distance to Leader (m)")
    plt.ylabel("Relative Velocity (m/s)")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        ensure_dir(save_path)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()

# --- Legacy Compatibility for Existing Code ---
def plot_q_heatmap(agent, discretizer, title="Agent Policy Heatmap"):
    # Alias to the new function
    plot_policy_heatmap(agent, discretizer, title=title)

def plot_memory_fade(agent, discretizer, save_path=None):
    """Legacy wrapper for Eligibility Traces visualization."""
    n_dist = len(discretizer.dist_bins)
    n_vel = len(discretizer.vel_bins)
    trace_grid = np.zeros((n_vel, n_dist))

    for r in range(n_vel):
        for c in range(n_dist):
            state_idx = c * n_vel + r
            if hasattr(agent, 'e_table'):
                trace_grid[r, c] = np.max(agent.e_table[state_idx])

    plt.figure(figsize=(10, 8))
    sns.heatmap(trace_grid, cmap="magma", annot=False,
                xticklabels=np.round(discretizer.dist_bins, 1),
                yticklabels=np.round(discretizer.vel_bins, 1))

    plt.xlabel("Distance to Leader (m)")
    plt.ylabel("Relative Velocity (m/s)")
    plt.title("Active Eligibility Traces")
    plt.gca().invert_yaxis()
    
    if save_path:
        ensure_dir(save_path)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()