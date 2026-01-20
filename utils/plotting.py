import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- SECTION 1: TYTUS'S VISUALIZATIONS (Eligibility Traces) ---

def plot_q_heatmap(agent, discretizer, title="Agent Policy Heatmap (Actions)"):
    """
    Visualizes the best action for every combination of Distance and Velocity.

    Args:
        agent: The RL agent (must have q_table accessible).
        discretizer: The helper class to map indices back to labels.
    """
    n_dist = len(discretizer.dist_bins)
    n_vel = len(discretizer.vel_bins)

    # Create a 2D grid to store the best action for each state
    # Grid shape: (Velocity Rows, Distance Columns)
    policy_grid = np.zeros((n_vel, n_dist))

    for r in range(n_vel): # Velocity (Rows)
        for c in range(n_dist): # Distance (Cols)
            # Reconstruct the state index
            # Note: We must match the logic in Discretizer.get_state_index
            state_idx = c * n_vel + r

            # Get best action (Argmax)
            best_action = np.argmax(agent.q_table[state_idx])
            policy_grid[r, c] = best_action

    # Plotting
    plt.figure(figsize=(10, 8))
    # We flip the Y-axis so positive velocity is up
    ax = sns.heatmap(policy_grid, cmap="coolwarm", annot=True, fmt=".0f",
                     xticklabels=np.round(discretizer.dist_bins, 1),
                     yticklabels=np.round(discretizer.vel_bins, 1))

    plt.xlabel("Distance to Leader (m)")
    plt.ylabel("Relative Velocity (m/s)")
    plt.title(title)
    plt.gca().invert_yaxis() # Ensure logic matches visual (Up = Fast)
    plt.show()

def plot_memory_fade(agent, discretizer):
    """
    Visualizes the 'Memory Fade' (Eligibility Traces).
    Shows which states the agent is currently 'thinking about' (attributing credit to).
    """
    n_dist = len(discretizer.dist_bins)
    n_vel = len(discretizer.vel_bins)

    # Max trace value across all actions for each state
    trace_grid = np.zeros((n_vel, n_dist))

    for r in range(n_vel):
        for c in range(n_dist):
            state_idx = c * n_vel + r
            # We take the max trace value (strongest memory for that state)
            trace_grid[r, c] = np.max(agent.e_table[state_idx])

    plt.figure(figsize=(10, 8))
    sns.heatmap(trace_grid, cmap="magma", annot=False,
                xticklabels=np.round(discretizer.dist_bins, 1),
                yticklabels=np.round(discretizer.vel_bins, 1))

    plt.xlabel("Distance to Leader (m)")
    plt.ylabel("Relative Velocity (m/s)")
    plt.title("Active Eligibility Traces (Memory Fade)")
    plt.gca().invert_yaxis()
    plt.show()


# --- SECTION 2: SHARED COMPARISON PLOTS (Placeholders) ---

def plot_learning_curve(rewards_sarsa_lambda, rewards_q_learning=None, rewards_sarsa=None):
    """
    Compares the learning progress of different agents.

    Args:
        rewards_sarsa_lambda (list): List of total rewards per episode for Agent 3.
        rewards_q_learning (list): Placeholder for Agent 1 data.
        rewards_sarsa (list): Placeholder for Agent 2 data.
    """
    plt.figure(figsize=(12, 6))

    # Plot Sarsa(λ) (Your Agent)
    plt.plot(rewards_sarsa_lambda, label="Sarsa(λ)", color='blue', alpha=0.8)

    # Placeholders for teammates
    if rewards_q_learning is not None:
        plt.plot(rewards_q_learning, label="Q-Learning", color='green', alpha=0.6)
    else:
        print("Info: Q-Learning data not provided yet.")

    if rewards_sarsa is not None:
        plt.plot(rewards_sarsa, label="SARSA", color='red', alpha=0.6)
    else:
        print("Info: SARSA data not provided yet.")

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Learning Curve Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_safety_margin(distances_sarsa_lambda, distances_q_learning=None):
    """
    Compare how well agents maintain the safety distance.
    TODO: Q-Learning and SARSA agents should pass their list of recorded distances here.
    """
    pass # Placeholder for future implementation
