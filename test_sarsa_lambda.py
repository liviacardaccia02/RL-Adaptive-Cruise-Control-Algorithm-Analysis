from utils.discretizer import Discretizer
from agents.sarsa_lambda import SarsaLambdaAgent
import numpy as np

# 1. Setup
discretizer = Discretizer()
n_states = discretizer.get_num_states()
n_actions = 5 # (Hard Brake, Light Brake, Coast, Light Accel, Hard Accel)

agent = SarsaLambdaAgent(n_states, n_actions, lambd=0.8)

print("--- Starting Test ---")

# 2. Simulate a small "episode"
# Assume we start at distance 50m, relative velocity 0
state_idx = discretizer.get_state_index(50, 0)
action = agent.get_action(state_idx)
agent.reset_traces()

for step in range(3):
    print(f"\nStep {step}:")
    print(f"Current State Index: {state_idx}, Action Taken: {action}")
    
    # Mock mechanics: We move closer (dist decreases)
    # Next state: distance 40m, same velocity
    next_dist = 50 - ((step + 1) * 10) 
    next_state_idx = discretizer.get_state_index(next_dist, 0)
    
    # Mock Reward: -1 per second
    reward = -1 
    
    # Agent chooses next action
    next_action = agent.get_action(next_state_idx)
    
    # UPDATE HAPPENS HERE
    print("Updating Q-table and Traces...")
    agent.update(state_idx, action, reward, next_state_idx, next_action)
    
    # Verify Trace: The state we just left should have a trace > 0
    trace_val = agent.e_table[state_idx, action]
    print(f"Trace value for state {state_idx}: {trace_val:.4f}")
    
    if trace_val > 0:
        print("SUCCESS: Eligibility trace is active!")
    else:
        print("ERROR: Trace not updating.")

    # Move to next step
    state_idx = next_state_idx
    action = next_action

print("\n--- Test Complete ---")