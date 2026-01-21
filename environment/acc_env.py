import numpy as np  

class ACCEnvironment: 
    def __init__(self, config):
        """Initialize the ACC environment with given configuration."""

        self.config = config

        # Configuration parameters
        self.time_step = config.get('time_step', 0.1)  # seconds
        self.target_distance = config.get('target_distance', 10.0)  # meters
        self.max_distance = config.get('max_distance', 100.0)  # meters
        self.min_distance = config.get('min_distance', 5.0)  # meters

        # Actions [Hard brake, Brake, Maintain, Accelerate, Hard Accelerate]
        self.actions = config.get('actions', [-5.0, -2.0, 0.0, 2.0, 5.0])  # m/s^2

        # State variables
        self.v_ego = 0.0  # Ego vehicle speed (m/s)
        self.v_leader = 0.0  # Leader vehicle speed (m/s)
        self.distance = 0.0  # Actual distance to target vehicle (m)
        self.state = None  # Current state
        

    def reset(self):
        """Reset the environment to an initial state."""

        self.v_ego = np.random.uniform(0, 30)  # Random ego speed between 0 and 30 m/s
        self.v_leader = np.random.uniform(0, 30)  # Random target speed between 0 and 30 m/s
        self.distance = np.random.uniform(self.min_distance, self.max_distance)  # Random initial distance
        
        vrel = self.v_leader - self.v_ego
        self.state = np.array([self.distance, vrel], dtype=np.float32)
        return self.state
    

    def step(self, action_index):
        """Take an action and update the environment state.

        Args:
            action_index (int): Index of the action to take. 
        Returns:
            state (np.array): The new state after taking the action.    
            reward (float): The reward obtained from taking the action.
            done (bool): Whether the episode has ended.
        """
        
        # 1. Action & Physics
        acceleration = self.actions[action_index]

        # Update ego vehicle speed
        self.v_ego = max(0.0, self.v_ego + acceleration * self.time_step)

        # Update leader vehicle speed (random acceleration/deceleration)
        leader_accel = np.random.uniform(-2, 2)
        self.v_leader = max(0.0, self.v_leader + leader_accel * self.time_step)

        # Update distance to leader
        vrel = self.v_leader - self.v_ego
        self.distance += vrel * self.time_step
        
        # Update state
        self.state = np.array([self.distance, vrel], dtype=np.float32)
        
        # 2. Dense Reward Calculation
        reward = 0
        done = False
        info = {}
        
        # Normalize distance error (0.0 to ~1.0)
        # We want to be at target_distance.
        dist_error = abs(self.distance - self.target_distance) / self.max_distance
        
        # Base Reward: Negative error (Higher is better, max 0)
        # This guides the agent at every single step, not just at the end.
        reward = -dist_error

        # Critical Events
        if self.distance <= 0: # Crash
            reward = -10.0 # Large penalty relative to step reward
            done = True
            info['msg'] = "CRASH"
            
        elif self.distance >= self.max_distance: # Lost Leader
            reward = -10.0
            done = True
            info['msg'] = "LOST_LEADER"
            
        else:
            # Bonus: If we are very close to target (within 5% error), give a boost
            # This encourages the agent to "stick" to the target
            if dist_error < 0.05: 
                reward += 0.5
                
        return self.state, reward, done, info