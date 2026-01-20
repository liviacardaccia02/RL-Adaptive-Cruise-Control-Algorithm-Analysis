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

        acceleration = self.actions[action_index]

        # Update ego vehicle speed
        self.v_ego = max(0.0, self.v_ego + acceleration * self.time_step)

        # Update leader vehicle speed (random acceleration/deceleration)
        leader_accel = np.random.uniform(-2, 2)
        self.v_leader = max(0.0, self.v_leader + leader_accel * self.time_step)

        # Update distance to leader
        # d_new = d_old + (v_leader - v_ego) * dt
        vrel = self.v_leader - self.v_ego
        self.distance += vrel * self.time_step
        
        # Update state
        self.state = np.array([self.distance, vrel], dtype=np.float32)
        
        # Calculate reward
        reward = 0
        done = False
        info = {}

        # 1. Crash
        if self.distance <= 1.0:
            reward = -100
            done = True
            info['msg'] = "CRASH"
            
        # 2. Leader Lost
        elif self.distance >= self.max_distance:
            reward = -50
            done = True
            info['msg'] = "LOST_LEADER"
            
        # 3. Normal Driving
        else:
            # Goal: Maintain safe distance (+/- 5m)
            if abs(self.distance - self.target_distance) < 2.0:
                reward += 1.0 # Optimal distance
            elif abs(self.distance - self.target_distance) <= 5.0:
                reward += 0.5 # Acceptable distance
            
            # Penalize being too close or too far
            if self.distance < self.target_distance - 5.0 or self.distance > self.target_distance + 5.0:
                reward -= 0.5
            # Penalize extreme actions
            if action_index == 0 or action_index == (len(self.actions) - 1):
                reward -= 0.5

        return self.state, reward, done, info