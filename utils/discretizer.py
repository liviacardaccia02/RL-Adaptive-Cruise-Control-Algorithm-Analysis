import numpy as np

class Discretizer:
    def __init__(self,
                 min_dist=0, max_dist=100, dist_buckets=20,
                 min_vel=-10, max_vel=10, vel_buckets=20):
        """
        Maps continuous state (distance, velocity) to a discrete index.
        Table size = 20 * 20 = 400 states.
        """
        self.dist_bins = np.linspace(min_dist, max_dist, dist_buckets)
        self.vel_bins = np.linspace(min_vel, max_vel, vel_buckets)
        self.n_states = dist_buckets * vel_buckets

    def get_state_index(self, distance, relative_velocity):
        """
        Maps continuous state to discrete index with boundary safety.
        """
        # 1. Get indices (np.digitize returns 1..N usually)
        d_idx = np.digitize(distance, self.dist_bins) - 1
        v_idx = np.digitize(relative_velocity, self.vel_bins) - 1

        # 2. Safety Clamp
        # Ensures index is between 0 and (num_buckets - 1)
        d_idx = max(0, min(d_idx, len(self.dist_bins) - 2))
        v_idx = max(0, min(v_idx, len(self.vel_bins) - 2))

        # 3. Flatten mapping
        # state_idx = row * n_cols + col
        return d_idx * (len(self.vel_bins) - 1) + v_idx

    def get_num_states(self):
        return self.n_states
