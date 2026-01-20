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
        Returns a single integer index (0 to 399) representing the state.
        """
        # np.digitize returns which "bin" the value falls into
        # We subtract 1 to make it 0-indexed
        d_idx = np.digitize(distance, self.dist_bins) - 1
        v_idx = np.digitize(relative_velocity, self.vel_bins) - 1

        # Clamp values to ensure they stay within bounds (e.g. if dist > 100)
        d_idx = max(0, min(len(self.dist_bins) - 1, d_idx))
        v_idx = max(0, min(len(self.vel_bins) - 1, v_idx))

        # Flatten 2D grid to 1D index:  row * width + col
        state_idx = d_idx * len(self.vel_bins) + v_idx
        return state_idx

    def get_num_states(self):
        return self.n_states
