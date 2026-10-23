import numpy as np


class Discretizer:
    """Map continuous (distance, relative velocity) states to tabular indices."""

    def __init__(
        self,
        min_dist=0,
        max_dist=100,
        dist_buckets=20,
        min_vel=-10,
        max_vel=10,
        vel_buckets=20,
    ):
        if dist_buckets < 1 or vel_buckets < 1:
            raise ValueError("Bucket counts must be positive.")
        if max_dist <= min_dist or max_vel <= min_vel:
            raise ValueError("Each maximum bound must be greater than its minimum.")

        self.dist_buckets = dist_buckets
        self.vel_buckets = vel_buckets
        # Bucket boundaries require N+1 edges to produce exactly N buckets.
        self.dist_bins = np.linspace(min_dist, max_dist, dist_buckets + 1)
        self.vel_bins = np.linspace(min_vel, max_vel, vel_buckets + 1)
        self.n_states = dist_buckets * vel_buckets

    def get_state_index(self, distance, relative_velocity):
        """Return a bounded, flattened state index in [0, n_states)."""
        d_idx = int(np.digitize(distance, self.dist_bins[1:-1]))
        v_idx = int(np.digitize(relative_velocity, self.vel_bins[1:-1]))
        d_idx = max(0, min(d_idx, self.dist_buckets - 1))
        v_idx = max(0, min(v_idx, self.vel_buckets - 1))
        return d_idx * self.vel_buckets + v_idx

    def get_num_states(self):
        return self.n_states
