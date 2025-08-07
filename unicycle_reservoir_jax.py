import jax
import jax.numpy as jnp
from jax import lax
from flax import linen as nn

def angle_to_unit_vector(theta):
    return jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)

def pairwise_differences(coords):
    # coords: (B, n_units, 2)
    diff = coords[:, :, None, :] - coords[:, None, :, :]  # (B, n_units, n_units, 2)
    return diff

class UnicycleReservoir:
    def __init__(self, *, n_inp, n_units, dt, seed=0,
                 lin_damping_min=0.0, lin_damping_max=0.1,
                 ang_damping_min=0.0, ang_damping_max=0.1,
                 lin_stiff_min=0.0, lin_stiff_max=1.0,
                 ang_stiff_min=0.0, ang_stiff_max=1.0,
                 eq_dist_min=0.1, eq_dist_max=1.0,
                 n_connections=2, n_connections_anchor=1,
                 eq_dist_min_ang=-1.0, eq_dist_max_ang=1.0,
                 n_connections_ang=2, n_connections_anchor_ang=1,
                 inp_bias=0.1,
                 n_past_steps_readout=0):

        self.n_inp = n_inp
        self.n_units = n_units
        self.dt = dt
        self.seed = seed
        self.lin_damping_min = lin_damping_min
        self.lin_damping_max = lin_damping_max
        self.ang_damping_min = ang_damping_min
        self.ang_damping_max = ang_damping_max
        self.lin_stiff_min = lin_stiff_min
        self.lin_stiff_max = lin_stiff_max
        self.ang_stiff_min = ang_stiff_min
        self.ang_stiff_max = ang_stiff_max
        self.eq_dist_min = eq_dist_min
        self.eq_dist_max = eq_dist_max
        self.n_connections = n_connections
        self.n_connections_anchor = n_connections_anchor
        self.eq_dist_min_ang = eq_dist_min_ang
        self.eq_dist_max_ang = eq_dist_max_ang
        self.n_connections_ang = n_connections_ang
        self.n_connections_anchor_ang = n_connections_anchor_ang
        self.inp_bias = inp_bias
        self.n_past_steps_readout = n_past_steps_readout

        self._init_params()

    def _init_params(self):
        key = jax.random.PRNGKey(self.seed)
        k1, k2, k3, k4, k5, k6, k7, k8 = jax.random.split(key, 8)
        nu = self.n_units

        def rand_uniform(key, shape, minval, maxval):
            return jax.random.uniform(key, shape, minval=minval, maxval=maxval)

        # Linear damping
        self.lin_damping = rand_uniform(k1, (1, nu), self.lin_damping_min, self.lin_damping_max)

        # Angular damping
        self.ang_damping = rand_uniform(k2, (1, nu), self.ang_damping_min, self.ang_damping_max)

        # Mass vector (first unit = 0)
        mass_vector = jnp.ones((1, nu))
        mass_vector = mass_vector.at[0, 0].set(0.0)
        self.mass_vector = mass_vector

        # Inertia vector (first unit = 0)
        j_vector = jnp.ones((1, nu))
        j_vector = j_vector.at[0, 0].set(0.0)
        self.j_vector = j_vector

        # Stiffness coupling matrix (symmetric, sparse)
        stiffnesses_array = rand_uniform(k3, nu * (nu - 1) // 2, self.lin_stiff_min, self.lin_stiff_max)
        stiffness_matrix = jnp.zeros((nu, nu))
        idx = 0
        for i in range(nu):
            for j in range(i + 1, nu):
                if i == 0:
                    if abs(i - j - 1) < self.n_connections_anchor:
                        stiffness_matrix = stiffness_matrix.at[i, j].set(stiffnesses_array[idx])
                        stiffness_matrix = stiffness_matrix.at[j, i].set(stiffnesses_array[idx])
                else:
                    if abs(i - j - 1) < self.n_connections:
                        stiffness_matrix = stiffness_matrix.at[i, j].set(stiffnesses_array[idx])
                        stiffness_matrix = stiffness_matrix.at[j, i].set(stiffnesses_array[idx])
                idx += 1
        self.stiffness_coupling_matrix = stiffness_matrix

        # Angular distance coupling (symmetric, sparse)
        ang_stiff_array = rand_uniform(k4, nu * (nu - 1) // 2, self.ang_stiff_min, self.ang_stiff_max)
        ang_matrix = jnp.zeros((nu, nu))
        idx = 0
        for i in range(nu):
            for j in range(i + 1, nu):
                if i == 0:
                    if abs(i - j - 1) < self.n_connections_anchor_ang:
                        ang_matrix = ang_matrix.at[i, j].set(ang_stiff_array[idx])
                        ang_matrix = ang_matrix.at[j, i].set(ang_stiff_array[idx])
                else:
                    if abs(i - j - 1) < self.n_connections_ang:
                        ang_matrix = ang_matrix.at[i, j].set(ang_stiff_array[idx])
                        ang_matrix = ang_matrix.at[j, i].set(ang_stiff_array[idx])
                idx += 1
        self.dist_ang_coupling = ang_matrix

        # Equilibrium distance matrix (symmetric)
        eq_dist_array = rand_uniform(k5, nu * (nu - 1) // 2, self.eq_dist_min, self.eq_dist_max)
        eq_matrix = jnp.zeros((nu, nu))
        idx = 0
        for i in range(nu):
            for j in range(i + 1, nu):
                eq_matrix = eq_matrix.at[i, j].set(eq_dist_array[idx])
                eq_matrix = eq_matrix.at[j, i].set(eq_dist_array[idx])
                idx += 1
        eq_matrix = eq_matrix[:, :, None]  # Add singleton dim
        self.eq_distances_matrix = eq_matrix

        # Equilibrium angular distances (antisymmetric)
        eq_ang_array = rand_uniform(k6, nu * (nu - 1) // 2, self.eq_dist_min_ang, self.eq_dist_max_ang)
        eq_ang_matrix = jnp.zeros((nu, nu))
        idx = 0
        for i in range(nu):
            for j in range(i + 1, nu):
                eq_ang_matrix = eq_ang_matrix.at[i, j].set(eq_ang_array[idx])
                eq_ang_matrix = eq_ang_matrix.at[j, i].set(-eq_ang_array[idx])
                idx += 1
        self.eq_distances_mat_ang = eq_ang_matrix

        # Input maps
        self.lin_input_map = jax.random.normal(k7, (self.n_inp, nu)) * 0.1
        self.ang_input_map = jax.random.normal(k8, (self.n_inp, nu)) * 0.1


    def _step(self, carry, inputs):
        x, z, theta, s, omega = carry
        u_lin_t, u_ang_t = inputs

        linear_inp_forces = jnp.tanh(u_lin_t + self.inp_bias) @ self.lin_input_map
        angular_inp_forces = u_ang_t @ self.ang_input_map

        coords_2d = jnp.stack((x, z), axis=-1)
        theta_vecs = angle_to_unit_vector(theta)
        distance_vectors = pairwise_differences(coords_2d)
        distance_magnitudes = jnp.linalg.norm(distance_vectors, axis=-1, keepdims=True)
        distance_vectors_normalized = jnp.nan_to_num(distance_vectors / distance_magnitudes)

        forces = self.stiffness_coupling_matrix[None, :, :, None] * \
                 (self.eq_distances_matrix - distance_magnitudes) * distance_vectors_normalized
        projected_forces = jnp.einsum('bijk,bik->bi', forces, theta_vecs)

        v_dot = (linear_inp_forces + projected_forces - s * self.lin_damping) * self.mass_vector
        s = s + v_dot * self.dt

        theta_diff = theta[:, :, None] - theta[:, None, :]
        coupling_term_ang = jnp.sum(self.dist_ang_coupling[None, :, :] *
                                    (self.eq_distances_mat_ang[None, :, :] - theta_diff), axis=-1)
        omega_dot = (angular_inp_forces + coupling_term_ang - omega * self.ang_damping) * self.j_vector
        omega = omega + omega_dot * self.dt

        theta = theta + omega * self.dt
        x = x + jnp.cos(theta) * s
        z = z + jnp.sin(theta) * s

        new_state = (x, z, theta, s, omega)
        output = jnp.hstack((x, z, theta, s, omega))  # (batch, 5*n_units)
        return new_state, output

    def integrate_dynamics(self, u_lin_seq, u_ang_seq, init_state):
        """
        Args:
            u_lin_seq: (T, B, n_inp)
            u_ang_seq: (T, B, n_inp)
            init_state: tuple of reservoir states (B, n_units)
        Returns:
            states_over_time: (T, B, 5 * n_units)
        """
        _, states_over_time = lax.scan(self._step, init_state, (u_lin_seq, u_ang_seq))
        return states_over_time

    def get_readout_input(self, states_over_time):
        if self.n_past_steps_readout > 0:
            T = states_over_time.shape[0]
            idxs = jnp.linspace(0, T - 1, self.n_past_steps_readout + 1, dtype=int)
            selected_states = states_over_time[idxs]  # (K, B, F)
            flat = selected_states.transpose(1, 0, 2).reshape(states_over_time.shape[1], -1)
        else:
            flat = states_over_time[-1]
        return flat

    
class Readout(nn.Module):
    n_out: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.n_out)(x)