from typing import Dict, Optional
import jax
import jax.numpy as jnp
from jax import lax
from flax import linen as nn
import numpy as np


def pairwise_differences(arr):
    return arr[:, :, None, :] - arr[:, None, :, :]


def angle_to_unit_vector(angle):
    return jnp.stack((jnp.cos(angle), jnp.sin(angle)), axis=-1)


class UnicycleReservoir(nn.Module):
    n_inp: int
    n_units: int
    n_out: int
    dt: float
    seed: int = 0

    # min/max ranges for uniform random params
    lin_damping_min: float = 0.0
    lin_damping_max: float = 0.1
    ang_damping_min: float = 0.0
    ang_damping_max: float = 0.1

    lin_stiff_min: float = 0.0
    lin_stiff_max: float = 1.0
    ang_stiff_min: float = 0.0
    ang_stiff_max: float = 1.0

    eq_dist_min: float = 0.1
    eq_dist_max: float = 1.0
    eq_dist_min_ang: float = -1.0
    eq_dist_max_ang: float = 1.0

    inp_bias: float = 0.1

    params_dict: Optional[Dict[str, jnp.ndarray]] = None  # <-- New optional dict for custom params

    def setup(self):
        key = jax.random.PRNGKey(self.seed)
        k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)

        pd = self.params_dict or {}

        def get(name, shape, key, minval=None, maxval=None, symmetric=False, antisymmetric=False):
            if name in pd:
                return pd[name]
            if minval is not None and maxval is not None:
                val = jax.random.uniform(key, shape, minval=minval, maxval=maxval)
            else:
                val = jax.random.normal(key, shape)
            if symmetric:
                val = (val + val.T) / 2
                val = val * (1 - jnp.eye(shape[0]))  # zero diagonal
            if antisymmetric:
                val = val - val.T
            return val

        self.lin_damping = get("lin_damping", (1, self.n_units), k1, self.lin_damping_min, self.lin_damping_max)
        self.ang_damping = get("ang_damping", (1, self.n_units), k2, self.ang_damping_min, self.ang_damping_max)

        self.mass_vector = get("mass_vector", (1, self.n_units), k3)
        self.mass_vector = self.mass_vector.at[:, 0].set(0.0)

        self.j_vector = get("j_vector", (1, self.n_units), k4)
        self.j_vector = self.j_vector.at[:, 0].set(0.0)

        self.stiffness_coupling_matrix = get("stiffness_coupling_matrix", (self.n_units, self.n_units), k5,
                                             self.lin_stiff_min, self.lin_stiff_max, symmetric=True)
        self.dist_ang_coupling = get("dist_ang_coupling", (self.n_units, self.n_units), k6,
                                     self.ang_stiff_min, self.ang_stiff_max, symmetric=True)

        eq_d = get("eq_distances_matrix", (self.n_units, self.n_units), k7,
                   self.eq_dist_min, self.eq_dist_max, symmetric=True)
        self.eq_distances_matrix = eq_d[:, :, None]  # shape (n_units, n_units, 1)

        eq_ang = get("eq_distances_mat_ang", (self.n_units, self.n_units), k7,
                     self.eq_dist_min_ang, self.eq_dist_max_ang, antisymmetric=True)
        self.eq_distances_mat_ang = eq_ang

        self.lin_input_map = get("lin_input_map", (self.n_inp, self.n_units), k6)
        self.ang_input_map = get("ang_input_map", (self.n_inp, self.n_units), k7)

        # Use inp_bias from params_dict if provided, else default attribute
        self.inp_bias = pd.get("inp_bias", self.inp_bias)

        self.readout = nn.Dense(self.n_out)

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

    def __call__(self, u_lin_seq, u_ang_seq, init_state):
        """
        Args:
            u_lin_seq: (T, B, n_inp)
            u_ang_seq: (T, B, n_inp)
            init_state: Tuple of 5 (x, z, theta, s, omega), each (B, n_units)
        Returns:
            states: (T, B, 5*n_units)
            readout_output: (B, n_out)
        """
        scan_out = lax.scan(self._step, init_state, (u_lin_seq, u_ang_seq))
        states_over_time = scan_out[1]  # (T, B, 5*n_units)

        if self.n_past_steps_readout > 0:
            T = states_over_time.shape[0]
            idxs = jnp.linspace(0, T - 1, self.n_past_steps_readout + 1, dtype=int)
            selected_states = states_over_time[idxs]  # (K, B, F)
            flat = selected_states.transpose(1, 0, 2).reshape(states_over_time.shape[1], -1)
        else:
            flat = states_over_time[-1]  # (B, F)

        return states_over_time, self.readout(flat)