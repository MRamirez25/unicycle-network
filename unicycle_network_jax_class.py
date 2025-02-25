import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn

class ReadoutLayer(nn.Module):
    n_units: int
    n_outputs: int

    def setup(self):
        self.dense = nn.Dense(self.n_outputs)

    def __call__(self, x):
        return self.dense(x)

class UnicycleModel(nn.Module):
    n_units: int
    n_inp: int
    n_out: int
    dt: float
    lin_stiff_min: float
    lin_stiff_max: float
    ang_stiff_min: float
    ang_stiff_max: float
    lin_damping_min: float
    lin_damping_max: float
    ang_damping_min: float
    ang_damping_max: float
    eq_dist_min: float
    eq_dist_max: float

    def setup(self):
        self.readout_layer = ReadoutLayer(self.n_units, self.n_out)
        self.params = self.initialize_params()
        self.input_map = self.initialize_input_map()

    def initialize_params(self):
        rng = random.PRNGKey(0)
        lin_damping = random.uniform(rng, (self.n_units, 1), minval=self.lin_damping_min, maxval=self.lin_damping_max)
        ang_damping = random.uniform(rng, (self.n_units, 1), minval=self.ang_damping_min, maxval=self.ang_damping_max)
        dist_ang_coupling = random.uniform(rng, (self.n_units, self.n_units), minval=self.ang_stiff_min, maxval=self.ang_stiff_max)
        
        n_pairs = (self.n_units * (self.n_units - 1)) // 2
        stiffness_array = random.uniform(rng, (n_pairs,), minval=self.lin_stiff_min, maxval=self.lin_stiff_max)
        eq_distances_array = random.uniform(rng, (n_pairs,), minval=self.eq_dist_min, maxval=self.eq_dist_max)

        upper_indices = jnp.triu_indices(self.n_units, k=1)

        stiffness_coupling_matrix = jnp.zeros((self.n_units, self.n_units))
        eq_distances_matrix = jnp.zeros((self.n_units, self.n_units))

        stiffness_coupling_matrix = stiffness_coupling_matrix.at[upper_indices].set(stiffness_array)
        stiffness_coupling_matrix = stiffness_coupling_matrix + stiffness_coupling_matrix.T

        eq_distances_matrix = eq_distances_matrix.at[upper_indices].set(eq_distances_array)
        eq_distances_matrix = eq_distances_matrix + eq_distances_matrix.T
        eq_distances_matrix = eq_distances_matrix.reshape(self.n_units, self.n_units, 1)

        return (lin_damping, ang_damping, dist_ang_coupling, stiffness_coupling_matrix, eq_distances_matrix)

    def initialize_input_map(self):
        rng = random.PRNGKey(1)
        return random.normal(rng, (self.n_inp, self.n_units)) * 0.1  # Example initialization

    def unicycle_network_forward(self, params, input_map, u_lin, u_ang, init_state, dt):
        lin_damping, ang_damping, dist_ang_coupling, stiffness_coupling_matrix, eq_distances_matrix = params
        x, z, theta, s, omega = init_state
        
        linear_inp_forces = jnp.dot(u_lin, input_map)
        angular_inp_forces = jnp.dot(u_ang, input_map)

        coords_2d = jnp.stack((x, z), axis=-1)
        theta_unit_vectors = jnp.stack((jnp.cos(theta), jnp.sin(theta)), axis=-1)
        distance_vectors = self.pairwise_differences(coords_2d)
        distance_magnitudes = jnp.linalg.norm(distance_vectors, axis=-1, keepdims=True)
        distance_vectors_normalized = jnp.nan_to_num(distance_vectors / distance_magnitudes)

        forces_before_projection = stiffness_coupling_matrix[None, :, :, None] * (eq_distances_matrix - distance_magnitudes) * distance_vectors_normalized
        projected_forces = jnp.einsum('bijk,bik->bi', forces_before_projection, theta_unit_vectors)

        v_dot = linear_inp_forces + projected_forces - (s @ lin_damping)
        s = s + v_dot * dt

        inp_term_theta = angular_inp_forces
        ang_distances = theta[:, :, None] - theta[:, None, :]
        coupling_term_ang = jnp.sum(dist_ang_coupling[None, :, :] * (-ang_distances), axis=2)

        omega_dot = (inp_term_theta + coupling_term_ang) - omega @ ang_damping
        omega = omega + omega_dot * dt
        theta = theta + dt * omega
        x = x + jnp.cos(theta) * s
        z = z + jnp.sin(theta) * s

        return x, z, theta, s, omega

    def pairwise_differences(self, arr):
        expanded_arr1 = arr[:, :, None, :]  # (batch, n_units, 1, 2)
        expanded_arr2 = arr[:, None, :, :]  # (batch, 1, n_units, 2)
        return expanded_arr1 - expanded_arr2

    def forward(self, u_lin, u_ang, init_state, dt, steps):
        state = init_state
        for _ in range(steps):
            state = self.unicycle_network_forward(self.params, self.input_map, u_lin, u_ang, state, dt)
        final_state = state
        readout_output = self.readout_layer(jnp.concatenate(final_state))
        return readout_output, final_state
    
    def __call__(self, u_lin, u_ang, init_state, dt, steps):
        readout_output, final_state = self.forward(u_lin, u_ang, init_state, dt, steps)
        return readout_output, final_state

    def loss_fn(self, u_lin, u_ang, init_state, target, dt, steps):
        readout_output, final_state = self.forward(u_lin, u_ang, init_state, dt, steps)
        return jnp.mean((readout_output - target) ** 2)  # Example loss function (MSE)

# # Example usage
# n_units = 100
# n_inp = 1
# n_out = 10
# dt = 0.01

# model = UnicycleModel(
#     n_units=n_units,
#     n_inp=n_inp,
#     n_out=n_out,
#     dt=dt,
#     lin_stiff_min=0.5,
#     lin_stiff_max=1.0,
#     ang_stiff_min=0.1,
#     ang_stiff_max=0.3,
#     lin_damping_min=0.1,
#     lin_damping_max=0.2,
#     ang_damping_min=0.1,
#     ang_damping_max=0.2,
#     eq_dist_min=0.5,
#     eq_dist_max=1.0
# )

# # Now you can call the forward method and compute loss
# u_lin = jnp.zeros((1, n_inp))  # Example input
# u_ang = jnp.zeros((1, n_inp))  # Example input
# init_state = (
#     random.uniform(0.1, 0.6, (n_units,)),
#     random.uniform(0.1, 0.2, (n_units,)),
#     random.uniform(-jnp.pi, jnp.pi, (n_units,)),
#     random.uniform(0.1, 1.0, (n_units,)),
#     random.uniform(-1.0, 1.0, (n_units,))
# )

# # Compute loss
# target = jnp.zeros((n_out,))  # Example target
# loss = model.loss_fn(u_lin, u_ang, init_state, target, dt, 100)
# print("Loss:", loss)
