import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax, random
import time

def initialize_params(rng, n_units, n_inp, lin_stiff_min, lin_stiff_max, ang_stiff_min, ang_stiff_max,
                      lin_damping_min, lin_damping_max, ang_damping_min, ang_damping_max, eq_dist_min, eq_dist_max):
    lin_damping = random.uniform(rng, (n_units, 1), minval=lin_damping_min, maxval=lin_damping_max)
    ang_damping = random.uniform(rng, (n_units, 1), minval=ang_damping_min, maxval=ang_damping_max)
    dist_ang_coupling = random.uniform(rng, (n_units, n_units), minval=ang_stiff_min, maxval=ang_stiff_max)

    # stiffnesses_array = random.uniform(rng, (int((n_units**2 - n_units) / 2),), minval=lin_stiff_min, maxval=lin_stiff_max)
    # stiffness_coupling_matrix = jnp.zeros((n_units, n_units))
    start = time.time()
    n_pairs = (n_units * (n_units - 1)) // 2

    # Initialize arrays with random values
    stiffness_array = random.uniform(rng, (n_pairs,), minval=lin_stiff_min, maxval=lin_stiff_max)
    eq_distances_array = random.uniform(rng, (n_pairs,), minval=eq_dist_min, maxval=eq_dist_max)

    # Get indices of the upper triangular part (excluding diagonal)
    upper_indices = jnp.triu_indices(n_units, k=1)

    # Initialize matrices
    stiffness_coupling_matrix = jnp.zeros((n_units, n_units))
    eq_distances_matrix = jnp.zeros((n_units, n_units))

    # Fill in the upper triangular part with values, then reflect to lower triangular
    stiffness_coupling_matrix = stiffness_coupling_matrix.at[upper_indices].set(stiffness_array)
    stiffness_coupling_matrix = stiffness_coupling_matrix + stiffness_coupling_matrix.T

    eq_distances_matrix = eq_distances_matrix.at[upper_indices].set(eq_distances_array)
    eq_distances_matrix = eq_distances_matrix + eq_distances_matrix.T

    # Add an extra dimension for eq_distances_matrix if needed
    eq_distances_matrix = eq_distances_matrix.reshape(n_units, n_units, 1)
    end = time.time()
    print("elapsed time filling matrices", end - start)

    return lin_damping, ang_damping, dist_ang_coupling, stiffness_coupling_matrix, eq_distances_matrix

def initialize_input_map(rng, input_dim, n_units):
    """
    Initializes an input map that transforms inputs of shape (batch_size, input_dim)
    to match the shape (batch_size, n_units).
    """
    # Initialize with random values; feel free to adjust the range or scale based on your needs
    input_map = random.normal(rng, (input_dim, n_units)) * 0.1
    return input_map

# Utilities (unchanged)
def pairwise_differences(arr):
    expanded_arr1 = arr[:, None, :]  # (n_units, 1, 2)
    expanded_arr2 = arr[None, :, :]  # (1, n_units, 2)
    return expanded_arr1 - expanded_arr2

def angle_to_unit_vector(angle):
    cos_angle = jnp.cos(angle)
    sin_angle = jnp.sin(angle)
    return jnp.stack((cos_angle, sin_angle), axis=-1)

# Single-system forward step (no batch dimension)
@jit
def unicycle_network_forward(params, input_map, u_lin, u_ang, x, z, theta, s, omega, dt):
    lin_damping, ang_damping, dist_ang_coupling, stiffness_coupling_matrix, eq_distances_matrix = params

    linear_inp_forces = jnp.dot(u_lin, input_map)      # (n_units,)
    angular_inp_forces = jnp.dot(u_ang, input_map)      # (n_units,)

    coords_2d = jnp.stack((x, z), axis=-1)              # (n_units, 2)
    theta_unit_vectors = angle_to_unit_vector(theta)    # (n_units, 2)
    distance_vectors = pairwise_differences(coords_2d)  # (n_units, n_units, 2)
    distance_magnitudes = jnp.linalg.norm(distance_vectors, axis=-1, keepdims=True)  # (n_units, n_units, 1)
    distance_vectors_normalized = jnp.nan_to_num(distance_vectors / distance_magnitudes)

    forces_before_projection = stiffness_coupling_matrix[:, :, None] * \
        (eq_distances_matrix[:, :, 0] - distance_magnitudes[:, :, 0])[:, :, None] * distance_vectors_normalized
    projected_forces = jnp.einsum('ijk,ik->i', forces_before_projection, theta_unit_vectors)  # (n_units,)

    manual_projection = jnp.sum(forces_before_projection * theta_unit_vectors[:, None, :], axis=(1, 2))
    print("check", jnp.allclose(projected_forces, manual_projection))

    v_dot = linear_inp_forces + projected_forces - (s * lin_damping[:, 0])
    s = s + v_dot * dt

    ang_distances = theta[:, None] - theta[None, :]
    coupling_term_ang = jnp.sum(dist_ang_coupling * (-ang_distances), axis=1)
    omega_dot = (angular_inp_forces + coupling_term_ang - omega * ang_damping[:, 0])
    omega = omega + omega_dot * dt

    theta = theta + dt * omega
    x = x + jnp.cos(theta) * s
    z = z + jnp.sin(theta) * s

    return x, z, theta, s, omega

# Single-system integration over time
@jit
def reservoir_forward(params, input_map, u_lin_seq, u_ang_seq, init_state, dt):
    def step_fn(carry, inputs):
        x, z, theta, s, omega = carry
        u_lin_t, u_ang_t = inputs
        x, z, theta, s, omega = unicycle_network_forward(
            params, input_map, u_lin_t, u_ang_t, x, z, theta, s, omega, dt
        )
        return (x, z, theta, s, omega), jnp.hstack([x, z, theta, s, omega])  # (n_units * 5,)
    
    final_state, state_trajectory = lax.scan(
        step_fn, init_state, (u_lin_seq, u_ang_seq), length=u_lin_seq.shape[0]
    )
    return state_trajectory, final_state

# Assume:
# u_lin_batch, u_ang_batch: (batch_size, time_steps, input_dim)
# init_state_batch: (batch_size, 5, n_units) → tuple of 5 arrays
# params, input_map are shared across the batch
# dt: scalar

def unpack_and_forward(params, input_map, u_lin, u_ang, init_state, dt):
    x, z, theta, s, omega = init_state
    return reservoir_forward(params, input_map, u_lin, u_ang, (x, z, theta, s, omega), dt)

batched_forward = vmap(unpack_and_forward, in_axes=(None, None, 0, 0, 0, None))

rng = random.PRNGKey(0)
batch_size, n_units, n_inp = 100, 200, 1  # Add batch size
params = initialize_params(rng, n_units, n_inp, 0.5, 1.0, 0.1, 0.3, 0.1, 0.2, 0.1, 0.2, 0.5, 1.0)
input_map = initialize_input_map(rng, 1, n_units)
# start = time.time()
init_state_batch = (
    random.uniform(rng, (batch_size, n_units), minval=0.1, maxval=0.6),
    random.uniform(rng, (batch_size, n_units), minval=0.1, maxval=0.2),
    random.uniform(rng, (batch_size, n_units)),
    random.uniform(rng, (batch_size, n_units)),
    random.uniform(rng, (batch_size, n_units))
)
# end = time.time()
# print("time to initialize init states", end - start)
dt = 0.01
steps = 784
u_lin_batch = jnp.zeros((batch_size, steps, 1))
u_ang_batch = jnp.zeros((batch_size, steps, 1))

start = time.time()
# Call:
trajectories, final_states = batched_forward(
    params, input_map,
    u_lin_batch, u_ang_batch,
    init_state_batch, dt
)
end = time.time()
print("elapsed time jax batched 1", end - start)
print("Final States Shape:", final_states[0].shape)  

start = time.time()
# Call:
trajectories, final_states = batched_forward(
    params, input_map,
    u_lin_batch, u_ang_batch,
    init_state_batch, dt
)
end = time.time()
print("elapsed time jax batched 2", end - start)
print("Final States Shape:", final_states[0].shape)  
print("Final States:", final_states)