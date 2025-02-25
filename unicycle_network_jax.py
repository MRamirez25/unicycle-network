import jax
import jax.numpy as jnp
from jax import random, jit, vmap
import time, timeit
from unicycle_network_class import UnicycleNetwork
import torch
import numpy as np
# jax.config.update('jax_platform_name', 'cpu')
# print("devices", jax.devices())
# Function to initialize random parameters
# jax.default_device = jax.devices("cpu")[0]
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

# Function to compute pairwise differences with batch support
def pairwise_differences(arr):
    expanded_arr1 = arr[:, :, None, :]  # (batch, n_units, 1, 2)
    expanded_arr2 = arr[:, None, :, :]  # (batch, 1, n_units, 2)
    return expanded_arr1 - expanded_arr2

# Convert angles to unit vectors with batch support
def angle_to_unit_vector(angle):
    cos_angle = jnp.cos(angle)
    sin_angle = jnp.sin(angle)
    return jnp.stack((cos_angle, sin_angle), axis=-1)

# Define forward function of UnicycleNetwork with batch support
def unicycle_network_forward(params, input_map, u_lin, u_ang, x, z, theta, s, omega, dt):

    lin_damping, ang_damping, dist_ang_coupling, stiffness_coupling_matrix, eq_distances_matrix = params

    # Apply input map to transform u_lin and u_ang to match the shape (batch_size, n_units)
    linear_inp_forces = jnp.dot(u_lin, input_map)  # shape should be (batch_size, n_units)
    angular_inp_forces = jnp.dot(u_ang, input_map)  # shape should be (batch_size, n_units)

    coords_2d = jnp.stack((x, z), axis=-1)
    theta_unit_vectors = angle_to_unit_vector(theta)
    distance_vectors = pairwise_differences(coords_2d)
    # breakpoint()
    distance_magnitudes = jnp.linalg.norm(distance_vectors, axis=-1, keepdims=True)
    distance_vectors_normalized = jnp.nan_to_num(distance_vectors / distance_magnitudes)
    forces_before_projection = stiffness_coupling_matrix[None, :, :, None] * \
        (eq_distances_matrix - distance_magnitudes) * distance_vectors_normalized
    # print(forces_before_projection)
    projected_forces = jnp.einsum('bijk,bik->bi', forces_before_projection, theta_unit_vectors)
    # print(projected_forces)
    # Update velocities and angular velocities
    v_dot = linear_inp_forces + projected_forces - (s @ lin_damping)
    s = s + v_dot * dt

    inp_term_theta = angular_inp_forces
    theta_expanded_1 = theta[:, :, None]
    theta_expanded_2 = theta[:, None, :]
    ang_distances = theta_expanded_1 - theta_expanded_2
    coupling_term_ang = jnp.sum(dist_ang_coupling[None, :, :] * (-ang_distances), axis=2)

    omega_dot = ((inp_term_theta + coupling_term_ang) - omega @ ang_damping)
    omega = omega + omega_dot * dt

    theta = theta + dt * omega
    x = x + jnp.cos(theta) * s
    z = z + jnp.sin(theta) * s

    return x, z, theta, s, omega

# Forward pass of UnicycleReservoir with batch support
@jit
def reservoir_forward(params, input_map, u_lin, u_ang, init_state, dt):
    x, z, theta, s, omega = init_state
    # u_lin = u_lin.reshape((steps, -1, x.shape[-1]))
    # u_ang = u_ang.reshape((steps, -1, x.shape[-1]))
    @jit
    def step_fn(carry, inputs):
        x, z, theta, s, omega = carry
        u_lin_t, u_ang_t = inputs
        x, z, theta, s, omega = unicycle_network_forward(params, input_map, u_lin_t, u_ang_t, x, z, theta, s, omega, dt)
        return (x, z, theta, s, omega), jnp.hstack([x, z, theta, s, omega])

    final_state, state_trajectory = jax.lax.scan(step_fn, (x, z, theta, s, omega), (u_lin, u_ang), length=784)
    return state_trajectory, final_state

# Initialization with batch size
rng = random.PRNGKey(0)
batch_size, n_units, n_inp = 100, 200, 1  # Add batch size
params = initialize_params(rng, n_units, n_inp, 0.5, 1.0, 0.1, 0.3, 0.1, 0.2, 0.1, 0.2, 0.5, 1.0)
input_map = initialize_input_map(rng, 1, n_units)
start = time.time()
init_state = (
    random.uniform(rng, (batch_size, n_units), minval=0.1, maxval=0.6),
    random.uniform(rng, (batch_size, n_units), minval=0.1, maxval=0.2),
    random.uniform(rng, (batch_size, n_units)),
    random.uniform(rng, (batch_size, n_units)),
    random.uniform(rng, (batch_size, n_units))
)
end = time.time()
print("time to initialize init states", end - start)
dt = 0.01
steps = 784
u_lin = jnp.zeros((steps, batch_size, 1))
u_ang = jnp.zeros((steps, batch_size, 1))
print("device", u_lin.device())
# Run forward pass with batches
print("start")
start = time.time()
state_trajectory, final_state = reservoir_forward(params, input_map, u_lin, u_ang, init_state, dt)
end = time.time()
print("elapsed time 1", end - start)
# print("Final State:", final_state)

x, z, theta, s, omega = init_state
start = time.time()
for t in range(steps):
    x, z, theta, s, omega = unicycle_network_forward(params, input_map, u_lin[t], u_ang[t], x, z, theta, s, omega, dt)
end = time.time()
print("elapsed time 2", end - start)
print(x.shape)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

unicycle_network_torch = UnicycleNetwork(1, 200, dt=0.01)
unicycle_network_torch.to(device)
x = torch.from_numpy(np.asarray(init_state[0].copy())).to(device)
z = torch.from_numpy(np.asarray(init_state[1].copy())).to(device)
theta = torch.from_numpy(np.asarray(init_state[2].copy())).to(device)
s = torch.from_numpy(np.asarray(init_state[3].copy())).to(device)
omega = torch.from_numpy(np.asarray(init_state[4].copy())).to(device)
u_lin = torch.from_numpy(np.asarray(u_lin).copy()).to(device)
unicycle_network_torch.lin_damping = unicycle_network_torch.lin_damping.to(device)
unicycle_network_torch.ang_damping = unicycle_network_torch.ang_damping.to(device)
start = time.time()
for t in range(steps):
    x, z, theta, s, omega = unicycle_network_torch.forward(u_lin[t], u_lin[t], x, z, theta, s, omega)
end = time.time()
print("elapsed time 3", end - start)
print(x.shape)

# # Assuming `reservoir_forward` is defined and you have parameters and initial state
# def loss_fn(params, u_lin, u_ang, init_state, dt, steps):
#     state_trajectory, final_state = reservoir_forward(params, input_map, u_lin, u_ang, init_state, dt, steps)
#     # Define your loss based on the final state or trajectory
#     return jnp.sum(final_state[0])  # Example loss

# Use jax.grad to compute gradients
# gradient_fn = jax.grad(loss_fn)
# gradients = gradient_fn(params, u_lin, u_ang, init_state, dt, steps)