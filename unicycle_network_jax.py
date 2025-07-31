import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax
import jax.numpy as jnp
from jax import random, jit, vmap
import time, timeit
from unicycle_network_class import UnicycleNetwork, UnicycleReservoir
from unicycle_network_vmap import batched_forward
import torch
import numpy as np
import optuna
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
@jit
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
# start = time.time()
init_state = (
    random.uniform(rng, (batch_size, n_units), minval=0.1, maxval=0.6),
    random.uniform(rng, (batch_size, n_units), minval=0.1, maxval=0.2),
    random.uniform(rng, (batch_size, n_units)),
    random.uniform(rng, (batch_size, n_units)),
    random.uniform(rng, (batch_size, n_units))
)
print("init",init_state)
# end = time.time()
# print("time to initialize init states", end - start)
dt = 0.01
steps = 784
u_lin = jnp.zeros((steps, batch_size, 1))
u_ang = jnp.zeros((steps, batch_size, 1))
print("device", u_lin.device)
# Run forward pass with batches
print("start")
start = time.time()
state_trajectory, final_state = reservoir_forward(params, input_map, u_lin, u_ang, init_state, dt)
end = time.time()
print("elapsed time jax 1", end - start)
print("Final State:", final_state[0].shape)

# print("Final State:", final_state)
print("start")
start = time.time()
state_trajectory, final_state = reservoir_forward(params, input_map, u_lin, u_ang, init_state, dt)
end = time.time()
print("elapsed time jax 2", end - start)
print("Final State:", final_state[0].shape)

print("start")
start = time.time()
state_trajectory, final_state = reservoir_forward(params, input_map, u_lin, u_ang, init_state, dt)
end = time.time()
print("elapsed time jax 3", end - start)
print("Final State:", final_state[0].shape)

print(final_state)

print("init", init_state)

start = time.time()   

x_cuda = jax.device_put(final_state[0], device=jax.devices("gpu")[0])
z_cuda = jax.device_put(final_state[1], device=jax.devices("gpu")[0])
theta_cuda = jax.device_put(final_state[2], device=jax.devices("gpu")[0])
s_cuda = jax.device_put(final_state[3], device=jax.devices("gpu")[0])
omega_cuda = jax.device_put(final_state[4], device=jax.devices("gpu")[0])

end = time.time()
print("elapsed time moving to gpu", end - start)

# dt = 0.01
# steps = 784
# u_lin_batch = jnp.zeros((batch_size, steps, 1))
# u_ang_batch = jnp.zeros((batch_size, steps, 1))
# init_state_batch = init_state
# start = time.time()
# # Call:
# trajectories, final_states = batched_forward(
#     params, input_map,
#     u_lin_batch, u_ang_batch,
#     init_state_batch, dt
# )
# end = time.time()
# print("elapsed time jax batched 1", end - start)
# print("Final States Shape:", final_states[0].shape)  

# start = time.time()
# # Call:
# trajectories, final_states = batched_forward(
#     params, input_map,
#     u_lin_batch, u_ang_batch,
#     init_state_batch, dt
# )
# end = time.time()
# print("elapsed time jax batched 2", end - start)
# print("Final States Shape:", final_states[0].shape)  
# print("Final States:", final_states)
# x, z, theta, s, omega = init_state
# start = time.time()
# for t in range(steps):
#     x, z, theta, s, omega = unicycle_network_forward(params, input_map, u_lin[t], u_ang[t], x, z, theta, s, omega, dt)
# end = time.time()
# print("elapsed time jax 2", end - start)
# print(x.shape)

torch.cuda.empty_cache()
torch.cuda.ipc_collect()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch device", device)
# Convert JAX parameters to PyTorch tensors

storage_name = f"unicycle_nets_mnist_all_digits"
study_name = "not_aligned_w_ang_input_w_ang_connections"
storage_name = f"sqlite:///optuna_databases/{storage_name}.db"
study = optuna.create_study(storage=storage_name, study_name=study_name, direction='maximize', load_if_exists="True")
params = study.best_params
#%%
aligned_orientations = False  # Set to True if you want aligned orientations, False otherwise
ang_input = True  # Set to True if you want angular input, False otherwise
#%%
n_units = n_units
lr = params['lr']
lin_stiff_min =  params['lin_stiff_min']
lin_stiff_max =  params['lin_stiff_max']
ang_stiff_min =  params['ang_stiff_min']
ang_stiff_max =  params['ang_stiff_max']
lin_damping_min =  params['lin_damping_min']
lin_damping_max =  params['lin_damping_max']
ang_damping_min =  params['ang_damping_min']
ang_damping_max  = params['ang_damping_max']
bs_train = batch_size
bs_test = bs_train
dt = params['dt']
inp_bias =  params['inp_bias']
anchor_con_fraction = params['anchor_con_fraction']
num_non_zero = params['non_zero_elements']
magnitude_min = params['magnitude_min']
magnitude_max = params['magnitude_max']
non_zero_elements_ang = params['non_zero_elements_ang']
magnitude_min_ang = -params['magnitude_max_ang']
magnitude_max_ang = params['magnitude_max_ang']
n_connections_fraction = params['n_connections_fraction']
n_connections = int(n_units*n_connections_fraction)
washup = params['washup_steps']
n_steps_readout = params['steps_readout']
anchor_con_fraction_ang = params['anchor_con_fraction_ang']
eq_dist_min = params['eq_dist_min']
eq_dist_max = params['eq_dist_max']
eq_dist_min_ang = params['eq_dist_min_ang']
eq_dist_max_ang = params['eq_dist_max_ang']
n_epochs = params['n_epochs']
n_connections_anchor = int(n_units * anchor_con_fraction)
n_connections_ang_fraction = params["n_connections_ang_fraction"]
n_connections_ang = int(n_connections_ang_fraction*n_units)
n_connections_anchor_ang = int(anchor_con_fraction_ang * n_units)
#%%
lin_input_map = torch.zeros(1, n_units)
num_non_zero = num_non_zero
non_zero_indices = torch.randperm(n_units)[:num_non_zero]  # Randomly select indices
non_zero_values_min = magnitude_min
non_zero_values_max = magnitude_max
lin_input_map[0, non_zero_indices] = torch.rand(num_non_zero) * (non_zero_values_max- non_zero_values_min) + non_zero_values_min  # Random magnitudes
#%%
# # Randomized ang_input_map
ang_input_map = torch.zeros(1, n_units)
if ang_input:
    num_non_zero_ang = non_zero_elements_ang
    non_zero_indices = torch.randperm(n_units)[:num_non_zero_ang]  # Randomly select indices
    non_zero_values_ang = magnitude_min_ang
    magnitude_max_ang = magnitude_max_ang
    ang_input_map[0, non_zero_indices] = torch.rand(num_non_zero_ang) * (magnitude_max_ang- non_zero_values_ang) + non_zero_values_ang  # Random magnitudes 
#%%
model = UnicycleReservoir(n_inp=1, n_units=n_units, dt=dt, n_out=10, lin_input_map=lin_input_map, 
                          lin_stiff_min=lin_stiff_min, lin_damping_min=lin_damping_min, lin_damping_max=lin_damping_max, lin_stiff_max=lin_stiff_max,
                          eq_dist_min=eq_dist_min, eq_dist_max=eq_dist_max, eq_dist_min_ang=eq_dist_min_ang,
                          eq_dist_max_ang=eq_dist_max_ang,  
                          n_connections=n_connections, n_connections_anchor=n_connections_anchor, 
                          n_past_steps_readout=n_steps_readout, n_connections_ang=n_connections_ang, n_connections_anchor_ang=n_connections_anchor_ang,
                          ang_stiff_min=ang_stiff_min, ang_stiff_max=ang_stiff_max, ang_damping_min=ang_damping_min, ang_damping_max=ang_damping_max,
                          inp_bias=inp_bias, ang_input_map=ang_input_map).to(device)



x = torch.from_numpy(np.asarray(init_state[0].copy())).to(device)
z = torch.from_numpy(np.asarray(init_state[1].copy())).to(device)
theta = torch.from_numpy(np.asarray(init_state[2].copy())).to(device)
s = torch.from_numpy(np.asarray(init_state[3].copy())).to(device)
omega = torch.from_numpy(np.asarray(init_state[4].copy())).to(device)
u_lin = torch.from_numpy(np.asarray(u_lin).copy()).to(device)

u_lin = u_lin.permute((1, 0, 2))  # Reshape to (batch_size, steps, n_inp)
print("u_lin shape", u_lin.shape)
# Move initial states to GPU if available
model.x_init = x
model.z_init = z
model.theta_init = theta
model.s_init = s
model.omega_init = omega
model.lin_input_map = model.lin_input_map.to(device)
model.ang_input_map = model.ang_input_map.to(device)
model.unicycle_network.lin_damping = model.unicycle_network.lin_damping.to(device)
model.unicycle_network.ang_damping = model.unicycle_network.ang_damping.to(device)
model.unicycle_network.mass_vector = model.unicycle_network.mass_vector.to(device)
model.unicycle_network.j_vector = model.unicycle_network.j_vector.to(device)

start = time.time()
states_list_torch, output_torch = model(u_lin, u_lin)
end = time.time()
print("elapsed time torch", end - start)
# # breakpoint()
# # print(states_list_torch[-1][:,:100].detach().cpu().numpy() - np.array(final_state[0]))

# # # Assuming `reservoir_forward` is defined and you have parameters and initial state
# # def loss_fn(params, u_lin, u_ang, init_state, dt, steps):
# #     state_trajectory, final_state = reservoir_forward(params, input_map, u_lin, u_ang, init_state, dt, steps)
# #     # Define your loss based on the final state or trajectory
# #     return jnp.sum(final_state[0])  # Example loss

# # Use jax.grad to compute gradients
# # gradient_fn = jax.grad(loss_fn)
# # gradients = gradient_fn(params, u_lin, u_ang, init_state, dt, steps)