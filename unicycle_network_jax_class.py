import jax
import jax.numpy as jnp
from jax import random, jit
import time
from torch import nn
import torch

class UnicycleModel:
    def __init__(self, rng, n_units, n_inp, lin_stiff_min=0.1, lin_stiff_max=0.5, ang_stiff_min=0.1, ang_stiff_max=0.5,
                 lin_damping_min=0.1, lin_damping_max=0.2, ang_damping_min=0.1, ang_damping_max=0.2, eq_dist_min=0.5, eq_dist_max=1.0, batch_size=100, 
                 sparsity_level=0.5, mlp_input_map=False):
        self.rng = rng
        self.n_units = n_units
        self.n_inp = n_inp
        self.mlp_input_map = mlp_input_map

        # Initialize parameters
        self.lin_damping, self.ang_damping, self.dist_ang_coupling, self.stiffness_coupling_matrix, self.eq_distances_matrix = self.initialize_params(
            lin_stiff_min, lin_stiff_max, ang_stiff_min, ang_stiff_max, lin_damping_min, lin_damping_max, ang_damping_min, ang_damping_max, eq_dist_min, eq_dist_max, 
            sparsity_level)

        # Initialize the input map
        self.init_state = (
    random.normal(rng, (batch_size, n_units)),
    random.normal(rng, (batch_size, n_units)),
    random.normal(rng, (batch_size, n_units)),
    random.normal(rng, (batch_size, n_units)),
    random.normal(rng, (batch_size, n_units))
)

    def initialize_params(self, lin_stiff_min, lin_stiff_max, ang_stiff_min, ang_stiff_max,
                          lin_damping_min, lin_damping_max, ang_damping_min, ang_damping_max, eq_dist_min, eq_dist_max, 
                          sparsity_level):
        lin_damping = random.uniform(self.rng, (1,self.n_units), minval=lin_damping_min, maxval=lin_damping_max)
        ang_damping = random.uniform(self.rng, (1,self.n_units), minval=ang_damping_min, maxval=ang_damping_max)
        dist_ang_coupling = random.uniform(self.rng, (self.n_units, self.n_units), minval=ang_stiff_min, maxval=ang_stiff_max)

        n_pairs = (self.n_units * (self.n_units - 1)) // 2
        n_units = self.n_units
        matrix = jnp.zeros((n_units, n_units))
        rng=self.rng
        stiffness_coupling_matrix = random.uniform(rng, (n_units, n_units), minval=lin_stiff_min, maxval=lin_stiff_max)
        eq_distances_matrix = random.uniform(rng, (n_units, n_units), minval=eq_dist_min, maxval=eq_dist_max)

        # Step 2: Create a mask with the desired sparsity level for the upper triangle (excluding the diagonal)
        upper_triangular_mask = random.bernoulli(rng, p=1 - sparsity_level, shape=(n_units, n_units))
        upper_triangular_mask = jnp.triu(upper_triangular_mask, k=1)  # Exclude diagonal for symmetry

        # Step 3: Apply the mask to create the sparse upper triangular matrix
        stiffness_coupling_matrix = stiffness_coupling_matrix * upper_triangular_mask
        eq_distances_matrix = eq_distances_matrix * upper_triangular_mask

        # Step 4: Make the matrix symmetric by adding the transpose
        stiffness_coupling_matrix = stiffness_coupling_matrix + stiffness_coupling_matrix.T
        eq_distances_matrix = eq_distances_matrix + eq_distances_matrix.T

        # Add an extra dimension for equilibrium distances
        eq_distances_matrix = eq_distances_matrix.reshape(self.n_units, self.n_units, 1)

        return lin_damping, ang_damping, dist_ang_coupling, stiffness_coupling_matrix, eq_distances_matrix

    def initialize_input_map(self, input_dim, n_units, num_non_zero, non_zero_min_magnitude):
        """
        Initializes an input map that transforms inputs of shape (batch_size, input_dim)
        to match the shape (batch_size, n_units).
        """
        # Step 1: Initialize a zero array
        lin_input_map = jnp.zeros((1, n_units))

        # Step 2: Randomly select the number of non-zero elements
        num_non_zero = num_non_zero

        # Step 3: Generate random indices for non-zero elements
        indices = random.permutation(self.rng, jnp.arange(n_units))[:num_non_zero]

        # Step 4: Get the range for the non-zero values
        non_zero_values = non_zero_min_magnitude

        # Step 5: Generate random values in the specified range and set them in `lin_input_map`
        rand_values = random.uniform(self.rng, (num_non_zero,), minval=non_zero_values, maxval=1.0)
        self.input_map = lin_input_map.at[0, indices].set(rand_values)
        # return lin_input_map

        # lin_input_map = torch.ones((1, n_units))
        # self.input_map = lin_input_map
        # self.input_map = torch.nn.Parameter(self.input_map)

    @staticmethod
    def pairwise_differences(arr):
        expanded_arr1 = arr[:, :, None, :]  # (batch, n_units, 1, 2)
        expanded_arr2 = arr[:, None, :, :]  # (batch, 1, n_units, 2)
        return expanded_arr1 - expanded_arr2

    @staticmethod
    def angle_to_unit_vector(angle):
        cos_angle = jnp.cos(angle)
        sin_angle = jnp.sin(angle)
        return jnp.stack((cos_angle, sin_angle), axis=-1)

    def unicycle_network_forward(self, u_lin, u_ang, x, z, theta, s, omega, dt, mlp_input_map=False):
        lin_damping = self.lin_damping
        ang_damping = self.ang_damping
        dist_ang_coupling = self.dist_ang_coupling
        stiffness_coupling_matrix = self.stiffness_coupling_matrix
        eq_distances_matrix = self.eq_distances_matrix

        # Apply input map
        # if not self.mlp_input_map:
        linear_inp_forces = jnp.dot(u_lin, self.input_map)  # shape: (batch_size, n_units)
        # else:
            # linear_inp_forces = u_lin
        ang_input_map = jnp.zeros_like(self.input_map)
        angular_inp_forces = jnp.dot(u_ang, ang_input_map)  # shape: (batch_size, n_units)

        coords_2d = jnp.stack((x, z), axis=-1)
        theta_unit_vectors = self.angle_to_unit_vector(theta)
        distance_vectors = self.pairwise_differences(coords_2d)

        distance_magnitudes = jnp.linalg.norm(distance_vectors, axis=-1, keepdims=True)
        distance_vectors_normalized = jnp.nan_to_num(distance_vectors / distance_magnitudes)
        forces_before_projection = stiffness_coupling_matrix[None, :, :, None] * \
            (eq_distances_matrix - distance_magnitudes) * distance_vectors_normalized

        projected_forces = jnp.einsum('bijk,bik->bi', forces_before_projection, theta_unit_vectors)

        # Update velocities and angular velocities
        v_dot = linear_inp_forces + projected_forces - (s * lin_damping)
        s = s + v_dot * dt

        inp_term_theta = angular_inp_forces
        theta_expanded_1 = theta[:, :, None]
        theta_expanded_2 = theta[:, None, :]
        ang_distances = theta_expanded_1 - theta_expanded_2
        coupling_term_ang = jnp.sum(dist_ang_coupling[None, :, :] * (-ang_distances), axis=2)

        omega_dot = ((inp_term_theta + coupling_term_ang) - omega * ang_damping)
        omega = omega + omega_dot * dt

        theta = theta + dt * omega
        x = x + jnp.cos(theta) * s
        z = z + jnp.sin(theta) * s

        return x, z, theta, s, omega
    
    def reservoir_forward(self, u_lin, u_ang, init_state, dt, steps):
        x, z, theta, s, omega = self.init_state

        def step_fn(carry, inputs):
            x, z, theta, s, omega = carry
            u_lin_t, u_ang_t = inputs
            x, z, theta, s, omega = self.unicycle_network_forward(u_lin_t, u_ang_t, x, z, theta, s, omega, dt)
            return (x, z, theta, s, omega), jnp.hstack([x, z, theta, s, omega])

        final_state, state_trajectory = jax.lax.scan(step_fn, (x, z, theta, s, omega), (u_lin, u_ang), length=steps)
        return state_trajectory, final_state

class ReadoutLayer(nn.Module):
    def __init__(self, input_size, output_size):
        """
        Initialize the ReadoutLayer.
        Parameters:
        - input_size (int): The size of the input features.
        - output_size (int): The size of the output features.
        """
        super(ReadoutLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        """
        Forward pass through the readout layer.
        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        
        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        return self.linear(x)
    
class SmallMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(SmallMLP, self).__init__()
        layers = []

        # Input layer to first hidden layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU())

        # Last hidden layer to output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        layers.append(nn.Softmax(dim=1))

        # Combine layers into a sequential module
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)