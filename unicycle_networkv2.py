#%%
import numpy as np
import torch
from torch import nn
import torch.utils.data as data
import torch.nn.functional as F
from scipy.spatial import distance
import matplotlib.pyplot as plt
import pdb
from matplotlib.animation import FuncAnimation
import torchvision
import torchvision.transforms as transforms
#%%
gamma = 2
n_hid = 3
n_inp = 10
dist_eq = 0.5
dt = 0.01
t_steps = 200
#%%
epsilon_min, epsilon_max = 0.1,0.2
linear_stiff_min, linear_stiff_max = 0.1,0.5
ang_stiff_min, ang_stiff_max = 0.4,0.6
lin_damping = torch.rand(n_hid,1, requires_grad=False) * (epsilon_max - epsilon_min) + epsilon_min
ang_damping = torch.rand(n_hid,1, requires_grad=False) * (epsilon_max - epsilon_min) + epsilon_min

dist_coupling = torch.rand(n_hid, n_hid) * (linear_stiff_max - linear_stiff_min) + linear_stiff_min
dist_ang_coupling = torch.rand(n_hid, n_hid) * (ang_stiff_max - ang_stiff_min) + ang_stiff_min

inp2h = torch.rand(n_hid, n_inp)
inp2h = nn.Parameter(inp2h, requires_grad=False)
inp_ang2h = torch.rand(n_hid, n_inp)
inp_ang2h = nn.Parameter(inp2h, requires_grad=False)
bias = (torch.rand(n_hid) * 2 - 1)
bias = nn.Parameter(bias, requires_grad=False)
#%%
def pairwise_differences(arr):
    """
    Computes all pairwise differences between vectors in the given 2D array.
    
    Parameters:
    arr (numpy.ndarray): A 2D array of shape (n, m), where n is the number of vectors
                         and m is the dimension of each vector.
                         
    Returns:
    numpy.ndarray: A 3D array of shape (n, n, m) where the element at [i, j, :] is the 
                   difference between the i-th and j-th vectors.
    """
    n, m = arr.shape
    # Expand dimensions to broadcast the subtraction over the pairs
    expanded_arr1 = np.expand_dims(arr, axis=1)
    expanded_arr2 = np.expand_dims(arr, axis=0)
    # Calculate pairwise differences
    differences = expanded_arr1 - expanded_arr2
    return differences
#%%
def angle_to_unit_vector(angle):
    return np.hstack((np.cos(angle), np.sin(angle)))
#%%
stiffnesses_array = torch.abs(torch.randn(int((n_hid**2-n_hid)/2),1)*0.2)
stiffness_coupling_matrix = np.zeros((n_hid, n_hid))
idx=0
for i in range(n_hid):
    for j in range(i + 1, n_hid):
        stiffness_coupling_matrix[i, j] = stiffnesses_array[idx]
        stiffness_coupling_matrix[j,i] = stiffnesses_array[idx]
        idx += 1
#%%
# # Example usage
# arr = np.array([[1, 2], [3, 2], [-4, 7]])
# thetas = np.array([[np.pi/2], [0.2], [0.1]])
# theta_unit_vectors = angle_to_unit_vector(thetas)
# distance_vectors = pairwise_differences(arr)
# distance_magnitudes = np.linalg.norm(distance_vectors, axis=-1, keepdims=True)
# distance_vectors_normalized = np.nan_to_num(distance_vectors / distance_magnitudes)
# forces1 = (stiffness_coupling_matrix[:,:,np.newaxis] * (distance_magnitudes - 0.5)).squeeze() * np.einsum('ijk,ik->ij', distance_vectors_normalized, theta_unit_vectors)
# forces_before_projection = stiffness_coupling_matrix[:,:,np.newaxis] * (distance_magnitudes - 0.5) * distance_vectors_normalized
# forces2 = np.einsum('ijk,ik->ij', forces_before_projection, theta_unit_vectors)
# print(differences)
#%%
def cell(inp_lin, inp_ang, x, y, theta, v, omega):
    linear_inp_forces = inp2h @ inp_lin
    coords_2d = torch.hstack((x,y))
    theta_unit_vectors = angle_to_unit_vector(theta)
    distance_vectors = pairwise_differences(coords_2d)
    distance_magnitudes = np.linalg.norm(distance_vectors, axis=-1, keepdims=True)
    distance_vectors_normalized = np.nan_to_num(distance_vectors / distance_magnitudes)
    forces_before_projection = stiffness_coupling_matrix[:,:,np.newaxis] * (0.5 - distance_magnitudes) * distance_vectors_normalized
    projected_forces = np.expand_dims(np.einsum('ijk,ik->i', forces_before_projection, theta_unit_vectors), axis=-1)
    v_dot = linear_inp_forces + projected_forces - lin_damping*v
    v = v + v_dot*dt

    inp_term_theta = inp_ang2h @ inp_ang
    ang_distances = theta - theta.T
    coupling_term_ang = torch.sum(dist_ang_coupling * ang_distances, dim=1, keepdim=True)

    omega_dot = ((inp_term_theta + coupling_term_ang) - ang_damping*omega)
    omega = omega + omega_dot*dt

    theta = theta + dt*omega
    x = x + np.cos(theta) * v
    y = y + np.sin(theta) * v

    return x, y, theta, v, omega
# %%
u_t_linear = torch.randn(n_inp, t_steps)*0
u_t_ang = torch.randn(n_inp, t_steps)*0
#%%
x_states = torch.empty(n_hid, 0)
y_states = torch.empty(n_hid, 0)
theta_states = torch.empty(n_hid, 0)
v_states = torch.empty(n_hid, 0)
omega_states = torch.empty(n_hid, 0)

x = torch.randn(n_hid,1)
y = torch.randn(n_hid,1)
theta = torch.randn(n_hid,1) *2
v = torch.randn(n_hid,1) * 0.1
omega = torch.randn(n_hid,1)
#%%
for t in range(t_steps):
    x_states = torch.hstack((x_states, x))
    y_states = torch.hstack((y_states, y))
    theta_states = torch.hstack((theta_states, theta))
    v_states = torch.hstack((v_states, v))
    omega_states = torch.hstack((omega_states, omega))
    x, y, theta, v, omega = cell(u_t_linear[:, t:t+1], u_t_ang[:, t:t+1], x, y, 
                                 theta, v, omega)

# %%
for dim in range(x_states.shape[0]):
    plt.plot(x_states[dim,:], y_states[dim,:])
#%%
# %%
fig, ax = plt.subplots()
lines = [ax.plot([], [], marker='o')[0] for _ in range(x_states.shape[0])]  # Create empty lines for each data set

# Set axis limits
ax.set_xlim(x_states.min()-0.2, x_states.max()+0.2)
ax.set_ylim(y_states.min()-0.2, y_states.max()+0.2)

# Animation function
def update(frame):
    for i in range(x_states.shape[0]):
        lines[i].set_data(x_states[i, :frame], y_states[i, :frame])  # Update data for each line
    return lines

# Create animation
ani = FuncAnimation(fig, update, frames=t_steps, blit=True)
ani.save('animation.mp4')
# %%
fig, ax = plt.subplots()

# Calculate the data ranges
x_range = x_states.max() - x_states.min()
y_range = y_states.max() - y_states.min()

# Set axis limits with a margin
ax.set_xlim(x_states.min() - 0.2, x_states.max() + 0.2)
ax.set_ylim(y_states.min() - 0.2, y_states.max() + 0.2)

# Dynamically set scale and width based on the data range
# These factors can be adjusted based on how the arrows appear
scale_factor = 0.3 * min(x_range, y_range).item()
width_factor = 0.002 * min(x_range, y_range).item()

# Initialize the positions and directions for the arrows
x_pos = x_states[:, 0]
y_pos = y_states[:, 0]
u = np.cos(theta_states[:, 0])  # x components of the arrow directions
v = np.sin(theta_states[:, 0])  # y components of the arrow directions

# Initialize a Quiver object for the arrows
quiver = ax.quiver(x_pos, y_pos, u, v, angles='xy', scale_units='xy', scale=scale_factor, width=width_factor)

quiver_history = []
# Animation function
def update(frame):
    # Get the current positions
    x_pos = x_states[:, frame]
    y_pos = y_states[:, frame]
    
    # Get the current orientations
    u = np.cos(theta_states[:, frame])  # x components of the arrow directions
    v = np.sin(theta_states[:, frame])  # y components of the arrow directions
    
    # Update the quiver with the current positions and orientations
    # quiver.set_offsets(np.c_[x_pos, y_pos])  # Update the arrow positions
    # quiver.set_UVC(u, v)  # Update the arrow directions
    quiver = ax.quiver(x_pos, y_pos, u, v, angles='xy', scale_units='xy', scale=scale_factor, width=width_factor, color=['blue', 'orange', 'green'])

    quiver_history.append(quiver)
    return quiver_history

# Create animation
ani = FuncAnimation(fig, update, frames=t_steps, blit=True)
ani.save('animation_with_arrows.mp4')
# %%

# %%

# %%
plt.plot(u_t_linear.T)
# %%
plt.plot(v_states[:, :].T)
#%%
plt.plot(theta_states.T)
#%%
plt.plot(omega_states.T)
#%%
plt.plot(x_states.T)
# %%
a = torch.randn(1,10)
b = torch.randn(1,10)
coords = torch.hstack((a.T,b.T))
# %%
distance.cdist(coords,coords,'euclidean')
# %%
