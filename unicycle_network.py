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
#%%
gamma = 2
n_hid = 10
n_inp = 10
dist_eq = 0.5
dt = 0.01
t_steps = 1000
#%%
epsilon_min, epsilon_max = 1,2
linear_stiff_min, linear_stiff_max = 0.1,0.5
ang_stiff_min, ang_stiff_max = 10,11
lin_damping = torch.rand(n_hid, requires_grad=False) * (epsilon_max - epsilon_min) + epsilon_min
ang_damping = torch.rand(n_hid, requires_grad=False) * (epsilon_max - epsilon_min) + epsilon_min

dist_coupling = torch.rand(n_hid, n_hid) * (linear_stiff_max - linear_stiff_min) + linear_stiff_min
dist_ang_coupling = torch.rand(n_hid, n_hid) * (ang_stiff_max - ang_stiff_min) + ang_stiff_min

inp2h = torch.rand(n_inp, n_hid)
inp2h = nn.Parameter(inp2h, requires_grad=False)
inp_ang2h = torch.rand(n_inp, n_hid)
inp_ang2h = nn.Parameter(inp2h, requires_grad=False)
bias = (torch.rand(n_hid) * 2 - 1)
bias = nn.Parameter(bias, requires_grad=False)
#%%
inp = torch.randn((10, 1, n_inp))
#%%
res = torch.matmul(inp, inp2h)
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
# Example usage
arr = np.array([[1, 2], [3, 2], [-4, 7]])
thetas = np.array([[np.pi/2], [0.2], [0.1]])
theta_unit_vectors = angle_to_unit_vector(thetas)
distance_vectors = pairwise_differences(arr)
distance_magnitudes = np.linalg.norm(distance_vectors, axis=-1, keepdims=True)
distance_vectors_normalized = np.nan_to_num(distance_vectors / distance_magnitudes)
forces1 = (stiffness_coupling_matrix[:,:,np.newaxis] * (distance_magnitudes - 0.5)).squeeze() * np.einsum('ijk,ik->ij', distance_vectors_normalized, theta_unit_vectors)
forces_before_projection = stiffness_coupling_matrix[:,:,np.newaxis] * (distance_magnitudes - 0.5) * distance_vectors_normalized
forces2 = np.einsum('ijk,ik->ij', forces_before_projection, theta_unit_vectors)
# print(differences)
#%%
def cell(inp_lin, inp_ang, x, y, theta, v, omega):
    inp_term = torch.matmul(inp_lin, inp2h)
    coords_2d = torch.hstack((x.T,y.T))
    distances = distance.cdist(coords_2d, coords_2d, 'euclidean')
    coupling_term_xy = torch.sum((dist_coupling * (dist_eq - distances).T), dim=1)

    v_dot = (torch.tanh(inp_term+ coupling_term_xy * torch.sign(v)) - lin_damping*v)
    # print(v_dot)
    v = v + v_dot*dt

    inp_term_theta = torch.matmul(inp_ang, inp_ang2h)
    ang_distances = theta - theta.T
    coupling_term_ang = torch.sum(dist_ang_coupling * ang_distances, dim=1)

    omega_dot = (torch.tanh(inp_term_theta + coupling_term_ang) - ang_damping*omega)
    omega = omega + omega_dot*dt

    theta = theta + dt*omega
    x = x + np.cos(theta) * v
    y = y + np.sin(theta) * v

    return x, y, theta, v, omega
#%%
n_hid=3
stiffnesses_array = np.random.randint(low=0, high=4, size=int((n_hid**2-n_hid)/2))
stiffness_coupling_matrix = np.zeros((n_hid, n_hid))
idx=0
for i in range(n_hid):
    for j in range(i + 1, n_hid):
        stiffness_coupling_matrix[i, j] = stiffnesses_array[idx]
        stiffness_coupling_matrix[j,i] = stiffnesses_array[idx]
        idx += 1
# %%
u_t_linear = torch.randn(n_inp, t_steps)*0
u_t_ang = torch.randn(n_inp, t_steps)*0
#%%
x_states = torch.empty(0, n_hid)
y_states = torch.empty(0, n_hid)
theta_states = torch.empty(0, n_hid)
v_states = torch.empty(0, n_hid)
omega_states = torch.empty(0, n_hid)

x = torch.randn(1, n_hid)
y = torch.randn(1, n_hid)
theta = torch.randn(1, n_hid) *2
v = torch.randn(1, n_hid) * 0.1
omega = torch.zeros(1, n_hid)
#%%
for t in range(t_steps):
    x_states = torch.vstack((x_states, x))
    y_states = torch.vstack((y_states, y))
    theta_states = torch.vstack((theta_states, theta))
    v_states = torch.vstack((v_states, v))
    omega_states = torch.vstack((omega_states, omega))
    x, y, theta, v, omega = cell(u_t_linear[:, t], u_t_ang[:, t], x, y, 
                                 theta, v, omega)

# %%
for dim in range(x_states.shape[1]):
    plt.plot(x_states[:,dim], y_states[:,dim])
#%%
fig, ax = plt.subplots()
lines = [ax.plot([], [])[0] for _ in range(x_states.shape[1])]  # Create empty lines for each data set

# Set axis limits
ax.set_xlim(min(x_states[-1,:].numpy()), max((x_states[-1,:].numpy())))
ax.set_ylim(min(y_states[-1,:].numpy()), max((y_states[-1,:].numpy())))

# Animation function
def update(frame):
    for i in range(x_states.shape[1]):
        lines[i].set_data(x_states[:frame, i], y_states[:frame,i])  # Update data for each line
    return lines

# Create animation
ani = FuncAnimation(fig, update, frames=t_steps, blit=True)
ani.save('animation.mp4')
# %%
plt.plot(u_t_linear.T)
# %%
plt.plot(v_states[:200, :])
#%%
plt.plot(theta_states)
#%%
plt.plot(omega_states)
#%%
plt.plot(x_states)
# %%
a = torch.randn(1,10)
b = torch.randn(1,10)
coords = torch.hstack((a.T,b.T))
# %%
distance.cdist(coords,coords,'euclidean')
# %%
