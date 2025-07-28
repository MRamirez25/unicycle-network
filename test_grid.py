#%%
from utils import get_mnist_data
from tqdm import tqdm
import matplotlib.pyplot as plt
from unicycle_network_class import UnicycleNetwork, UnicycleReservoir
from torch import nn, optim
import torch
import optuna
from matplotlib.animation import FuncAnimation
import numpy as np
# %%
study_name = f"unicycle_opt_all_classes_w_ang_input"
storage_name = "sqlite:///optuna_databases/{}.db".format(study_name)
study = optuna.create_study(storage=storage_name, study_name=study_name, direction='maximize', load_if_exists="True")
params = study.best_params
#%%
n_units = params['n_units']
lr = params['lr']
lin_stiff_min =  params['lin_stiff_min']
lin_stiff_max =  params['lin_stiff_max']
ang_stiff_min =  0.1#params['ang_stiff_min']
ang_stiff_max =  0.2#params['ang_stiff_max']
lin_damping_min =  params['lin_damping_min']*1.5
lin_dmping_max =  params['lin_damping_max']*1.5
ang_damping_min =  params['ang_damping_min']
ang_damping_max  = params['ang_damping_max']
bs_train = params['batch_size']
bs_test = bs_train
dt = params['dt']
inp_bias =  params['inp_bias']
anchor_con_fraction = params['anchor_con_fraction']
num_non_zero = params['non_zero_elements']
magnitude_min = params['magnitude_min']*0.1
magnitude_max = params['magnitude_max']*0.1
non_zero_elements_ang = params['non_zero_elements_ang']
magnitude_min_ang = params['magnitude_min_ang']*0.5
magnitude_max_ang = params['magnitude_max_ang']
n_connections = 3#params['n_connections']
washup = 0#params['washup_steps']
n_steps_readout = 0#params['steps_readout']
n_connections_ang = 0#params['n_connections_ang']
anchor_con_fraction_ang = params['anchor_con_fraction_ang']
eq_dist_min = params['eq_dist_min']
eq_dist_max = params['eq_dist_max']
eq_dist_min_ang = params['eq_dist_min_ang']
eq_dist_max_ang = params['eq_dist_max_ang']
n_epochs = 3#params['n_epochs']
n_connections_anchor = int(n_connections * anchor_con_fraction)
n_connections_anchor_ang = int(n_connections_ang * anchor_con_fraction_ang)

#%%
train_loader, valid_loader, test_loader = get_mnist_data(bs_train=bs_train, bs_test=bs_test, classes=[0,1,2,3,4,5,6,7,8,9], new_fraction=0.5, test_fraction=0.5)
#%%
lin_input_map = torch.zeros(1, n_units)
num_non_zero = num_non_zero
non_zero_indices = torch.randperm(n_units)[:num_non_zero]  # Randomly select indices
non_zero_values_min = magnitude_min
non_zero_values_max = magnitude_max
lin_input_map[0, non_zero_indices] = torch.rand(num_non_zero) * (non_zero_values_max- non_zero_values_min) + non_zero_values_min  # Random magnitudes#%%
#%%
# Randomized ang_input_map
ang_input_map = torch.zeros(1, n_units)
non_zero_indices = torch.randperm(n_units)[:non_zero_elements_ang]  # Randomly select indices
non_zero_values_ang = magnitude_min_ang
magnitude_max_ang = magnitude_max_ang
ang_input_map[0, non_zero_indices] = torch.rand(non_zero_elements_ang) * (magnitude_max_ang- non_zero_values_ang) + non_zero_values_ang  
#%%
model = UnicycleReservoir(n_inp=1, n_units=n_units, dt=dt, n_out=10, lin_input_map=lin_input_map, 
                          lin_stiff_min=lin_stiff_min, lin_damping_min=lin_damping_min, lin_damping_max=lin_dmping_max, lin_stiff_max=lin_stiff_max,
                          eq_dist_min=eq_dist_min, eq_dist_max=eq_dist_max, eq_dist_min_ang=eq_dist_min_ang,
                          eq_dist_max_ang=eq_dist_max_ang,  
                          n_connections=n_connections, n_connections_anchor=n_connections_anchor, 
                          n_past_steps_readout=n_steps_readout, n_connections_ang=n_connections_ang, n_connections_anchor_ang=n_connections_anchor_ang,
                          ang_stiff_min=ang_stiff_min, ang_stiff_max=ang_stiff_max, ang_damping_min=ang_damping_min, ang_damping_max=ang_damping_max,
                          inp_bias=inp_bias, ang_input_map=ang_input_map)
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
#%%
def test(data_loader):
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            # images, labels = images.to(device), labels.to(device)
            images = images.reshape(bs_test, 1, 784)
            images = images.permute(0, 2, 1)
            angular_input = torch.zeros_like(images)

            images = images.to(device)
            # images = images[:, perm, :]
            angular_input = angular_input.to(device)
            labels = labels.to(device)

            _, output = model(angular_input, images)
            test_loss += objective(output, labels).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum()
    test_loss /= i+1
    accuracy = 100. * correct / len(data_loader.dataset)

    return accuracy.item()
#%%
for i, (images, labels) in enumerate(test_loader):
    images = images.reshape(bs_test, 1, 784)
#%%
n_epochs = n_epochs
objective = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
#%%
theta_init_test = torch.randn(n_units).repeat(bs_train,1)

#%%
# # %%
# for parameter in model.parameters():
#     print(parameter)
# model.set_init_states_random(bs_train)
num_rows = 5
num_cols = 4
model.set_init_states_grid(bs_train, num_rows=num_rows, num_cols=num_cols, spacing=(0.75, 0.75))
model.x_init = model.x_init.to(device)
model.z_init = model.z_init.to(device)
model.theta_init = theta_init_test.to(device)
model.s_init = model.s_init.to(device)
model.omega_init = torch.zeros_like(model.omega_init).to(device)
model.lin_input_map = model.lin_input_map.to(device)
model.ang_input_map = model.ang_input_map.to(device)
model.unicycle_network.lin_damping = model.unicycle_network.lin_damping.to(device)
model.unicycle_network.ang_damping = model.unicycle_network.ang_damping.to(device)
model.unicycle_network.mass_vector = model.unicycle_network.mass_vector.to(device)
model.unicycle_network.j_vector = model.unicycle_network.j_vector.to(device)

model.s_init[:,0] = 0
model.omega_init[:,0] = 0
# #%%
# valid_score = test(valid_loader)
# test_score = test(test_loader)
# print(f"Validation score: {valid_score}")
# print(f"Test score: {test_score}")
#%%
def lattice_adjacency_matrix(rows, cols, connect_diagonal=False):
    """Generate an adjacency matrix for a grid with optional diagonal connections."""
    N = rows * cols
    adjacency_matrix = np.zeros((N, N), dtype=int)

    def index(r, c):
        """Convert 2D grid coordinates to 1D index."""
        return r * cols + c

    # Loop through each grid cell
    for r in range(rows):
        for c in range(cols):
            i = index(r, c)

            # 4-connected neighbors
            if r > 0:  # Up
                adjacency_matrix[i, index(r - 1, c)] = 1
            if r < rows - 1:  # Down
                adjacency_matrix[i, index(r + 1, c)] = 1
            if c > 0:  # Left
                adjacency_matrix[i, index(r, c - 1)] = 1
            if c < cols - 1:  # Right
                adjacency_matrix[i, index(r, c + 1)] = 1

            # 8-connected neighbors (diagonals)
            if connect_diagonal:
                if r > 0 and c > 0:  # Top-left
                    adjacency_matrix[i, index(r - 1, c - 1)] = 1
                if r > 0 and c < cols - 1:  # Top-right
                    adjacency_matrix[i, index(r - 1, c + 1)] = 1
                if r < rows - 1 and c > 0:  # Bottom-left
                    adjacency_matrix[i, index(r + 1, c - 1)] = 1
                if r < rows - 1 and c < cols - 1:  # Bottom-right
                    adjacency_matrix[i, index(r + 1, c + 1)] = 1

    return adjacency_matrix + adjacency_matrix.T  # Ensure symmetry
#%%
adjacency_matrix = lattice_adjacency_matrix(num_rows, num_cols, False)
#%%
model.unicycle_network.stiffness_coupling_matrix = torch.nn.Parameter(torch.from_numpy(adjacency_matrix).to(device), requires_grad=False)
#%%
x = model.x_init[0:1,:]
z = model.z_init[0:1,:]
theta = model.theta_init[0:1,:]
s = model.s_init[0:1,:]
omega = model.omega_init[0:1,:]
states_list = []
#%%
u_lin = torch.zeros((1, 4000, 1), device=device)
u_ang = torch.zeros_like(u_lin, device=device)

for t in range(u_lin.size()[1]):
    linear_input = torch.tanh(u_lin[:, t] +model.inp_bias) @ model.lin_input_map
    angular_input = u_ang[:, t] @ model.ang_input_map

    x, z, theta, s, omega = model.unicycle_network(linear_input, angular_input, x, z, theta, s, omega)

    concatenated_states = torch.hstack((x, z, theta, s, omega))
    states_list.append(concatenated_states)
#%%
all_states_time = torch.vstack(states_list)
plt.plot(all_states_time[:,0:n_units].cpu().detach().numpy())
plt.show()
plt.plot(all_states_time[:,n_units*3:n_units*4].cpu().detach().numpy())
plt.show()
plt.plot(all_states_time[:,n_units*4:n_units*5].cpu().detach().numpy())
plt.show()
#%%
model.set_init_states(bs_train, x,z,theta,s,omega)
perm = torch.randperm(784).to(device)
#%%
def plot_graph(x_coords, y_coords, adjacency_matrix, save_path=None):
    """
    Plots N points in 2D based on x_coords and y_coords.
    Draws a line between points i and j if adjacency_matrix[i, j] is nonzero.
    
    Parameters:
    x_coords (array-like): 1D array of x coordinates of N points.
    y_coords (array-like): 1D array of y coordinates of N points.
    adjacency_matrix (2D array): Symmetric N x N matrix representing connections.
    """
    N = len(x_coords)
    
    if len(y_coords) != N or adjacency_matrix.shape != (N, N):
        raise ValueError("Input dimensions are inconsistent.")

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot points
    ax.scatter(x_coords, y_coords, color='blue', zorder=2)

    # Draw edges based on adjacency matrix
    for i in range(N):
        for j in range(i+1, N):  # Only check upper triangle (since matrix is symmetric)
            if adjacency_matrix[i, j] != 0:
                ax.plot([x_coords[i], x_coords[j]], [y_coords[i], y_coords[j]], 'k-', alpha=0.5, zorder=1)

    # Label points
    for i in range(N):
        ax.text(x_coords[i], y_coords[i], str(i), fontsize=12, ha='right', color='red')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("2D Graph Plot")
    plt.grid(True)
    if save_path:
        plt.savefig(f"plots/{save_path}")
    plt.show()
#%%
plot_graph(x.T.cpu().detach().numpy(),z.T.cpu().detach().numpy(), model.unicycle_network.stiffness_coupling_matrix.cpu().detach().numpy(), 
save_path=None)
#%%
for epoch in range(n_epochs):
    model.train()
    progress_bar = tqdm(train_loader)
    for images, labels in progress_bar:
        images = images.reshape(images.shape[0], 1, 784)
        images = images.permute(0,2,1)
        angular_input = torch.zeros_like(images)

        images = images.to(device)
        # images = images[:, perm, :]
        angular_input = angular_input.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        # with torch.no_grad():
        states_list, output = model(angular_input, images)
        loss = objective(output, labels)
        loss.backward()
        # Check gradients
        # if epoch % 10 == 0:
        #     for name, param in model.named_parameters():
        #         if param.grad is not None:
        #             print(f"Gradient of {name}:")
        #             print(param.grad)
        #         else:
        #             print(f"No gradient computed for {name}")
        optimizer.step()
        progress_bar.set_postfix(loss=loss.item())
    valid_score = test(valid_loader)
    test_score = test(test_loader)
    print(f"Validation score: {valid_score}")
    print(f"Test score: {test_score}")
    # print(model.lin_input_map)
           # %%
#%%
sample_idx = 16
print(labels[sample_idx])
#%%
all_states_time_res = torch.stack(states_list, dim=1)
plt.plot(all_states_time_res[sample_idx,:,0:n_units].cpu().detach().numpy())
plt.show()
plt.plot(all_states_time_res[sample_idx,:,1*n_units:2*n_units].cpu().detach().numpy())
plt.show()
plt.plot(all_states_time_res[sample_idx,:,2*n_units:3*n_units].cpu().detach().numpy())
plt.show()
plt.plot(all_states_time_res[sample_idx,:,3*n_units:4*n_units].cpu().detach().numpy())
plt.show()
plt.plot(all_states_time_res[sample_idx,:,4*n_units:5*n_units].cpu().detach().numpy())
plt.show()
# plt.plot(all_states_time_res[sample_idx,:,n_units*3:n_units*4].cpu().detach().numpy())
# plt.show()
# %%
x_states = all_states_time_res[sample_idx,:,0:n_units].detach().cpu().numpy().T
y_states = all_states_time_res[sample_idx,:,n_units:2*n_units].detach().cpu().numpy().T
theta_states = all_states_time_res[sample_idx,:,n_units*2:3*n_units].detach().cpu().numpy().T
t_steps = 784
#%% ### To render to different evolutions on top of each other
sample_idx_second = 1
x_states = np.vstack((x_states, all_states_time_res[sample_idx_second,:,0:n_units].detach().cpu().numpy().T))
y_states = np.vstack((y_states, all_states_time_res[sample_idx_second,:,n_units:2*n_units].detach().cpu().numpy().T))
theta_states = np.vstack((theta_states, all_states_time_res[sample_idx_second,:,n_units*2:3*n_units].detach().cpu().numpy().T))
# %%
fig, ax = plt.subplots()
# Calculate the data ranges
x_range = x_states.max() - x_states.min()
y_range = y_states.max() - y_states.min()

# Set axis limits with a margin
# ax.set_xlim(x_states.min() - 0.2, x_states.max() + 0.2)
# ax.set_ylim(y_states.min() - 0.2, y_states.max() + 0.2)

ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)

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
ani.save(f'state_evolution_{labels[sample_idx].detach()}.mp4')

# %%
fig, ax = plt.subplots()
# Calculate the data ranges
x_range = x_states.max() - x_states.min()
y_range = y_states.max() - y_states.min()

# Set axis limits with a margin
ax.set_xlim(x_states.min() - 0.2, x_states.max() + 0.2)
ax.set_ylim(y_states.min() - 0.2, y_states.max() + 0.2)

# ax.set_xlim(-4, 1)
# ax.set_ylim(-4, 2)

# Dynamically set scale and width based on the data range
# These factors can be adjusted based on how the arrows appear
scale_factor = 1.5 * min(x_range, y_range).item()
width_factor = 0.002 * min(x_range, y_range).item()

# Initialize the positions and directions for the arrows
x_pos = x_states[:, 0]
y_pos = y_states[:, 0]
u = np.cos(theta_states[:, 0])  # x components of the arrow directions
v = np.sin(theta_states[:, 0])  # y components of the arrow directions

colors = np.linspace(0, 1, n_units)  # Generate values from 0 to 1
cmap = plt.get_cmap("hsv")  # Use HSV colormap for distinct colors

arrow_colors = cmap(colors)

# colors_1 = np.ones(20)*0.5
# colors_0 = np.zeros(20)

# colors = np.hstack((colors_0, colors_1))
# arrow_colors = cmap(colors)  # Map colors to colormap

# Initialize a Quiver object for the arrows
quiver = ax.quiver(x_pos, y_pos, u, v, angles='xy', scale_units='xy', scale=scale_factor, width=width_factor, color=arrow_colors)
def update_no_history(frame):
    # Get the current positions
    x_pos = x_states[:, frame]
    y_pos = y_states[:, frame]
    
    # Get the current orientations
    u = np.cos(theta_states[:, frame])  # x components of the arrow directions
    v = np.sin(theta_states[:, frame])  # y components of the arrow directions
    # Update the quiver with the current positions and orientations
    quiver.set_offsets(np.c_[x_pos, y_pos])  # Update the arrow positions
    quiver.set_UVC(u, v)  # Update the arrow directions

    # quiver_history.append(quiver)
    return quiver,
#%%
print(labels[sample_idx])
#%%
# Create animation
ani = FuncAnimation(fig, update_no_history, frames=784, blit=True)
ani.save(f'animations/state_evolution_{labels[sample_idx].detach()}_grid.mp4')
# %%
