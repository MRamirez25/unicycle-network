#%%
import os
import sys
import time
#%%
# Add the parent directory to the system path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
#%%
from utils import get_FordA_data, n_params
from tqdm import tqdm
import matplotlib.pyplot as plt
from unicycle_network_class import UnicycleNetwork, UnicycleReservoir

from torch import nn, optim
import torch
import optuna
from matplotlib.animation import FuncAnimation
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import random

# %%
storage_name = f"unicycle_nets_forda_logreg"
study_name = "only_last_state_as_readout_20_units_not_aligned_less_ang_magnitude_no_tanh"
storage_name = f"sqlite:///{parent_dir}/optuna_databases/{storage_name}.db"
study = optuna.create_study(storage=storage_name, study_name=study_name, direction='maximize', load_if_exists="True")
params = study.best_params
#%%
aligned_orientations = False  # Set to True if you want aligned orientations, False otherwise
ang_input = True  # Set to True if you want angular input, False otherwise
ang_connections = True  # Set to True if you want angular connections, False otherwise
#%%
n_units = 20#params['n_units']
lr = params['lr']
lin_stiff_min =  params['lin_stiff_min']
lin_stiff_max =  params['lin_stiff_max']
ang_stiff_min =  params['ang_stiff_min']
ang_stiff_max =  params['ang_stiff_max']
lin_damping_min =  params['lin_damping_min']
lin_damping_max =  params['lin_damping_max']
ang_damping_min =  params['ang_damping_min']
ang_damping_max  = params['ang_damping_max']
bs_train = params['batch_size']
bs_test = bs_train
dt = params['dt']
inp_bias =  params['inp_bias']
anchor_con_fraction = params['anchor_con_fraction']
num_non_zero = params['non_zero_elements']
magnitude_min = params['magnitude_min']
magnitude_max = params['magnitude_max']
if ang_input:
    non_zero_elements_ang = params['non_zero_elements_ang']
    magnitude_min_ang = params['magnitude_min_ang']
    magnitude_max_ang = params['magnitude_max_ang']
else:
    non_zero_elements_ang = 0
    magnitude_min_ang = 0
    magnitude_max_ang = 0
n_connections_fraction = params['n_connections_fraction']
n_connections = int(n_units*n_connections_fraction)
washup = params['washup_steps']
n_steps_readout = 0#params['steps_readout']
if ang_connections:
    n_connections_ang_fraction = params['n_connections_ang_fraction']
    anchor_con_fraction_ang = params['anchor_con_fraction_ang']
    n_connections_anchor_ang = int(anchor_con_fraction_ang * n_units)
    n_connections_ang = int(n_connections_ang_fraction*n_units)
else:
    anchor_con_fraction_ang = 0
    n_connections_anchor_ang = 0
    n_connections_ang = 0

eq_dist_min = params['eq_dist_min']
eq_dist_max = params['eq_dist_max']
eq_dist_min_ang = params['eq_dist_min_ang']
eq_dist_max_ang = params['eq_dist_max_ang']
# n_epochs = params['n_epochs']
n_connections_anchor = int(n_units * anchor_con_fraction)
#%%
seed = 33
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# If using CUDA
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

# Ensure deterministic behavior (might affect performance)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#%%
root = parent_dir + '/data/'
# Load the FordA dataset
train_loader, valid_loader, test_loader = get_FordA_data(bs_train, bs_test)
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
model = UnicycleReservoir(n_inp=1, n_units=n_units, dt=dt, n_out=2, lin_input_map=lin_input_map, 
                          lin_stiff_min=lin_stiff_min, lin_damping_min=lin_damping_min, lin_damping_max=lin_damping_max, lin_stiff_max=lin_stiff_max,
                          eq_dist_min=eq_dist_min, eq_dist_max=eq_dist_max, eq_dist_min_ang=eq_dist_min_ang,
                          eq_dist_max_ang=eq_dist_max_ang,  
                          n_connections=n_connections, n_connections_anchor=n_connections_anchor, 
                          n_past_steps_readout=n_steps_readout, n_connections_ang=n_connections_ang, n_connections_anchor_ang=n_connections_anchor_ang,
                          ang_stiff_min=ang_stiff_min, ang_stiff_max=ang_stiff_max, ang_damping_min=ang_damping_min, ang_damping_max=ang_damping_max,
                          inp_bias=inp_bias, ang_input_map=ang_input_map)
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")  # Force CPU for testing purposes
print("Using", device)
model = model.to(device)
#%%
def test(data_loader):
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for i, (x, labels) in enumerate(data_loader):
            x = x.to(device)
            labels = labels.to(device)
            _, output = model(x, x)
            test_loss += objective(output, labels).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum()
    test_loss /= i+1
    accuracy = 100. * correct / len(data_loader.dataset)

    return accuracy.item()

#%%
@torch.no_grad()
def test_esn(data_loader, model, classifier, scaler):
    activations, ys = [], []
    for x, labels in tqdm(data_loader):
        x = x.to(device)
        labels = labels.to(device)
        states_list, output, mid_states = model(x, x)
        mid_states = mid_states[:,:]  # Apply the same permutation to mid_states
        activations.append(mid_states.cpu())
        ys.append(labels.cpu())
    activations = torch.cat(activations, dim=0).numpy()
    activations = scaler.transform(activations)
    ys = torch.cat(ys, dim=0).numpy()
    return classifier.score(activations, ys)
#%%
for i, (x, labels) in enumerate(test_loader):
    print(f"FordA data shape: {x.shape}, labels shape: {labels.shape}")
#%%
# n_epochs = n_epochs
objective = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
#%%
# # %%
# for parameter in model.parameters():
#     print(parameter)
model.set_init_states_random(bs_train)
model.x_init = model.x_init.to(device)
model.z_init = model.z_init.to(device)
model.theta_init = model.theta_init.to(device)
model.s_init = model.s_init.to(device)
model.omega_init = model.omega_init.to(device)
model.lin_input_map = model.lin_input_map.to(device)
model.ang_input_map = model.ang_input_map.to(device)
model.unicycle_network.lin_damping = model.unicycle_network.lin_damping.to(device)
model.unicycle_network.ang_damping = model.unicycle_network.ang_damping.to(device)
model.unicycle_network.mass_vector = model.unicycle_network.mass_vector.to(device)
model.unicycle_network.j_vector = model.unicycle_network.j_vector.to(device)

model.s_init[:,0] = 0
model.omega_init[:,:] = 0
if not aligned_orientations:
    model.theta_init[:,:] = torch.rand(model.theta_init.size()) * (2*torch.pi - (-2*torch.pi)) + -2*torch.pi
    model.theta_init[:,0] = 0
else:
    print("aligned orientations")
    model.theta_init[:,:] = torch.rand(1) * (2*torch.pi - (-2*torch.pi)) + -2*torch.pi
#%%
x = model.x_init[0:1,:]
z = model.z_init[0:1,:]
theta = model.theta_init[0:1,:]
s = model.s_init[0:1,:]
omega = model.omega_init[0:1,:]
states_list = []
#%%
u_lin = torch.zeros((1, washup, 1), device=device)
u_ang = torch.zeros_like(u_lin, device=device)

for t in range(u_lin.size()[1]):
    linear_input = (u_lin[:, t]) @ model.lin_input_map
    angular_input = u_ang[:, t] @ model.ang_input_map

    x, z, theta, s, omega = model.unicycle_network(linear_input, angular_input, x, z, theta, s, omega)

    concatenated_states = torch.hstack((x, z, theta, s, omega))
    states_list.append(concatenated_states)
#%%
all_states_time = torch.vstack(states_list)
plt.plot(all_states_time[:,0:n_units].cpu().detach().numpy())
plt.show()
plt.plot(all_states_time[:,n_units*2:n_units*3].cpu().detach().numpy())
plt.show()
plt.plot(all_states_time[:,n_units*4:n_units*5].cpu().detach().numpy())
plt.show()
#%%
model.set_init_states(bs_train, x,z,theta,s,omega)
perm = torch.randperm(784).to(device)

#%%
# Logistic regression ESN training loop
progress_bar = tqdm(train_loader)
activations, ys = [], []
for x, labels in progress_bar:
    x = x.to(device)
    labels = labels.to(device)
    # model should return mid_states as third output
    states_list, output, mid_states = model(x, x)
    mid_states = mid_states[:,:]  # Apply the same permutation to mid_states
    activations.append(mid_states.detach().cpu())
    ys.append(labels.cpu())

activations = torch.cat(activations, dim=0).numpy()
ys = torch.cat(ys, dim=0).numpy()

# Check for NaN values in activations
import numpy as np
if np.isnan(activations).any():
    print("NaN values detected in activations")
else:
    print("No NaN values detected, proceeding with training")

scaler = preprocessing.StandardScaler().fit(activations)
activations = scaler.transform(activations)
classifier = LogisticRegression(max_iter=1000).fit(activations, ys)

# Validation and test scores using ESN
valid_score = test_esn(valid_loader, model, classifier, scaler)
test_score = test_esn(test_loader, model, classifier, scaler)
print(f"Validation score (ESN): {valid_score}")
print(f"Test score (ESN): {test_score}")
# %%
# Final validation and test scores
print(f"Number of trainable parameters: {n_params(classifier)}")
#%%
sample_idx = 7
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
t_steps = 500
#%% ### To render to different evolutions on top of each other
sample_idx_second = 5
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
ani.save(f'state_evolution_{labels[sample_idx].detach()}.mp4', dpi=300)

# %%
fig, ax = plt.subplots()
# Calculate the data ranges
x_range = x_states.max() - x_states.min()
y_range = y_states.max() - y_states.min()

# Set axis limits with a margin
ax.set_xlim(x_states.min() - 0.2, x_states.max() + 0.2)
ax.set_ylim(y_states.min() - 0.2, y_states.max() + 0.2)

# ax.set_xlim(-6, 6)
# ax.set_ylim(-6, 6)

# Dynamically set scale and width based on the data range
# These factors can be adjusted based on how the arrows appear
scale_factor = 0.3 * min(x_range, y_range).item()
width_factor = 0.001 * min(x_range, y_range).item()

# Initialize the positions and directions for the arrows
x_pos = x_states[:, 0]
y_pos = y_states[:, 0]
u = np.cos(theta_states[:, 0])  # x components of the arrow directions
v = np.sin(theta_states[:, 0])  # y components of the arrow directions

colors = np.linspace(0, 1, n_units)  # Generate values from 0 to 1
cmap = plt.get_cmap("hsv")  # Use HSV colormap for distinct colors

colors_1 = np.ones(n_units)*0.55
colors_0 = np.zeros(n_units)

colors = np.hstack((colors_0, colors_1))
arrow_colors = cmap(colors)  # Map colors to colormap
plt.tick_params(left=False, right=False, labelleft=False, 
                labelbottom=False, bottom=False)
plt.tight_layout()
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
ani = FuncAnimation(fig, update_no_history, frames=t_steps, blit=True)
ani.save(f'state_evolution_{labels[sample_idx].detach().item()}_latest.mp4', dpi=300)
# %%
# %%
x_states_last = states_list[-1].T[sample_idx,0:n_units].detach().cpu().numpy().T
y_states_last = states_list[-1].T[sample_idx,n_units:2*n_units].detach().cpu().numpy().T
theta_states_last = states_list[-1].T[sample_idx,2*n_units:3*n_units].detach().cpu().numpy().T

# %%
idxs_0 = torch.where(labels == 0)[0].cpu().detach().numpy()
idxs_1 = torch.where(labels == 1)[0].cpu().detach().numpy()

# %%
x_states_last_0 = np.empty((n_units,0))
for idx in idxs_0:
    x_states = states_list[-1][idx,0:n_units].detach().cpu().numpy().reshape((n_units,1))
    x_states_last_0 = np.hstack((x_states_last_0, x_states))
# %%
y_states_last_0 = np.empty((n_units,0))
for idx in idxs_0:
    y_states = states_list[-1][idx,n_units:2*n_units].detach().cpu().numpy().reshape((n_units,1))
    y_states_last_0 = np.hstack((y_states_last_0, y_states))

# %%
x_states_last_1 = np.empty((n_units,0))
for idx in idxs_1:
    x_states = states_list[-1][idx,0:n_units].detach().cpu().numpy().reshape((n_units,1))
    x_states_last_1 = np.hstack((x_states_last_1, x_states))
# %%
y_states_last_1 = np.empty((n_units,0))
for idx in idxs_1:
    y_states = states_list[-1][idx,n_units:2*n_units].detach().cpu().numpy().reshape((n_units,1))
    y_states_last_1 = np.hstack((y_states_last_1, y_states))
# %%
for n in range(8,9):
    plt.scatter(x_states_last_0[n,:], y_states_last_0[n,:])
    plt.scatter(x_states_last_1[n,:], y_states_last_1[n,:])
    plt.show()
#%%
for n in range(n_units):
    plt.scatter(x_states_last_0[n,:], y_states_last_0[n,:])
    plt.xlim(-10,10)
    plt.ylim(-10,10)

#%%
for n in range(n_units):
    plt.scatter(x_states_last_1[n,:], y_states_last_1[n,:])
    plt.xlim(-10,10)
    plt.ylim(-10,10)
#%%
#%%
for n in range(n_units):
    plt.scatter(x_states_last_1[n,:], y_states_last_1[n,:],color='blue')
    plt.scatter(x_states_last_0[n,:], y_states_last_0[n,:],color='orange')
    plt.xlim(-5,5)
    plt.ylim(-5,5)

# %%
# %%
theta_states_last_1 = np.empty((20,0))
for idx in idxs_1:
    theta_states = states_list[-1][idx,n_units:2*n_units].detach().cpu().numpy().reshape((20,1))
    theta_states_last_1 = np.hstack((theta_states_last_1, theta_states))
# %%
# Visualize classifier weights
weights = classifier.coef_[0]  # Shape: (100,) for binary classification
bias = classifier.intercept_[0]

print(f"Weights shape: {weights.shape}")
print(f"Bias: {bias:.4f}")

# Let's first understand how activations are organized
print(f"Activations shape: {activations.shape}")
print("First few activation values:", activations[0, :10])

# Check how mid_states is constructed in your model to understand the ordering
# From your code, mid_states comes from concatenated_states = torch.hstack((x, z, theta, s, omega))
# This means the 100 features are: [x0, x1, ..., x19, z0, z1, ..., z19, theta0, ..., theta19, s0, ..., s19, omega0, ..., omega19]
n_states_classifier = 5
# Reshape weights to (n_states, n_units) = (5, 20) to match [all_x, all_z, all_theta, all_s, all_omega]
weights_reshaped = weights.reshape(n_states_classifier, n_units)  # Shape: (5, 20)

# State names for better visualization
state_names = ['x (position)', 'z (velocity)', 'theta (angle)', 's (angular pos)', 'omega (angular vel)']

# %%
# 1. Heatmap of all weights
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
im = plt.imshow(weights_reshaped, cmap='RdBu_r', aspect='auto')
plt.colorbar(im)
plt.xlabel('Unicycle Unit')
plt.ylabel('State')
plt.title('Classifier Weights Heatmap')
plt.yticks(range(n_states_classifier), state_names[:n_states_classifier])

# %%
# 2. Bar plot of weights by state
plt.subplot(2, 3, 2)
state_means = np.mean(np.abs(weights_reshaped), axis=1)  # Mean across units for each state
bars = plt.bar(range(5), state_means)
plt.xlabel('State')
plt.ylabel('Mean Absolute Weight')
plt.title('Average Weight Magnitude by State')
plt.xticks(range(5), [name.split(' ')[0] for name in state_names], rotation=45)

# Color bars by magnitude
for i, bar in enumerate(bars):
    bar.set_color(plt.cm.viridis(state_means[i] / np.max(state_means)))

# %%
# 3. Weights distribution for each state
plt.subplot(2, 3, 3)
for i, state_name in enumerate(state_names):
    plt.hist(weights_reshaped[i, :], alpha=0.6, label=state_name.split(' ')[0], bins=10)
plt.xlabel('Weight Value')
plt.ylabel('Frequency')
plt.title('Weight Distributions by State')
plt.legend()

# %%
# 4. Individual unit contributions (sum of absolute weights across all states)
plt.subplot(2, 3, 4)
unit_contributions = np.sum(np.abs(weights_reshaped), axis=0)  # Sum across states for each unit
plt.bar(range(n_units), unit_contributions)
plt.xlabel('Unicycle Unit')
plt.ylabel('Total Absolute Weight')
plt.title('Total Contribution by Unit')

# Highlight most important units
top_units = np.argsort(unit_contributions)[-5:]  # Top 5 most important units
for unit in top_units:
    plt.bar(unit, unit_contributions[unit], color='red', alpha=0.7)

# %%
# 5. Scatter plot: weight magnitude vs unit index for each state
plt.subplot(2, 3, 5)
colors = plt.cm.Set1(np.linspace(0, 1, 5))
for i, (state_name, color) in enumerate(zip(state_names, colors)):
    plt.scatter(range(n_units), np.abs(weights_reshaped[i, :]), 
               alpha=0.7, label=state_name.split(' ')[0], color=color, s=30)
plt.xlabel('Unicycle Unit')
plt.ylabel('Absolute Weight')
plt.title('Weight Magnitudes by Unit and State')
plt.legend()

# %%
# 6. Show the most important features
plt.subplot(2, 3, 6)
# Find top 10 most important features (highest absolute weights)
top_features_idx = np.argsort(np.abs(weights))[-10:]
top_weights = weights[top_features_idx]
top_feature_labels = []

for idx in top_features_idx:
    state_idx = idx // n_units  # Which state (0-4)
    unit_idx = idx % n_units    # Which unit (0-19)
    top_feature_labels.append(f'U{unit_idx}_{state_names[state_idx].split(" ")[0]}')

plt.barh(range(len(top_weights)), top_weights)
plt.yticks(range(len(top_weights)), top_feature_labels)
plt.xlabel('Weight Value')
plt.title('Top 10 Most Important Features')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Print summary statistics
print("\nWeight Analysis Summary:")
print("-" * 40)
print(f"Total weights: {len(weights)}")
print(f"Mean absolute weight: {np.mean(np.abs(weights)):.4f}")
print(f"Max absolute weight: {np.max(np.abs(weights)):.4f}")
print(f"Min absolute weight: {np.min(np.abs(weights)):.4f}")
print(f"Standard deviation: {np.std(weights):.4f}")

print(f"\nMost important state (by avg abs weight): {state_names[np.argmax(state_means)]}")
print(f"Least important state (by avg abs weight): {state_names[np.argmin(state_means)]}")

most_important_unit = np.argmax(unit_contributions)
print(f"Most important unit: Unit {most_important_unit} (total abs weight: {unit_contributions[most_important_unit]:.4f})")

# Show which features have the strongest positive and negative weights
max_positive_idx = np.argmax(weights)
max_negative_idx = np.argmin(weights)

max_pos_state = max_positive_idx // n_units
max_pos_unit = max_positive_idx % n_units
max_neg_state = max_negative_idx // n_units
max_neg_unit = max_negative_idx % n_units

print(f"\nStrongest positive weight: Unit {max_pos_unit}, {state_names[max_pos_state]} = {weights[max_positive_idx]:.4f}")
print(f"Strongest negative weight: Unit {max_neg_unit}, {state_names[max_neg_state]} = {weights[max_negative_idx]:.4f}")
