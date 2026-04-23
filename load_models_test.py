#%%
import torch
from torch import nn, optim
from tqdm import tqdm
import optuna
from unicycle_network_jax import UnicycleModel
from utils import get_mnist_data
from unicycle_network_jax import ReadoutLayer
import jax
import jax.numpy as jnp
import numpy as np
import time
from jax import random
import pickle
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
#%%

path = "/media/mariano/05757408-5986-4b0d-930b-043e4c78f5bc/mariano/phd_code/unicycle_unit/models/"
readout_layer = torch.load(path+"linear_layer_2_classes.pth")

with open(path + "unicycle_net_2_classes", 'rb') as f:
    unicycle_net = pickle.load(f)
#%%
dt=0.001
# %%
# Evaluation function for validation and testing accuracy
def test(data_loader, unicycle_net, readout_layer, objective_fn):
    readout_layer.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images = images.reshape(images.shape[0], 1, 784)
            images = images.permute(2, 0, 1)
            images = jnp.array(images)
            angular_input = jnp.zeros_like(images)

            state_sequence, final_state = unicycle_net.reservoir_forward(images, angular_input, None, dt=dt, steps=784)
            final_state = jnp.hstack(final_state)
            output = readout_layer(torch.from_numpy(np.asarray(final_state).copy()))
            test_loss += objective_fn(output, labels).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum()

    accuracy = 100. * correct / len(data_loader.dataset)
    return accuracy.item(), state_sequence
#%%
bs_train, bs_test = 50,50
classes = [0,1]
objective_fn = nn.CrossEntropyLoss()
train_loader, valid_loader, test_loader = get_mnist_data(bs_train=bs_train, bs_test=bs_test, classes=classes)

validation_accuracy = test(valid_loader, unicycle_net, readout_layer, objective_fn)
test_accuracy = test(test_loader, unicycle_net, readout_layer, objective_fn)
#%%
print(validation_accuracy, test_accuracy)
# %%
study_name = f"unicycle_opt_2_classes_normal_init_jax"
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(storage=storage_name, study_name=study_name, direction='maximize', load_if_exists="True")

params = study.best_params
print(params)
n_units = params['n_units']
lr = params['lr']
lin_stiff_min = params['lin_stiff_min']
lin_stiff_max = params['lin_stiff_max']
ang_stiff_min = params['ang_stiff_min']
ang_stiff_max = params['ang_stiff_max']
lin_damping_min = params['lin_damping_min']
lin_damping_max = params['lin_damping_max']
ang_damping_min = params['ang_damping_min']
ang_damping_max = params['ang_stiff_max']
num_non_zero = params['non_zero_elements']
non_zero_values = params['magnitude_min']
# n_connections = trial.suggest_int('n_connections', int(n_units / 10), n_units)
sparsity_level = params['sparsity']
bs = params['batch_size']

#%%
rng = jax.random.PRNGKey(0)
unicycle_net_random = UnicycleModel(rng=rng, n_units=n_units, n_inp=1, batch_size=bs_train, lin_stiff_min=lin_stiff_min, lin_stiff_max=lin_stiff_max,
    ang_stiff_min=ang_stiff_min, ang_stiff_max=ang_stiff_max,
    lin_damping_min=lin_damping_min, lin_damping_max=lin_damping_max,
    ang_damping_min=ang_damping_min, ang_damping_max=ang_damping_max, sparsity_level=sparsity_level)
unicycle_net_random.initialize_input_map(1, n_units=n_units, non_zero_min_magnitude=non_zero_values, num_non_zero=num_non_zero)
#%%
validation_accuracy = test(valid_loader, unicycle_net_random, readout_layer, objective_fn)
test_accuracy = test(test_loader, unicycle_net_random, readout_layer, objective_fn)
#%%
print(validation_accuracy, test_accuracy)
# %%
with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images = images.reshape(images.shape[0], 1, 784)
        images = images.permute(2, 0, 1)
        images = jnp.array(images)
        angular_input = jnp.zeros_like(images)

        state_sequence, final_state = unicycle_net.reservoir_forward(images, angular_input, None, dt=dt, steps=784)
        final_state = jnp.hstack(final_state)
        output = readout_layer(torch.from_numpy(np.asarray(final_state).copy()))
        break
#%%
probs = nn.Softmax(dim=1)(output)
# %%
labels.shape
# %%
state_sequence.shape
# %%
batch_index = 0
n_units = unicycle_net.n_units
t_steps = state_sequence.shape[0]
#%%
x_states = state_sequence[:,batch_index,0:n_units].T
y_states = state_sequence[:,batch_index,n_units:n_units*2].T
theta_states = state_sequence[:,batch_index,n_units*2:n_units*3].T
s_states = state_sequence[:,batch_index,n_units*3:n_units*4].T
omega_states = state_sequence[:,batch_index,n_units*4:n_units*5].T
#%%
x_states.shape
# %%
fig, ax = plt.subplots()

# Calculate the data ranges
x_range = x_states.max() - x_states.min()
y_range = y_states.max() - y_states.min()

# Set axis limits with a margin
ax.set_xlim(x_states.min() - 0.2, x_states.max() + 0.2)
ax.set_ylim(y_states.min() - 0.2, y_states.max() + 0.2)

ax.set_xlabel("x [-]")
ax.set_ylabel("y [-]")

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
ani.save('animation_with_arrows_.mp4')
#%%
# %%
fig, ax = plt.subplots()

# Calculate the data ranges
x_range = x_states.max() - x_states.min()
y_range = y_states.max() - y_states.min()

# Set axis limits with a margin
ax.set_xlim(-8,8)
ax.set_ylim(-8,8)
ax.set_xlabel("x [-]")
ax.set_ylabel("y [-]")
# Dynamically set scale and width based on the data range
# These factors can be adjusted based on how the arrows appear
scale_factor = 0.1 * min(x_range, y_range).item()
width_factor = 0.001 * min(x_range, y_range).item()

# Initialize the positions and directions for the arrows
x_pos = x_states[:, 0]
y_pos = y_states[:, 0]
u = np.cos(theta_states[:, 0])  # x components of the arrow directions
v = np.sin(theta_states[:, 0])  # y components of the arrow directions

colors = ['blue', 'orange', 'green', 'red', 'purple']
num_arrows = x_pos.shape[0]
arrow_colors = [colors[i % len(colors)] for i in range(num_arrows)]  # Initial colors

# Initialize a Quiver object for the arrows
quiver = ax.quiver(x_pos, y_pos, u, v, angles='xy', scale_units='xy', scale=scale_factor, width=width_factor, color=arrow_colors)

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
    quiver.set_offsets(np.c_[x_pos, y_pos])  # Update the arrow positions
    quiver.set_UVC(u, v)  # Update the arrow directions
    # quiver = ax.quiver(x_pos, y_pos, u, v, angles='xy', scale_units='xy', scale=scale_factor, width=width_factor, color=['blue', 'orange', 'green'])

    return quiver,

# Create animation
ani = FuncAnimation(fig, update, frames=t_steps, blit=True)
ani.save('animation_with_arrows_no_history.mp4')
# %%
plt.plot(theta_states.T)
# %%
n=4
torch.abs(readout_layer.state_dict()['linear.weight'][0,80*n:80*(n+1)]).mean()
# %%
def plot_single_arrow_plot(x_states, y_states, theta_states, digit, index):
    fig, ax = plt.subplots()

    # # Calculate the data ranges
    # x_range = x_states.max() - x_states.min()
    # y_range = y_states.max() - y_states.min()

    # Set axis limits with a margin
    ax.set_xlim(-8,8)
    ax.set_ylim(-8,8)

    # Dynamically set scale and width based on the data range
    # These factors can be adjusted based on how the arrows appear
    scale_factor = 0.1 * 16
    width_factor = 0.001 * 16
    # Initialize the positions and directions for the arrows
    x_pos = x_states
    y_pos = y_states
    u = np.cos(theta_states)  # x components of the arrow directions
    v = np.sin(theta_states)  # y components of the arrow directions

    colors = ['blue', 'orange', 'green', 'red', 'purple']
    num_arrows = x_pos.shape[0]
    arrow_colors = [colors[i % len(colors)] for i in range(num_arrows)]  # Initial colors

    # Initialize a Quiver object for the arrows
    quiver = ax.quiver(x_pos, y_pos, u, v, angles='xy', scale_units='xy', scale=scale_factor, width=width_factor, color=arrow_colors)
    fig.savefig(f"./final_state_plots/{digit}/plot_{index}")
#%%
indices_zero = np.where(labels==0)[0]
indices_one = np.where(labels==1)[0]
# %%
for index in indices_zero:
    x_states = state_sequence[-1,index,0:n_units].T
    y_states = state_sequence[-1,index,n_units:n_units*2].T
    theta_states = state_sequence[-1,index,n_units*2:n_units*3].T
    plot_single_arrow_plot(x_states, y_states, theta_states, digit="zero", index=index)
#%%
for index in indices_one:
    x_states = state_sequence[-1,index,0:n_units].T
    y_states = state_sequence[-1,index,n_units:n_units*2].T
    theta_states = state_sequence[-1,index,n_units*2:n_units*3].T
    plot_single_arrow_plot(x_states, y_states, theta_states, digit="one", index=index)
# %%
x_states_all = np.empty((0,1))
y_states_all = np.empty((0,1))
theta_states_all = np.empty((0,1))

for index in indices_zero:
    x_states = state_sequence[-1,index,0:n_units].T.reshape((-1,1))
    y_states = state_sequence[-1,index,n_units:n_units*2].T.reshape((-1,1))
    theta_states = state_sequence[-1,index,n_units*2:n_units*3].T.reshape((-1,1))
    x_states_all = np.vstack((x_states_all, x_states))
    y_states_all = np.vstack((y_states_all, y_states))
    theta_states_all = np.vstack((theta_states_all, theta_states))

plot_single_arrow_plot(x_states_all, y_states_all, theta_states_all, digit="zero", index='all')
#%%
x_states_all = np.empty((0,1))
y_states_all = np.empty((0,1))
theta_states_all = np.empty((0,1))

for index in indices_one:
    x_states = state_sequence[-1,index,0:n_units].T.reshape((-1,1))
    y_states = state_sequence[-1,index,n_units:n_units*2].T.reshape((-1,1))
    theta_states = state_sequence[-1,index,n_units*2:n_units*3].T.reshape((-1,1))
    x_states_all = np.vstack((x_states_all, x_states))
    y_states_all = np.vstack((y_states_all, y_states))
    theta_states_all = np.vstack((theta_states_all, theta_states))
plot_single_arrow_plot(x_states_all, y_states_all, theta_states_all, digit="one", index='all')

# %%
