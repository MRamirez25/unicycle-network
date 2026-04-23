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

study_name = f"unicycle_opt_7_classes_normal_init_jax"
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(storage=storage_name, study_name=study_name, direction='maximize', load_if_exists="True")

params = study.best_params
print(params)
n_units = 150
lr = 0.001
lin_stiff_min = 0.1
lin_stiff_max = 0.2
ang_stiff_min = 0.1
ang_stiff_max = 0.3
lin_damping_min = 5.
lin_damping_max = 10.
ang_damping_min = 0.1
ang_damping_max = 0.2
num_non_zero = 100
non_zero_values = 0.5
# n_connections = trial.suggest_int('n_connections', int(n_units / 10), n_units)
sparsity_level = 0.95
bs = 100

bs_train, bs_test = bs, bs
classes = [0,1,2,3,4,5,6]
train_loader, valid_loader, test_loader = get_mnist_data(bs_train=bs_train, bs_test=bs_test, classes=classes)

#%%
# Initialize the model with the suggested hyperparameters
rng = jax.random.PRNGKey(0)
unicycle_net = UnicycleModel(rng=rng, n_units=n_units, n_inp=1, batch_size=bs_train, lin_stiff_min=lin_stiff_min, lin_stiff_max=lin_stiff_max,
    ang_stiff_min=ang_stiff_min, ang_stiff_max=ang_stiff_max,
    lin_damping_min=lin_damping_min, lin_damping_max=lin_damping_max,
    ang_damping_min=ang_damping_min, ang_damping_max=ang_damping_max, sparsity_level=sparsity_level)
unicycle_net.initialize_input_map(1, n_units=n_units, non_zero_min_magnitude=non_zero_values, num_non_zero=num_non_zero)
#%%
if classes is not None:
    n_classes = len(classes)
else:
    n_classes = 10
readout_layer = ReadoutLayer(n_units*5, n_classes)
#%%
optimizer = optim.Adam(readout_layer.parameters(), lr=lr)
objective_fn = nn.CrossEntropyLoss()

n_epochs = 1  # Reduce the number of epochs to speed up the process
# unicycle_net.set_init_states(bs_train)
dt = 0.001
# # Move initial states to GPU if available
# unicycle_net.x_init = unicycle_net.x_init.to(device)
# unicycle_net.z_init = unicycle_net.z_init.to(device)
# unicycle_net.theta_init = unicycle_net.theta_init.to(device)
# unicycle_net.s_init = unicycle_net.s_init.to(device)
# unicycle_net.omega_init = unicycle_net.omega_init.to(device)
# unicycle_net.lin_input_map = unicycle_net.lin_input_map.to(device)
# unicycle_net.ang_input_map = unicycle_net.ang_input_map.to(device)
# unicycle_net.unicycle_network.lin_damping = unicycle_net.unicycle_network.lin_damping.to(device)
# unicycle_net.unicycle_network.ang_damping = unicycle_net.unicycle_network.ang_damping.to(device)

#%%
zero_input = jnp.zeros((1000,1,1))
_, final_state_eq = unicycle_net.reservoir_forward(zero_input, zero_input, init_state=None, dt=dt,steps=1000)
unicycle_net.init_state = final_state_eq
#%%
# Train and validate the model
for epoch in range(n_epochs):
    readout_layer.train()
    progress_bar = tqdm(train_loader)
    for i, (images, labels) in enumerate(progress_bar):
        images = images.reshape(images.shape[0], 1, 784)
        images = images.permute(2, 0, 1)
        images = jnp.array(images)
        angular_input = jnp.zeros_like(images)

        optimizer.zero_grad()
        # start = time.time()
        _, final_state = unicycle_net.reservoir_forward(images, angular_input, None, dt=dt, steps=784)
        # end = time.time()
        final_state = jnp.hstack(final_state)
        output = readout_layer(torch.from_numpy(np.asarray(final_state).copy()))
        # print(f"Dynamics time {end-start}")
        loss = objective_fn(output, labels)
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix(loss=loss.item())


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

            _, final_state = unicycle_net.reservoir_forward(images, angular_input, None, dt=dt, steps=784)
            final_state = jnp.hstack(final_state)
            output = readout_layer(torch.from_numpy(np.asarray(final_state).copy()))
            test_loss += objective_fn(output, labels).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum()
    accuracy = 100. * correct / (len(data_loader.dataset))
    return accuracy.item()

validation_accuracy = test(valid_loader, unicycle_net, readout_layer, objective_fn)
test_accuracy = test(test_loader, unicycle_net, readout_layer, objective_fn)

print(f"accuracies {validation_accuracy} {test_accuracy}")

# path = "/media/mariano/05757408-5986-4b0d-930b-043e4c78f5bc/mariano/phd_code/unicycle_unit/models/"
# torch.save(readout_layer, path+"linear_layer_2_classes.pth")

# with open(path + "unicycle_net_2_classes", 'wb') as f:
#     pickle.dump(unicycle_net, f)
# %%
