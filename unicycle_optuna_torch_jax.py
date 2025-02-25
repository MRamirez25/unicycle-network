import torch
from torch import nn, optim
from tqdm import tqdm
import optuna
from unicycle_jax_classv2 import UnicycleModel
from utils import get_mnist_data
from unicycle_jax_classv2 import ReadoutLayer, SmallMLP
import jax
import jax.numpy as jnp
import numpy as np
import time
from jax import random
import pdb

# Objective function for Optuna
def objective(trial):
    # Suggest hyperparameters for Optuna to search
    n_units = trial.suggest_int('n_units', 50, 200, step=10)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    lin_stiff_min = trial.suggest_float('lin_stiff_min', 0.1, 1.0)
    lin_stiff_max = trial.suggest_float('lin_stiff_max', lin_stiff_min, 3.0)  # lin_stiff_max >= lin_stiff_min
    ang_stiff_min = trial.suggest_float('ang_stiff_min', 0.1, 1.0)
    ang_stiff_max = trial.suggest_float('ang_stiff_max', ang_stiff_min, 2.0)  # ang_stiff_max >= ang_stiff_min
    lin_damping_min = trial.suggest_float('lin_damping_min', 0.1, 1.0)
    lin_damping_max = trial.suggest_float('lin_damping_max', lin_damping_min, 10.0)
    ang_damping_min = trial.suggest_float('ang_damping_min', 0.1, 1.0)
    ang_damping_max = trial.suggest_float('ang_damping_max', ang_damping_min, 2.0)
    num_non_zero = trial.suggest_int("non_zero_elements", 1, n_units, step=10)
    non_zero_values = trial.suggest_float("magnitude_min", 0.1, 2.0)
    # n_connections = trial.suggest_int('n_connections', int(n_units / 10), n_units)
    sparsity_level = trial.suggest_float("sparsity", 0.8, 0.9)
    bs = trial.suggest_int("batch_size", 50, 300, step=50)
    dt = trial.suggest_float("dt", 0.0001, 0.01, step=0.0005)


    bs_train, bs_test = bs, bs
    classes = [0,1,2,3,4]
    train_loader, valid_loader, test_loader = get_mnist_data(bs_train=bs_train, bs_test=bs_test, classes=classes)


    # Initialize the model with the suggested hyperparameters
    rng = jax.random.PRNGKey(0)
    unicycle_net = UnicycleModel(rng=rng, n_units=n_units, n_inp=1, batch_size=bs_train, lin_stiff_min=lin_stiff_min, lin_stiff_max=lin_stiff_max,
        ang_stiff_min=ang_stiff_min, ang_stiff_max=ang_stiff_max,
        lin_damping_min=lin_damping_min, lin_damping_max=lin_damping_max,
        ang_damping_min=ang_damping_min, ang_damping_max=ang_damping_max, sparsity_level=sparsity_level)
    unicycle_net.initialize_input_map(1, n_units=n_units, non_zero_min_magnitude=non_zero_values, num_non_zero=num_non_zero)

    if classes is not None:
        n_classes = len(classes)
    else:
        n_classes = 10
    readout_layer = ReadoutLayer(n_units*5, n_classes)
    unicycle_net.mlp_input_map = False
    input_mlp = SmallMLP(784, [64, 32, 16], output_size=n_units)
    optimizer = optim.Adam(readout_layer.parameters(), lr=lr)
    objective_fn = nn.CrossEntropyLoss()

    n_epochs = 1  # Reduce the number of epochs to speed up the process
    # unicycle_net.set_init_states(bs_train)

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
    zero_input = jnp.zeros((1000,1,1))
    _, final_state_eq = unicycle_net.reservoir_forward(zero_input, zero_input, init_state=None, dt=dt,steps=1000)
    unicycle_net.init_state = final_state_eq

    # Train and validate the model
    for epoch in range(n_epochs):
        readout_layer.train()
        progress_bar = tqdm(train_loader)
        for images, labels in progress_bar:
            images = images.reshape(images.shape[0], 1, 784)
            # if not unicycle_net.mlp_input_map:
            images = images.permute(2, 0, 1)
            images = jnp.array(images)
            # else:
            #     images = input_mlp(images)
            angular_input = jnp.zeros(shape=(784, bs_train, 1))

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

    # Calculate validation accuracy
    validation_accuracy = test(valid_loader, unicycle_net, readout_layer, objective_fn, dt, bs_test=bs_test)
    
    # Optuna will aim to maximize this value
    return validation_accuracy

# Evaluation function for validation and testing accuracy
def test(data_loader, unicycle_net, readout_layer, objective_fn, dt, bs_test):
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

    accuracy = 100. * correct / len(data_loader.dataset)
    return accuracy.item()

# Define the search space and start the optimization
if __name__ == '__main__':
    # Define the GPU or CPU device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    # Load your data (train_loader, valid_loader, and test_loader)
    print("jax devices", jax.devices())
    # Create a study to optimize the validation accuracy
    study_name = f"unicycle_opt_5_classes_newest"
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(storage=storage_name, study_name=study_name, direction='maximize', load_if_exists="True")
    # for trial in study.trials:
    #     if trial.datetime_complete and trial.datetime_start:
    #         duration = trial.datetime_complete - trial.datetime_start
    #         print(f"Trial {trial.number} took {duration.total_seconds()} seconds")
    study.optimize(objective, timeout=60*60*8)

    # Get the best hyperparameters
    best_params = study.best_params
    print(f"Best hyperparameters: {best_params}")

    # # Test the model with the best hyperparameters on the test set
    # test_accuracy = test(test_loader)
    # print(f"Test accuracy with the best model: {test_accuracy}")
