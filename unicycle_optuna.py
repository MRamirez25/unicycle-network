import torch
from torch import nn, optim
from tqdm import tqdm
import optuna
from unicycle_network_class import UnicycleReservoir
from utils import get_mnist_data
import time

# Objective function for Optuna
def objective(trial):
    # Suggest hyperparameters for Optuna to search
    n_units = trial.suggest_int('n_units', 10, 200, step=10)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    lin_stiff_min = trial.suggest_float('lin_stiff_min', 0.1, 1.0)
    lin_stiff_max = trial.suggest_float('lin_stiff_max', lin_stiff_min, 5.0)  # lin_stiff_max >= lin_stiff_min
    ang_stiff_min = trial.suggest_float('ang_stiff_min', 0., 1.0)
    ang_stiff_max = trial.suggest_float('ang_stiff_max', ang_stiff_min, 2.0)  # ang_stiff_max >= ang_stiff_min
    lin_damping_min = trial.suggest_float('lin_damping_min', 0.1, 1.0)
    lin_damping_max = trial.suggest_float('lin_damping_max', lin_damping_min, 5.0)
    ang_damping_min = trial.suggest_float('ang_damping_min', 0.1, 1.0)
    ang_damping_max = trial.suggest_float('ang_damping_max', ang_damping_min, 2.0)
    bs = trial.suggest_int("batch_size", 50, 300, step=50)
    dt = trial.suggest_float("dt", 0.0001, 0.01, step=0.0005)
    inp_bias = trial.suggest_float("inp_bias", -1,1)
    n_connections_anchor_fraction = trial.suggest_float("anchor_con_fraction", 0, 1.0, step=0.1)

    # Randomized lin_input_map
    lin_input_map = torch.zeros(1, n_units)
    num_non_zero = trial.suggest_int("non_zero_elements", 1, n_units)
    non_zero_indices = torch.randperm(n_units)[:num_non_zero]  # Randomly select indices
    non_zero_values = trial.suggest_float("magnitude_min", 1.0, 10.0)
    magnitude_max = trial.suggest_float("magnitude_max", non_zero_values, 20)
    lin_input_map[0, non_zero_indices] = torch.rand(num_non_zero) * (magnitude_max- non_zero_values) + non_zero_values  # Random magnitudes

    # Randomized lin_input_map
    ang_input_map = torch.zeros(1, n_units)
    num_non_zero_ang = trial.suggest_int("non_zero_elements_ang", 1, n_units)
    non_zero_indices = torch.randperm(n_units)[:num_non_zero_ang]  # Randomly select indices
    non_zero_values_ang = trial.suggest_float("magnitude_min_ang", 0.1, 10.0)
    magnitude_max_ang = trial.suggest_float("magnitude_max_ang", non_zero_values_ang, 20)
    ang_input_map[0, non_zero_indices] = torch.rand(num_non_zero_ang) * (magnitude_max_ang- non_zero_values_ang) + non_zero_values_ang  # Random magnitudes

    n_connections = trial.suggest_int('n_connections', 3, int(n_units*0.7), step=5)
    washup = trial.suggest_int('washup_steps', 0, 4000, step=1000)
    n_connections_anchor = int(n_connections_anchor_fraction * n_connections)
    n_steps_readout = trial.suggest_int("steps_readout", 0, 100, step=10)
    n_connections_ang = trial.suggest_int("n_connections_ang", 2, int(n_units*0.7), step=5)
    n_connections_anchor_fraction_ang = trial.suggest_float("anchor_con_fraction_ang", 0, 1.0, step=0.1)
    n_connections_anchor_ang = int(n_connections_anchor_fraction_ang * n_connections)

    eq_dist_min = trial.suggest_float("eq_dist_min", 0.2,1.0)
    eq_dist_max = trial.suggest_float("eq_dist_max", eq_dist_min, 2.0)
    eq_dist_min_ang = trial.suggest_float("eq_dist_min_ang", 0.0, 3.14)
    eq_dist_max_ang = trial.suggest_float("eq_dist_max_ang", eq_dist_min_ang, 6.28)

    # Initialize the model with the suggested hyperparameters
    classes = [0,1,2,3,4,5,6,7,8,9]
    model = UnicycleReservoir(
        n_inp=1, n_units=n_units, dt=dt, n_out=len(classes),
        lin_stiff_min=lin_stiff_min, lin_stiff_max=lin_stiff_max,
        ang_stiff_min=ang_stiff_min, ang_stiff_max=ang_stiff_max,
        lin_damping_min=lin_damping_min, lin_damping_max=lin_damping_max,
        ang_damping_min=ang_damping_min, ang_damping_max=ang_damping_max,
        eq_dist_min=eq_dist_min, eq_dist_max=eq_dist_max, 
        eq_dist_min_ang=eq_dist_min_ang, eq_dist_max_ang=eq_dist_max_ang,
    n_connections=n_connections, inp_bias=inp_bias, lin_input_map=lin_input_map, 
    n_connections_anchor=n_connections_anchor, ang_input_map=ang_input_map,
    n_connections_ang=n_connections_ang, n_connections_anchor_ang=n_connections_anchor_ang, 
    n_past_steps_readout=n_steps_readout).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    objective_fn = nn.CrossEntropyLoss()
    bs_train, bs_test = bs, bs
    train_loader, valid_loader, test_loader = get_mnist_data(bs_train=bs_train, bs_test=bs_test, classes=classes, new_fraction=0.2, test_fraction=0.2)
    n_epochs = trial.suggest_int("n_epochs", 1,10)
    model.set_init_states_random(bs_train)

    # Move initial states to GPU if available
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
    model.omega_init[:,0] = 0

    x = model.x_init[0:1,:]
    z = model.z_init[0:1,:]
    theta = model.theta_init[0:1,:]
    s = model.s_init[0:1,:]
    omega = model.omega_init[0:1,:]
    states_list = []

    u_lin = torch.zeros((1, washup, 1), device=device)
    u_ang = torch.zeros_like(u_lin, device=device)

    for t in range(u_lin.size()[1]):
        linear_input = (u_lin[:, t]) @ model.lin_input_map
        angular_input = u_ang[:, t] @ model.ang_input_map

        x, z, theta, s, omega = model.unicycle_network(linear_input, angular_input, x, z, theta, s, omega)

        concatenated_states = torch.hstack((x, z, theta, s, omega))
        states_list.append(concatenated_states)
    model.set_init_states(bs_train, x,z,theta,s,omega)

    # Train and validate the model
    for epoch in range(n_epochs):
        model.train()
        progress_bar = tqdm(train_loader)
        for images, labels in progress_bar:
            images = images.reshape(images.shape[0], 1, 784)
            images = images.permute(0, 2, 1)
            # angular_input = torch.zeros_like(images)
            images = images.to(device)
            # angular_input = angular_input.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            # start = time.time()
            states_list, output = model(images, images)
            # end = time.time()
            # print(f"dynamics time {end - start}")
            loss = objective_fn(output, labels)
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())
            if torch.isnan(loss):
                break

        # Calculate validation accuracy
        if epoch == 0:
            validation_accuracy = test(valid_loader, model, objective_fn, bs_test)
            if validation_accuracy < 50:
                break
    else:
        validation_accuracy = test(valid_loader, model, objective_fn, bs_test)
    
    # Optuna will aim to maximize this value
    return validation_accuracy

# Evaluation function for validation and testing accuracy
def test(data_loader, model, objective_fn, bs_test):
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images = images.reshape(bs_test, 1, 784)
            images = images.permute(0, 2, 1)
            # angular_input = torch.zeros_like(images)
            images = images.to(device)
            # angular_input = angular_input.to(device)
            labels = labels.to(device)

            _, output = model(images, images)
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

    # Create a study to optimize the validation accuracy
    study_name = "unicycle_opt_all_classes_w_ang_input"
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(storage=storage_name, study_name=study_name, direction='maximize', load_if_exists="True")
    # for trial in study.trials:
    #     if trial.datetime_complete and trial.datetime_start:
    #         duration = trial.datetime_complete - trial.datetime_start
    #         print(f"Trial {trial.number} took {duration.total_seconds()} seconds")
    study.optimize(objective)

    # Get the best hyperparameters
    best_params = study.best_params
    print(f"Best hyperparameters: {best_params}")

    # # Test the model with the best hyperparameters on the test set
    # test_accuracy = test(test_loader)
    # print(f"Test accuracy with the best model: {test_accuracy}")
