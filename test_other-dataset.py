#%%
from utils import get_mnist_data, get_FordA_data
from tqdm import tqdm
import matplotlib.pyplot as plt
from unicycle_network_class import UnicycleNetwork, UnicycleReservoir
from torch import nn, optim
import torch
import optuna
#%%
bs_train, bs_test = 100, 100
train_loader, valid_loader, test_loader = get_FordA_data(bs_train, bs_test)
# # %%
# for images, labels in tqdm(train_loader):
#     break
# # %%
# image0 = images[0]
# #%%
# image0 = image0.reshape(1,28*28)
# images_reshaped = images.reshape(-1, 1, 28*28)
# images_permuted = images_reshaped.permute(0,2,1)
# # %%
# plt.imshow(images[8].squeeze())
# # %%
# labels
# # %%
# unicycle_network = UnicycleNetwork(n_inp=3, n_units=3, dt=0.01)
# #%%
# bs_train=1
x = torch.randn(bs_train, 3)
z = torch.randn(bs_train, 3)
theta = torch.randn(bs_train, 3)
s = torch.randn(bs_train, 3)
omega = torch.randn(bs_train, 3)
#%%
t_steps = 784
u_t_linear = torch.randn(3, t_steps)*0
u_t_ang = torch.randn(3, t_steps)*0
#%%
x_history = torch.empty((bs_train,0))
z_history = torch.empty((bs_train,0))
theta_history = torch.empty((bs_train,0))
s_history = torch.empty((bs_train,0))
omega_history = torch.empty((bs_train,0))
# %%
# for t in range(t_steps):
#     x,z,theta,s,omega = unicycle_network(u_t_linear[:, t], u_t_ang[:,t], x, z, theta, s, omega)
#     x_history = torch.hstack((x_history, x))
#     z_history = torch.hstack((z_history, z))
#     theta_history = torch.hstack((theta_history, theta))
#     s_history = torch.hstack((s_history, s))
#     omega_history = torch.hstack((omega_history, omega))
#%%
# plt.plot(omega_history[0][0::3])
#%%
# theta_expanded_1 = theta[:, :, None]   # shape (b, n_units, 1)
# theta_expanded_2 = theta[:, None, :]   # shape (b, 1, n_units)
# ang_distances = theta_expanded_1 - theta_expanded_2
# coupling_term_ang = torch.sum(unicycle_network.dist_ang_coupling[None, :, :] * ang_distances, dim=2, keepdim=False)  # shape (b, n_units, 1)
#%%
# theta_expanded_1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
# Randomized lin_input_map
n_units = 50
lin_input_map = torch.zeros(1, n_units)
num_non_zero = 40
non_zero_indices = torch.randperm(n_units)[:num_non_zero]  # Randomly select indices
non_zero_values_min = 0.1
non_zero_values_max = 0.5
lin_input_map[0, non_zero_indices] = torch.rand(num_non_zero) * (non_zero_values_max- non_zero_values_min) + non_zero_values_min  # Random magnitudes
#%%
# lin_input_map = torch.zeros(1, n_units)
# lin_input_map[0,0] = 10.0
model = UnicycleReservoir(n_inp=1, n_units=n_units, dt=0.01, n_out=2, lin_input_map=lin_input_map, 
                          lin_stiff_min=0.1, lin_damping_min=1., lin_damping_max=2.0, lin_stiff_max=0.2, n_connections=10, n_past_steps_readout=10)
#%%
model = model.to(device)
#%%
def test(data_loader):
    model.eval()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for i, (x, labels) in enumerate(data_loader):
            # images, labels = images.to(device), labels.to(device)
            x = x.to(device)*0.1
            angular_input = torch.zeros_like(x)
            angular_input = angular_input.to(device)

            labels = labels.to(device)
            labels = labels.squeeze()
            labels = labels.long()

            _, output = model(x, angular_input)
            test_loss += objective(output, labels).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum()
    test_loss /= i+1
    accuracy = 100. * correct / len(data_loader.dataset)

    return accuracy.item()
#%%
n_epochs = 3
objective = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)
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
model.s_init[:,0] = 0
# #%%
# valid_score = test(valid_loader)
# test_score = test(test_loader)
# print(f"Validation score: {valid_score}")
# print(f"Test score: {test_score}")
#%%
x = model.x_init[0:1,:]
z = model.z_init[0:1,:]
theta = model.theta_init[0:1,:]
s = model.s_init[0:1,:]
omega = model.omega_init[0:1,:]
states_list = []

u_lin = torch.zeros((1, 4000, 1), device=device)
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
plt.plot(all_states_time[:,n_units*3:n_units*4].cpu().detach().numpy())
plt.show()
#%%
model.set_init_states(bs_train, x,z,theta,s,omega)
#%%
for epoch in range(n_epochs):
    model.train()
    progress_bar = tqdm(train_loader)
    for x, labels in progress_bar:

        x = x.to(device)*0.1
        angular_input = torch.zeros_like(x)
        angular_input = angular_input.to(device)
        labels = labels.to(device)
        labels = labels.squeeze()
        labels = labels.long()

        optimizer.zero_grad()
        # with torch.no_grad():
        states_list, output = model(x, angular_input)
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
#%%
#%%
sample_idx = 3
print(labels[sample_idx])
all_states_time_res = torch.stack(states_list, dim=1)
plt.plot(all_states_time_res[sample_idx,:,0:n_units].cpu().detach().numpy())
plt.show()
plt.plot(all_states_time_res[sample_idx,:,n_units*0:n_units*1].cpu().detach().numpy())
plt.show()
# %%
states_list[-1][0]
# %%
output.shape
# %%
model.readout(states_list[-1])
# %%
states_list[0][0,3:6]
# %%
study_name = f"unicycle_forda_w_ang_input"
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(storage=storage_name, study_name=study_name, direction='maximize', load_if_exists="True")
params = study.best_params
#%%
n_units = params['n_units']
lr = params['lr']
lin_stiff_min =  params['lin_stiff_min']
lin_stiff_max =  params['lin_stiff_max']
ang_stiff_min =  params['ang_stiff_min']
ang_stiff_max =  params['ang_stiff_max']
lin_damping_min =  params['lin_damping_min']
lin_dmping_max =  params['lin_damping_max']
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
non_zero_elements_ang = params['non_zero_elements_ang']
magnitude_min_ang = params['magnitude_min_ang']
magnitude_max_ang = params['magnitude_max_ang']
n_connections = params['n_connections']
washup = 0#params['washup_steps']
n_steps_readout = params['steps_readout']
n_connections_ang = params['n_connections_ang']
anchor_con_fraction_ang = params['anchor_con_fraction_ang']
eq_dist_min = params['eq_dist_min']
eq_dist_max = params['eq_dist_max']
eq_dist_min_ang = params['eq_dist_min_ang']
eq_dist_max_ang = params['eq_dist_max_ang']
n_epochs = params['n_epochs']
n_connections_anchor = int(n_connections * anchor_con_fraction)
n_connections_anchor_ang = int(n_connections_ang * anchor_con_fraction_ang)
#%%
train_loader, valid_loader, test_loader = get_FordA_data(bs_train, bs_test)
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
model = UnicycleReservoir(n_inp=1, n_units=n_units, dt=dt, n_out=2, lin_input_map=lin_input_map, 
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
        for i, (x, labels) in enumerate(data_loader):
            # images, labels = images.to(device), labels.to(device)
            x = x.to(device)
            angular_input = torch.zeros_like(x)
            angular_input = angular_input.to(device)
            labels = labels.to(device)
            labels = labels.squeeze()
            labels = labels.long()

            _, output = model(x, x)
            test_loss += objective(output, labels).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum()
        test_loss /= i+1
        total_samples = sum(len(batch[1]) for batch in data_loader)
        accuracy = 100. * correct / total_samples

    return accuracy.item()
#%%
# for i, (images, labels) in enumerate(test_loader):
#     images = images.reshape(bs_test, 1, 784)
#%%
n_epochs = n_epochs
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
model.omega_init[:,0] = 0
# #%%
# valid_score = test(valid_loader)
# test_score = test(test_loader)
# print(f"Validation score: {valid_score}")
# print(f"Test score: {test_score}")
#%%
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
#%%
all_states_time = torch.stack(states_list, dim=1)
#%%
plt.plot(all_states_time[0,:,0:n_units].cpu().detach().numpy())
plt.show()
plt.plot(all_states_time[0,:,n_units*3:n_units*4].cpu().detach().numpy())
plt.show()
#%%
model.set_init_states(bs_train, x,z,theta,s,omega)
# perm = torch.randperm(784).to(device)

#%%
for epoch in range(n_epochs):
    model.train()
    progress_bar = tqdm(train_loader)
    for x, labels in progress_bar:
        x = x.to(device)
        angular_input = torch.zeros_like(x)
        angular_input = angular_input.to(device)
        labels = labels.to(device)
        labels = labels.squeeze()
        labels = labels.long()

        optimizer.zero_grad()
        # with torch.no_grad():
        states_list, output = model(x, x)
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
