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
from utils import get_cifar_data, n_params
from tqdm import tqdm

# Fix matplotlib backend issues - use non-interactive backend
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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
storage_name = f"unicycle_nets_cifar10_logreg"
study_name = "only_last_state_as_readout_200_units_ang_input_ang_coupled"
storage_name = f"sqlite:///{parent_dir}/optuna_databases/{storage_name}.db"
study = optuna.create_study(storage=storage_name, study_name=study_name, direction='maximize', load_if_exists="True")
params = study.best_params
#%%
aligned_orientations = False  # Set to True if you want aligned orientations, False otherwise
ang_input = True  # Set to True if you want angular input, False otherwise
ang_connections = True  # Set to True if you want angular connections, False otherwise
#%%
n_units = 200  # CIFAR-10 uses larger networks
n_inp = 96     # CIFAR-10 input dimension
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
non_zero_fraction = params['non_zero_fraction']
magnitude_min = params['magnitude_min']
magnitude_max = params['magnitude_max']
if ang_input:
    non_zero_fraction_ang = params['non_zero_fraction_ang']
    magnitude_min_ang = params['magnitude_min_ang']
    magnitude_max_ang = params['magnitude_max_ang']
else:
    non_zero_fraction_ang = 0
    magnitude_min_ang = 0
    magnitude_max_ang = 0
n_connections_fraction = params['n_connections_fraction']
n_connections = int(n_units*n_connections_fraction)
washup = params['washup_steps']
n_steps_readout = 0  # params['steps_readout']
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
# Load the CIFAR-10 dataset
train_loader, valid_loader, test_loader = get_cifar_data(bs_train, bs_test, new_fraction=0.4, test_fraction=0.4)
#%%
# CIFAR-10 uses 2D input maps (n_inp x n_units)
lin_input_map = torch.zeros(n_inp, n_units)
total_elements = n_inp * n_units
num_non_zero = int(total_elements * non_zero_fraction)

# Flatten indices and randomly select
flat_indices = torch.randperm(total_elements)[:num_non_zero]
row_indices = flat_indices // n_units
col_indices = flat_indices % n_units

non_zero_values_min = magnitude_min
non_zero_values_max = magnitude_max
random_values = torch.rand(num_non_zero) * (non_zero_values_max - non_zero_values_min) + non_zero_values_min
lin_input_map[row_indices, col_indices] = random_values

#%%
# Randomized ang_input_map for CIFAR-10
ang_input_map = torch.zeros(n_inp, n_units)
if ang_input:
    num_non_zero_ang = int(total_elements * non_zero_fraction_ang)
    
    flat_indices_ang = torch.randperm(total_elements)[:num_non_zero_ang]
    row_indices_ang = flat_indices_ang // n_units
    col_indices_ang = flat_indices_ang % n_units
    
    non_zero_values_ang = magnitude_min_ang
    magnitude_max_ang_val = magnitude_max_ang
    random_values_ang = torch.rand(num_non_zero_ang) * (magnitude_max_ang_val - non_zero_values_ang) + non_zero_values_ang
    ang_input_map[row_indices_ang, col_indices_ang] = random_values_ang

#%%
model = UnicycleReservoir(n_inp=n_inp, n_units=n_units, dt=dt, n_out=10, lin_input_map=lin_input_map, 
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
        for i, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)
            # Process CIFAR-10 images same way as in optuna
            images = torch.cat((images.permute(0,2,1,3).reshape(bs_test,32,96), rand_test),dim=1)
            _, output = model(images, images)
            test_loss += objective(output, labels).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum()
    test_loss /= i+1
    accuracy = 100. * correct / len(data_loader.dataset)

    return accuracy.item()

#%%
@torch.no_grad()
def test_esn(data_loader, model, classifier, scaler, rand_test):
    activations, ys = [], []
    for images, labels in tqdm(data_loader):
        images = images.to(device)
        labels = labels.to(device)
        # Process CIFAR-10 images same way as in optuna
        images = torch.cat((images.permute(0,2,1,3).reshape(bs_test,32,96), rand_test),dim=1)
        states_list, output, mid_states = model(images, images)
        activations.append(mid_states.cpu())
        ys.append(labels.cpu())
    activations = torch.cat(activations, dim=0).numpy()
    activations = scaler.transform(activations)
    ys = torch.cat(ys, dim=0).numpy()
    return classifier.score(activations, ys)
#%%
for i, (images, labels) in enumerate(test_loader):
    print(f"CIFAR-10 data shape: {images.shape}, labels shape: {labels.shape}")
    break
#%%
# n_epochs = n_epochs
objective = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
#%%
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
# CIFAR-10 uses higher dimensional washup
u_lin = torch.zeros((1, washup, n_inp), device=device)
u_ang = torch.zeros_like(u_lin, device=device)

for t in range(u_lin.size()[1]):
    linear_input = (u_lin[:, t]) @ model.lin_input_map
    angular_input = u_ang[:, t] @ model.ang_input_map

    x, z, theta, s, omega = model.unicycle_network(linear_input, angular_input, x, z, theta, s, omega)

    concatenated_states = torch.hstack((x, z, theta, s, omega))
    states_list.append(concatenated_states)
#%%
all_states_time = torch.vstack(states_list)
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(all_states_time[:,0:n_units].cpu().detach().numpy())
plt.title('X states during washup')
plt.xlabel('Time step')
plt.ylabel('State value')

plt.subplot(1, 3, 2)
plt.plot(all_states_time[:,n_units*2:n_units*3].cpu().detach().numpy())
plt.title('Theta states during washup')
plt.xlabel('Time step')
plt.ylabel('Angle (rad)')

plt.subplot(1, 3, 3)
plt.plot(all_states_time[:,n_units*4:n_units*5].cpu().detach().numpy())
plt.title('Omega states during washup')
plt.xlabel('Time step')
plt.ylabel('Angular velocity')

plt.tight_layout()
plot_filename = f"{parent_dir}/plots/cifar10_washup_states.png"
os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
print(f"Washup states plot saved to: {plot_filename}")
plt.close()
#%%
model.set_init_states(bs_train, x,z,theta,s,omega)

# Create random padding for CIFAR-10 (extends 32 timesteps to 1000)
rands = torch.randn(1, 1000 - 32, n_inp).to(device)
rand_train = rands.repeat(bs_train,1,1)
rand_test = rands.repeat(bs_test,1,1)

#%%
# Logistic regression ESN training loop
progress_bar = tqdm(train_loader)
activations, ys = [], []
for images, labels in progress_bar:
    images = images.to(device)
    labels = labels.to(device)
    # Process CIFAR-10: reshape and add random padding
    images = torch.cat((images.permute(0,2,1,3).reshape(bs_train,32,96), rand_train),dim=1)
    
    # model should return mid_states as third output
    states_list, output, mid_states = model(images, images)
    activations.append(mid_states.detach().cpu())
    ys.append(labels.cpu())

activations = torch.cat(activations, dim=0).numpy()
ys = torch.cat(ys, dim=0).numpy()

# Check for NaN values in activations
if np.isnan(activations).any():
    print("NaN values detected in activations")
else:
    print("No NaN values detected, proceeding with training")

scaler = preprocessing.StandardScaler().fit(activations)
activations = scaler.transform(activations)
classifier = LogisticRegression(max_iter=500).fit(activations, ys)

# Validation and test scores using ESN
valid_score = test_esn(valid_loader, model, classifier, scaler, rand_test)
test_score = test_esn(test_loader, model, classifier, scaler, rand_test)
print(f"Validation score (ESN): {valid_score}")
print(f"Test score (ESN): {test_score}")
# %%
# Final validation and test scores
print(f"Number of trainable parameters: {n_params(classifier)}")
#%%
# Analyze a sample batch
sample_batch_images, sample_batch_labels = next(iter(test_loader))
print(f"Sample batch - images: {sample_batch_images.shape}, labels: {sample_batch_labels.shape}")
print(f"Label distribution: {torch.bincount(sample_batch_labels)}")
sample_idx = 0
print(f"Sample label: {sample_batch_labels[sample_idx].item()}")
#%%
# Process sample for reservoir
sample_processed = torch.cat((sample_batch_images.permute(0,2,1,3).reshape(sample_batch_images.shape[0],32,96), 
                             rand_test[:sample_batch_images.shape[0]]),dim=1)
sample_processed = sample_processed.to(device)
sample_batch_labels = sample_batch_labels.to(device)

# Get reservoir states for the sample batch
states_list_sample, output_sample, mid_states_sample = model(sample_processed, sample_processed)
all_states_time_res = torch.stack(states_list_sample, dim=1)

print(f"Reservoir output shape: {all_states_time_res.shape}")
print(f"Mid states shape: {mid_states_sample.shape}")
#%%
all_states_time_res = torch.stack(states_list, dim=1)

#%%
# Plot state evolution for sample
plt.figure(figsize=(15, 10))

# Plot first few units for different state types
n_units_to_plot = min(10, n_units)  # Plot only first 10 units for visibility

plt.subplot(2, 3, 1)
plt.plot(all_states_time_res[sample_idx,:,0:n_units_to_plot].cpu().detach().numpy())
plt.title(f'X states evolution (sample {sample_idx}, first {n_units_to_plot} units)')
plt.xlabel('Time step')
plt.ylabel('Position')

plt.subplot(2, 3, 2)
plt.plot(all_states_time_res[sample_idx,:,1*n_units:1*n_units+n_units_to_plot].cpu().detach().numpy())
plt.title(f'Z states evolution (sample {sample_idx})')
plt.xlabel('Time step')
plt.ylabel('Velocity')

plt.subplot(2, 3, 3)
plt.plot(all_states_time_res[sample_idx,:,2*n_units:2*n_units+n_units_to_plot].cpu().detach().numpy())
plt.title(f'Theta states evolution (sample {sample_idx})')
plt.xlabel('Time step')
plt.ylabel('Angle')

plt.subplot(2, 3, 4)
plt.plot(all_states_time_res[sample_idx,:,3*n_units:3*n_units+n_units_to_plot].cpu().detach().numpy())
plt.title(f'S states evolution (sample {sample_idx})')
plt.xlabel('Time step')
plt.ylabel('Angular position')

plt.subplot(2, 3, 5)
plt.plot(all_states_time_res[sample_idx,:,4*n_units:4*n_units+n_units_to_plot].cpu().detach().numpy())
plt.title(f'Omega states evolution (sample {sample_idx})')
plt.xlabel('Time step')
plt.ylabel('Angular velocity')

plt.subplot(2, 3, 6)
# Show original CIFAR-10 image
original_image = sample_batch_images[sample_idx].permute(1, 2, 0)
# Denormalize if needed (assuming standard CIFAR-10 normalization)
original_image = (original_image + 1) / 2  # Assuming images are normalized to [-1, 1]
original_image = torch.clamp(original_image, 0, 1)
plt.imshow(original_image.cpu().numpy())
plt.title(f'Original CIFAR-10 image\nLabel: {sample_batch_labels[sample_idx].item()}')
plt.axis('off')

plt.tight_layout()
plot_filename = f"{parent_dir}/plots/cifar10_state_evolution_sample_{sample_idx}.png"
plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
print(f"State evolution plot saved to: {plot_filename}")
plt.show()
plt.close()
# %%
# Final state analysis by class
print("Analyzing final states by class...")

# Get final states for each class
final_states_by_class = {}
for class_label in range(10):  # CIFAR-10 has 10 classes
    class_indices = torch.where(sample_batch_labels == class_label)[0].cpu().numpy()
    if len(class_indices) > 0:
        final_states_by_class[class_label] = all_states_time_res[class_indices, -1, :].detach().cpu().numpy()
        print(f"Class {class_label}: {len(class_indices)} samples")

# %%
# Visualize classifier weights
weights = classifier.coef_  # Shape: (10, n_features) for 10-class classification
bias = classifier.intercept_

print(f"Weights shape: {weights.shape}")
print(f"Bias shape: {bias.shape}")

# Let's first understand how activations are organized
print(f"Activations shape: {activations.shape}")
print("First few activation values:", activations[0, :10])

# The features are organized as: [x0, x1, ..., x199, z0, z1, ..., z199, theta0, ..., theta199, s0, ..., s199, omega0, ..., omega199]
n_states_classifier = 5
n_features_per_state = n_units
total_features = n_states_classifier * n_features_per_state

print(f"Expected total features: {total_features}, Actual: {weights.shape[1]}")

# Reshape weights for visualization (10 classes x 5 states x n_units)
weights_reshaped = weights.reshape(10, n_states_classifier, n_units)  # Shape: (10, 5, n_units)

# State names for better visualization
state_names = ['x (position)', 'z (velocity)', 'theta (angle)', 's (angular pos)', 'omega (angular vel)']
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# %%
# 1. Heatmap of weights by class and state
plt.figure(figsize=(16, 12))

# Plot average weights by state for each class
plt.subplot(2, 3, 1)
class_state_means = np.mean(np.abs(weights_reshaped), axis=2)  # Average across units: (10, 5)
im = plt.imshow(class_state_means.T, cmap='viridis', aspect='auto')
plt.colorbar(im, label='Mean |Weight|')
plt.xlabel('Class')
plt.ylabel('State')
plt.title('Average Weight Magnitude by Class and State')
plt.xticks(range(10), range(10))
plt.yticks(range(5), [name.split(' ')[0] for name in state_names])

# %%
# 2. Most important states overall
plt.subplot(2, 3, 2)
overall_state_importance = np.mean(np.mean(np.abs(weights_reshaped), axis=2), axis=0)  # Average across classes and units
bars = plt.bar(range(5), overall_state_importance)
plt.xlabel('State')
plt.ylabel('Mean Absolute Weight')
plt.title('Overall State Importance')
plt.xticks(range(5), [name.split(' ')[0] for name in state_names], rotation=45)

# Color bars by magnitude
for i, bar in enumerate(bars):
    bar.set_color(plt.cm.viridis(overall_state_importance[i] / np.max(overall_state_importance)))

# %%
# 3. Weight distribution across all classes and states
plt.subplot(2, 3, 3)
plt.hist(weights.flatten(), bins=50, alpha=0.7, edgecolor='black')
plt.xlabel('Weight Value')
plt.ylabel('Frequency')
plt.title('Weight Distribution (All Classes)')
plt.grid(True, alpha=0.3)

# %%
# 4. Class-specific weight patterns (show a few representative classes)
plt.subplot(2, 3, 4)
selected_classes = [0, 2, 4, 6, 8]  # Select a few classes for visualization
colors = plt.cm.Set1(np.linspace(0, 1, len(selected_classes)))

for i, class_idx in enumerate(selected_classes):
    class_weights = np.mean(np.abs(weights_reshaped[class_idx]), axis=1)  # Average across units
    plt.plot(class_weights, 'o-', label=f'Class {class_idx} ({class_names[class_idx]})', 
             color=colors[i], linewidth=2, markersize=6)

plt.xlabel('State')
plt.ylabel('Mean |Weight|')
plt.title('Weight Patterns by Class')
plt.xticks(range(5), [name.split(' ')[0] for name in state_names], rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# %%
# 5. Unit importance across all classes and states
plt.subplot(2, 3, 5)
unit_importance = np.mean(np.mean(np.abs(weights_reshaped), axis=1), axis=0)  # Average across classes and states
plt.plot(unit_importance, 'b-', alpha=0.7)
plt.xlabel('Unit Index')
plt.ylabel('Mean |Weight|')
plt.title(f'Unit Importance (n_units={n_units})')
plt.grid(True, alpha=0.3)

# Highlight most important units
top_units = np.argsort(unit_importance)[-10:]  # Top 10 most important units
plt.scatter(top_units, unit_importance[top_units], color='red', s=30, zorder=5, label='Top 10 units')
plt.legend()

# %%
# 6. Inter-class weight differences
plt.subplot(2, 3, 6)
# Calculate pairwise differences between classes (simplified view)
class_signatures = np.mean(np.abs(weights_reshaped), axis=2)  # (10, 5)
class_distances = []
for i in range(10):
    for j in range(i+1, 10):
        dist = np.linalg.norm(class_signatures[i] - class_signatures[j])
        class_distances.append(dist)

plt.hist(class_distances, bins=20, alpha=0.7, edgecolor='black')
plt.xlabel('Euclidean Distance')
plt.ylabel('Frequency')
plt.title('Inter-class Weight Signature Distances')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plot_filename = f"{parent_dir}/plots/cifar10_classifier_weights_analysis.png"
plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
print(f"Classifier weights analysis saved to: {plot_filename}")
plt.close()

# %%
# Print summary statistics
print("\nCIFAR-10 Weight Analysis Summary:")
print("-" * 50)
print(f"Total weights per class: {weights.shape[1]}")
print(f"Total parameters: {weights.size + bias.size}")
print(f"Mean absolute weight: {np.mean(np.abs(weights)):.4f}")
print(f"Max absolute weight: {np.max(np.abs(weights)):.4f}")
print(f"Min absolute weight: {np.min(np.abs(weights)):.4f}")
print(f"Weight standard deviation: {np.std(weights):.4f}")

print(f"\nMost important state overall: {state_names[np.argmax(overall_state_importance)]}")
print(f"Least important state overall: {state_names[np.argmin(overall_state_importance)]}")

# Find most important units
most_important_unit = np.argmax(unit_importance)
print(f"Most important unit: Unit {most_important_unit} (avg abs weight: {unit_importance[most_important_unit]:.4f})")

# Show most distinctive class (highest weight magnitudes)
class_weight_magnitudes = np.mean(np.abs(weights), axis=1)
most_distinctive_class = np.argmax(class_weight_magnitudes)
print(f"Most distinctive class: {most_distinctive_class} ({class_names[most_distinctive_class]}) - avg |weight|: {class_weight_magnitudes[most_distinctive_class]:.4f}")

print(f"\nTest accuracy achieved: {test_score:.4f}")
print(f"Validation accuracy achieved: {valid_score:.4f}")

print("\nTop 5 most important units:")
for i, unit_idx in enumerate(np.argsort(unit_importance)[-5:]):
    print(f"{i+1}. Unit {unit_idx}: {unit_importance[unit_idx]:.4f}")
