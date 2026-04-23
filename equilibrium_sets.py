#%%
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import torch
from unicycle_network import UnicycleNetwork
import torch
from equilibrium_simple_case import solve_mass_spring_system, plot_mass_spring_system
#%%
#%%
# Fixed mass positions
fixed_points = {0: (0, 0)}  # Mass 1 is fixed

# Moving mass initial positions
moving_points = {1: (0.0, 1.0), 2: (1.0, 1.0),}  # Mass 2 & 3 are moving

# Spring connections (mass_i, mass_j, stiffness, rest_length)
springs = [
    (0, 1, 1.0, 0.6),  # Spring between Mass 2 and fixed Mass 1
    (0, 2, 1.0, 0.8),  # Spring between Mass 3 and fixed Mass 1
    (1, 2, 1.0, 0.7),
]

# Constraint angles (radians)
alphas = {1: np.radians(30), 2: np.radians(45)}

# Solve system
final_positions = solve_mass_spring_system(fixed_points, moving_points, springs, alphas)

# Print final positions
for mass, (x, y) in final_positions.items():
    print(f"Mass {mass}: Final Position = ({x:.4f}, {y:.4f})")
# %%
alphas_set = [0,45,90,135]
final_positions_1 = []
final_positions_2 = []
# %%
for alpha1 in alphas_set:
    for alpha2 in alphas_set:
        alphas = {1: np.radians(alpha1), 2: np.radians(alpha2)}
        final_positions = solve_mass_spring_system(fixed_points, moving_points, springs, alphas)
        final_positions_1.append(final_positions[1])
        final_positions_2.append(final_positions[2])
        plot_mass_spring_system(fixed_points, moving_points, springs, final_positions)
# %%
plt.figure(figsize=(6, 6))

# Plot initial configuration (blue dashed lines)
for i, j, _, _ in springs:
    x_i, y_i = fixed_points[i] if i in fixed_points else moving_points[i]
    x_j, y_j = fixed_points[j] if j in fixed_points else moving_points[j]
    plt.plot([x_i, x_j], [y_i, y_j], 'bo--', alpha=0.6, label="Initial" if i == 0 and j == 1 else "")

for n in range(len(final_positions_1)):
    plt.scatter(final_positions_1[n][0], final_positions_1[n][1], label='1\'', color='g')
    plt.scatter(final_positions_2[n][0], final_positions_2[n][1], label='2\'', color='orange')
# %%
