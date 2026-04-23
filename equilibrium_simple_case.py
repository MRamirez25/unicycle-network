#%%
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import torch
from unicycle_network import UnicycleNetwork
import torch
#%%
# Given constants
L = 1.0   # Initial side length of the equilateral triangle
a12 = 1.0  # Rest length of spring between Mass 1 and Mass 2
a13 = 0.75 # Rest length of spring between Mass 1 and Mass 3
a23 = 0.5  # Rest length of spring between Mass 2 and Mass 3

# Fixed position of mass 1
x1, y1 = 0, 0

# Random movement constraint angles (in radians)
alpha_2 = np.radians(30)   # Example: Mass 2 moves at 30 degrees
alpha_3 = np.radians(-45)  # Example: Mass 3 moves at -45 degrees

# Updated position equations with constraints along alpha directions
def x2(u2): return L + u2 * np.cos(alpha_2)
def y2(u2): return u2 * np.sin(alpha_2)
def x3(u3): return L/2 + u3 * np.cos(alpha_3)
def y3(u3): return np.sqrt(3)*L/2 + u3 * np.sin(alpha_3)

# Updated distance functions
def d12(u2):
    return np.sqrt((x2(u2) - x1)**2 + (y2(u2) - y1)**2)

def d13(u3):
    return np.sqrt((x3(u3) - x1)**2 + (y3(u3) - y1)**2)

def d23(u2, u3):
    return np.sqrt((x3(u3) - x2(u2))**2 + (y3(u3) - y2(u2))**2)

# System of equations with random alphas
def equations_general(vars):
    u2, u3 = vars

    # Force balance for mass 2 (projected along alpha_2)
    eq1 = (a12 - d12(u2)) * np.cos(alpha_2) + (a23 - d23(u2, u3)) * ((x2(u2) - x3(u3)) / d23(u2, u3)) * np.cos(alpha_2)

    # Force balance for mass 3 (projected along alpha_3)
    eq2 = (a13 - d13(u3)) * np.sin(alpha_3) + (a23 - d23(u2, u3)) * ((y3(u3) - y2(u2)) / d23(u2, u3)) * np.sin(alpha_3)

    return [eq1, eq2]

# Initial guess for u2 and u3
initial_guess = [0, 0]

# Solve the system
solution_general = fsolve(equations_general, initial_guess)
u2_solution_general, u3_solution_general = solution_general

print(f"Numerical solution: u2 = {u2_solution_general}, u3 = {u3_solution_general}")

# New positions after displacement
x2_new, y2_new = x2(u2_solution_general), y2(u2_solution_general)
x3_new, y3_new = x3(u3_solution_general), y3(u3_solution_general)

# Plot original and deformed triangle
plt.figure(figsize=(6, 6))
plt.plot([x1, L, L/2, x1], [y1, 0, np.sqrt(3)*L/2, y1], 'bo--', label="Original Triangle")
plt.plot([x1, x2_new, x3_new, x1], [y1, y2_new, y3_new, y1], 'ro-', label="Deformed Triangle")

# Annotate points
plt.text(x1, y1, " 1 (fixed)", fontsize=12, verticalalignment='bottom', color='blue')
plt.text(L, 0, " 2", fontsize=12, verticalalignment='bottom', color='blue')
plt.text(L/2, np.sqrt(3)*L/2, " 3", fontsize=12, verticalalignment='bottom', color='blue')

plt.text(x2_new, y2_new, " 2'", fontsize=12, verticalalignment='bottom', color='red')
plt.text(x3_new, y3_new, " 3'", fontsize=12, verticalalignment='bottom', color='red')

# Labels and legend
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.legend()
plt.title("Deformation of the Triangle with Motion Constraints")
plt.grid()
plt.show()
#%%
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# Fixed position of mass 1
x1, y1 = 0, 0

# User-defined initial positions for masses 2 and 3
x2_init, y2_init = 0.5, 1.0   # Adjust as needed
x3_init, y3_init = 1.0, 0.0   # Adjust as needed

# Rest lengths of the springs
a12 = 1.0   # Rest length between Mass 1 and Mass 2
a13 = 0.75  # Rest length between Mass 1 and Mass 3
a23 = 0.5   # Rest length between Mass 2 and Mass 3

# Spring stiffness values
k12 = 1.0   # Stiffness of spring between Mass 1 and Mass 2
k13 = 1.0   # Stiffness of spring between Mass 1 and Mass 3
k23 = 1.0   # Stiffness of spring between Mass 2 and Mass 3

# Angles of movement constraints (in radians)
alpha_2 = np.radians(30)   # Mass 2 moves at 30 degrees
alpha_3 = np.radians(45)  # Mass 3 moves at -45 degrees

# Position functions constrained to motion along axes
def x2(u2): return x2_init + u2 * np.cos(alpha_2)
def y2(u2): return y2_init + u2 * np.sin(alpha_2)
def x3(u3): return x3_init + u3 * np.cos(alpha_3)
def y3(u3): return y3_init + u3 * np.sin(alpha_3)

# Distance functions
def d12(u2):
    return np.sqrt((x2(u2) - x1)**2 + (y2(u2) - y1)**2)

def d13(u3):
    return np.sqrt((x3(u3) - x1)**2 + (y3(u3) - y1)**2)

def d23(u2, u3):
    return np.sqrt((x3(u3) - x2(u2))**2 + (y3(u3) - y2(u2))**2)

# System of equations with movement constraints
def equations(vars):
    u2, u3 = vars

    # Compute current distances
    d12_val = d12(u2)
    d13_val = d13(u3)
    d23_val = d23(u2, u3)

    # Compute force components from spring 12
    F12_x = k12 * (a12 - d12_val) * (x1 - x2(u2)) / d12_val
    F12_y = k12 * (a12 - d12_val) * (y1 - y2(u2)) / d12_val

    # Compute force components from spring 13
    F13_x = k13 * (a13 - d13_val) * (x1 - x3(u3)) / d13_val
    F13_y = k13 * (a13 - d13_val) * (y1 - y3(u3)) / d13_val

    # Compute force components from spring 23
    F23_x = k23 * (a23 - d23_val) * (x3(u3) - x2(u2)) / d23_val
    F23_y = k23 * (a23 - d23_val) * (y3(u3) - y2(u2)) / d23_val

    # Force balance for mass 2 (projected along alpha_2)
    eq1 = (F12_x + F23_x) * np.cos(alpha_2) + (F12_y + F23_y) * np.sin(alpha_2)

    # Force balance for mass 3 (projected along alpha_3)
    eq2 = (F13_x - F23_x) * np.cos(alpha_3) + (F13_y - F23_y) * np.sin(alpha_3)

    return [eq1, eq2]

# Initial guess for u2 and u3
initial_guess = [0, 0]

# Solve the system
solution = fsolve(equations, initial_guess)
u2_solution, u3_solution = solution

# Compute final positions
x2_solution, y2_solution = x2(u2_solution), y2(u2_solution)
x3_solution, y3_solution = x3(u3_solution), y3(u3_solution)

print(f"Numerical solution: u2 = {u2_solution}, u3 = {u3_solution}")
print(f"Final positions: x2 = {x2_solution}, y2 = {y2_solution}, x3 = {x3_solution}, y3 = {y3_solution}")

# Plot original and deformed triangle
plt.figure(figsize=(6, 6))
plt.plot([x1, x2_init, x3_init, x1], [y1, y2_init, y3_init, y1], 'bo--', label="Initial Configuration")
plt.plot([x1, x2_solution, x3_solution, x1], [y1, y2_solution, y3_solution, y1], 'ro-', label="Deformed Configuration")

# Annotate points
plt.text(x1, y1, " 1 (fixed)", fontsize=12, verticalalignment='bottom', color='blue')
plt.text(x2_init, y2_init, " 2", fontsize=12, verticalalignment='bottom', color='blue')
plt.text(x3_init, y3_init, " 3", fontsize=12, verticalalignment='bottom', color='blue')

plt.text(x2_solution, y2_solution, " 2'", fontsize=12, verticalalignment='bottom', color='red')
plt.text(x3_solution, y3_solution, " 3'", fontsize=12, verticalalignment='bottom', color='red')

# Labels and legend
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.legend()
plt.title("Triangle Deformation with Constrained Motion Axes & Custom Stiffnesses")
plt.grid()
plt.show()
#%%
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

def solve_mass_spring_system(fixed_points, moving_points, springs, alphas, initial_guess=None):
    """
    Solves a system of N masses constrained to move along given angles with springs.

    Parameters:
    - fixed_points: Dict {mass_index: (x, y)} for fixed masses.
    - moving_points: Dict {mass_index: (x_init, y_init)} for moving masses.
    - springs: List of (mass_i, mass_j, stiffness, rest_length).
    - alphas: Dict {mass_index: alpha} for movement angles.
    - initial_guess: Initial displacement guess for fsolve (default is zeros).

    Returns:
    - Final positions of moving masses.
    """

    moving_indices = list(moving_points.keys())
    fixed_indices = list(fixed_points.keys())
    num_moving = len(moving_indices)

    # Convert input to numpy arrays
    moving_points = {i: np.array(pos) for i, pos in moving_points.items()}
    fixed_points = {i: np.array(pos) for i, pos in fixed_points.items()}

    # Initial guess for displacements (default to zero)
    if initial_guess is None:
        initial_guess = np.zeros(num_moving)

    # Position functions constrained to motion along axes
    def x(i, u): return moving_points[i][0] + u * np.cos(alphas[i])
    def y(i, u): return moving_points[i][1] + u * np.sin(alphas[i])

    # Function to compute distance between two masses
    def distance(i, j, u_vars):
        # Determine positions of mass i
        if i in moving_points:
            x_i, y_i = x(i, u_vars[moving_indices.index(i)]), y(i, u_vars[moving_indices.index(i)])
        else:
            x_i, y_i = fixed_points[i]

        # Determine positions of mass j
        if j in moving_points:
            x_j, y_j = x(j, u_vars[moving_indices.index(j)]), y(j, u_vars[moving_indices.index(j)])
        else:
            x_j, y_j = fixed_points[j]

        return np.sqrt((x_j - x_i) ** 2 + (y_j - y_i) ** 2)

    # Define system of equations
    def equations(u_vars):
        eqs = np.zeros(num_moving)
        
        for i, mass_index in enumerate(moving_indices):  # Only compute equations for moving masses
            Fx_total, Fy_total = 0, 0  # Net force components
            
            for mi, mj, k, a in springs:
                # Ensure correct indices
                d = distance(mi, mj, u_vars)

                # Get positions
                if mi in moving_points:
                    x_i, y_i = x(mi, u_vars[moving_indices.index(mi)]), y(mi, u_vars[moving_indices.index(mi)])
                else:
                    x_i, y_i = fixed_points[mi]

                if mj in moving_points:
                    x_j, y_j = x(mj, u_vars[moving_indices.index(mj)]), y(mj, u_vars[moving_indices.index(mj)])
                else:
                    x_j, y_j = fixed_points[mj]

                # Compute force components
                F_x = k * (a - d) * (x_j - x_i) / d
                F_y = k * (a - d) * (y_j - y_i) / d

                # Apply Newton’s Third Law
                if mi == mass_index:
                    Fx_total += F_x
                    Fy_total += F_y
                elif mj == mass_index:
                    Fx_total -= F_x
                    Fy_total -= F_y

            # Project force onto movement direction
            eqs[i] = Fx_total * np.cos(alphas[mass_index]) + Fy_total * np.sin(alphas[mass_index])

        return eqs

    # Solve system
    solution = fsolve(equations, initial_guess)
    
    # Compute final positions
    final_positions = {mass_index: (x(mass_index, solution[i]), y(mass_index, solution[i]))
                       for i, mass_index in enumerate(moving_indices)}
    
    return final_positions

#%%
# Fixed mass positions
fixed_points = {0: (0, 0)}  # Mass 1 is fixed

# Moving mass initial positions
moving_points = {1: (0.0, 1.0), 2: (1.0, 1.0), 3: (1.0, 0.0)}  # Mass 2 & 3 are moving

# Spring connections (mass_i, mass_j, stiffness, rest_length)
springs = [
    (0, 1, 1.0, 0.6),  # Spring between Mass 2 and fixed Mass 1
    (0, 3, 1.0, 0.8),  # Spring between Mass 3 and fixed Mass 1
    (1, 2, 1.0, 0.7),
    (2, 3, 1.0, 0.5)   # Spring between Mass 2 and Mass 3
]

# Constraint angles (radians)
alphas = {1: np.radians(30), 2: np.radians(45), 3: np.radians(-15)}

# Solve system
final_positions = solve_mass_spring_system(fixed_points, moving_points, springs, alphas)

# Print final positions
for mass, (x, y) in final_positions.items():
    print(f"Mass {mass}: Final Position = ({x:.4f}, {y:.4f})")
#%%
def plot_mass_spring_system(fixed_points, moving_points, springs, final_positions):
    """
    Plots the initial and deformed configurations of a mass-spring system.

    Parameters:
    - fixed_points: Dict {mass_index: (x, y)} for fixed masses.
    - moving_points: Dict {mass_index: (x_init, y_init)} for moving masses.
    - springs: List of (mass_i, mass_j, stiffness, rest_length).
    - final_positions: Dict {mass_index: (x_final, y_final)} for moving masses.
    """

    plt.figure(figsize=(6, 6))

    # Plot initial configuration (blue dashed lines)
    for i, j, _, _ in springs:
        x_i, y_i = fixed_points[i] if i in fixed_points else moving_points[i]
        x_j, y_j = fixed_points[j] if j in fixed_points else moving_points[j]
        plt.plot([x_i, x_j], [y_i, y_j], 'bo--', alpha=0.6, label="Initial" if i == 0 and j == 1 else "")

    # Plot deformed configuration (red solid lines)
    for i, j, _, _ in springs:
        x_i, y_i = fixed_points[i] if i in fixed_points else final_positions[i]
        x_j, y_j = fixed_points[j] if j in fixed_points else final_positions[j]
        plt.plot([x_i, x_j], [y_i, y_j], 'ro-', alpha=0.8, label="Deformed" if i == 0 and j == 1 else "")

    # Annotate fixed points (blue)
    for mass, (x, y) in fixed_points.items():
        plt.scatter(x, y, color='blue', zorder=3)
        plt.text(x, y, f" {mass} (fixed)", fontsize=12, verticalalignment='bottom', color='blue')

    # Annotate initial moving points (blue)
    for mass, (x, y) in moving_points.items():
        plt.scatter(x, y, color='blue', zorder=3)
        plt.text(x, y, f" {mass}", fontsize=12, verticalalignment='bottom', color='blue')

    # Annotate final moving points (red)
    for mass, (x, y) in final_positions.items():
        plt.scatter(x, y, color='red', zorder=3)
        plt.text(x, y, f" {mass}'", fontsize=12, verticalalignment='bottom', color='red')

    # Labels and legend
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.legend()
    plt.title("Mass-Spring System: Initial vs. Deformed Configuration")
    plt.grid()
    plt.show()

#%%
plot_mass_spring_system(fixed_points, moving_points, springs, final_positions)
#%%
n_units = 4
dt = 0.01
#%%
unicycle_network = UnicycleNetwork(n_inp=1, n_units=n_units, dt=dt, lin_stiff_min=0.1, lin_stiff_max=0.5, 
                 ang_stiff_min=0.1, ang_stiff_max=0.3, lin_damping_min=1.0, lin_damping_max=1.0,
                 ang_damping_min=0.1, ang_damping_max=0.2, eq_dist_min=0.5, eq_dist_max=1.0, eq_dist_min_ang=0.0,
                 eq_dist_max_ang=np.pi,
                 lin_input_map=None, ang_input_map=None, n_connections=3, n_connections_anchor=2,
                 n_connections_ang=0, n_connections_anchor_ang=2)
#%%
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = unicycle_network.to(device)
#%%
stiffness_coupling_matrix = torch.ones((n_units, n_units))
stiffness_coupling_matrix.fill_diagonal_(0)
stiffness_coupling_matrix[0,2], stiffness_coupling_matrix[2,0] = 0, 0
stiffness_coupling_matrix[1,3], stiffness_coupling_matrix[3,1] = 0, 0 
#%%
unicycle_network.stiffness_coupling_matrix = torch.nn.Parameter((stiffness_coupling_matrix).to(device), requires_grad=False)
#%%
unicycle_network.eq_distances_matrix[0,1,0], unicycle_network.eq_distances_matrix[1,0,0] = 0.6, 0.6
unicycle_network.eq_distances_matrix[0,3,0], unicycle_network.eq_distances_matrix[3,0,0] = 0.8, 0.8
unicycle_network.eq_distances_matrix[1,2,0], unicycle_network.eq_distances_matrix[2,1,0] = 0.7, 0.7
unicycle_network.eq_distances_matrix[3,2,0], unicycle_network.eq_distances_matrix[2,3,0] = 0.5, 0.5

unicycle_network.lin_damping = unicycle_network.lin_damping.to(device)
unicycle_network.ang_damping = unicycle_network.ang_damping.to(device)
unicycle_network.mass_vector = unicycle_network.mass_vector.to(device)
unicycle_network.j_vector = unicycle_network.j_vector.to(device)
#%%
x = torch.tensor([[0,moving_points[1][0],moving_points[2][0], moving_points[3][0]]]).to(device).double()
z = torch.tensor([[0,moving_points[1][1],moving_points[2][1], moving_points[3][1]]]).to(device).double()
theta = torch.tensor([[0, alphas[1], alphas[2], alphas[3]]]).to(device).double()
omega = torch.zeros((1,n_units)).to(device)
s = torch.zeros((1,n_units)).to(device)
#%%
states_list = []
u_lin = torch.zeros((1, 8000, 1), device=device)
u_ang = torch.zeros_like(u_lin, device=device)

for t in range(u_lin.size()[1]):
    linear_input = torch.tanh(u_lin[:, t]) @ torch.zeros((1,n_units)).to(device)
    angular_input = u_ang[:, t] @ torch.zeros((1,n_units)).to(device)

    x, z, theta, s, omega = unicycle_network(linear_input, angular_input, x, z, theta, s, omega)

    concatenated_states = torch.hstack((x, z, theta, s, omega))
    states_list.append(concatenated_states)
# %%
#%%
all_states_time = torch.vstack(states_list)
plt.plot(all_states_time[:,0:n_units].cpu().detach().numpy())
plt.show()
plt.plot(all_states_time[:,n_units*3:n_units*4].cpu().detach().numpy())
plt.show()
plt.plot(all_states_time[:,n_units*4:n_units*5].cpu().detach().numpy())
plt.show()
#%%
print(f"final coords {x[0,1].item()}, {z[0,1].item()} and {x[0,2].item()}, {z[0,2].item()}")
#%%
plt.scatter(x.cpu().detach().numpy(), z.cpu().detach().numpy())
plt.show()
# %%
print(f"Difference between simulation and numerical solution: \n {x[0,1].item()-x2_solution}, {z[0,1].item()-y2_solution} and {x[0,2].item()-x3_solution}, {z[0,2].item()-y3_solution}")
# %%
