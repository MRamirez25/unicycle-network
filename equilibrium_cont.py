import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

# ==============================
# PARAMETERS
# ==============================
N = 10
n_connections_anchor = 11
n_connections = 8

# Initial headings (radians)
theta = np.linspace(0, np.pi/3, N)
e_fixed = np.stack([np.cos(theta), np.sin(theta)], axis=1)

# Reference points
p0 = np.random.rand(N, 2) * 2 - 1

# Random symmetric stiffness and rest lengths
num_connections = N*(N-1)//2
stiffnesses_array = np.random.rand(num_connections) * 2 + 0.5
restlengths_array = np.random.rand(num_connections) * 1 + 0.5

K_mat = np.zeros((N,N))
A_mat = np.zeros((N,N))
idx = 0
for i in range(N):
    for j in range(i+1, N):
        if i==0:
            if np.abs(i-j-1) < n_connections_anchor:
                K_mat[i,j] = stiffnesses_array[idx]
                K_mat[j,i] = stiffnesses_array[idx]
                A_mat[i,j] = restlengths_array[idx]
                A_mat[j,i] = restlengths_array[idx]
        else:
            if np.abs(i-j-1) < n_connections:
                K_mat[i,j] = stiffnesses_array[idx]
                K_mat[j,i] = stiffnesses_array[idx]
                A_mat[i,j] = restlengths_array[idx]
                A_mat[j,i] = restlengths_array[idx]
        idx += 1

print("K_mat:\n", K_mat)
# ==============================
# POSITION FUNCTION
# ==============================
def positions(s, e):
    """Compute 2D positions from s and current e vectors"""
    s_full = np.zeros(N)
    s_full[1:] = s
    return p0 + s_full[:, None] * e

# ==============================
# EQUILIBRIUM EQUATIONS
# ==============================
def equilibrium_equations_loop(s, e):
    """Compute projected forces along lines for reduced coordinates s"""
    p = positions(s, e)
    f = np.zeros(N-1)
    for i in range(1, N):
        Fi = 0.0
        for j in range(N):
            if i==j:
                continue
            d = p[i]-p[j]
            r = np.linalg.norm(d)
            if r==0:
                continue
            Fi += K_mat[i,j]*(A_mat[i,j]-r)*np.dot(d, e[i])/r
        f[i-1] = Fi
    return f

# ==============================
# INITIAL EQUILIBRIUM
# ==============================
s0 = np.random.randn(N-1)
sol = root(lambda s: equilibrium_equations_loop(s, e_fixed), s0)
if not sol.success:
    raise RuntimeError("Initial equilibrium not found")
s_prev = sol.x

# ==============================
# CONTINUATION PARAMETERS
# ==============================
n_steps = 100
theta_index = 1   # which unicycle to rotate
theta_values = np.linspace(theta[theta_index], theta[theta_index]+np.pi*2, n_steps)

continuation_points = []
rotated_lines = []

for th in theta_values:
    # 1. Rotate only the chosen line
    e_rot = e_fixed.copy()
    e_rot[theta_index] = [np.cos(th), np.sin(th)]
    
    # 2. Solve equilibrium using current e
    sol = root(lambda s: equilibrium_equations_loop(s, e_rot), s_prev)
    if sol.success and np.linalg.norm(sol.fun) < 1e-6:
        s_prev = sol.x
        continuation_points.append(s_prev)
        rotated_lines.append(e_rot.copy())
    else:
        print(f"Warning: solver failed at theta={th}")

# ==============================
# PLOT CONTINUATION WITH LINES
# ==============================
continuation_points = np.array(continuation_points)
colors = plt.cm.viridis(np.linspace(0,1,len(continuation_points)))
t_line = np.linspace(-2.5, 2.5, 200)

plt.figure(figsize=(7,7))

for k, s in enumerate(continuation_points):
    e_current = rotated_lines[k]
    p = positions(s, e_current)
    
    # plot equilibrium points
    plt.scatter(p[:,0], p[:,1], color=colors[k], s=60)
    
    # plot lines for each unicycle
    for i in range(N):
        line = p0[i] + t_line[:, None] * e_current[i]
        if i == theta_index:
            plt.plot(line[:,0], line[:,1], color=colors[k], alpha=0.6)
        else:
            plt.plot(line[:,0], line[:,1], '--', color='gray', alpha=0.4)

plt.xlabel("x")
plt.ylabel("z")
plt.title(f"Continuation of equilibria with rotating line θ[{theta_index}]")
plt.grid(True)
plt.axis("equal")
plt.show()
