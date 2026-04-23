import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

# ============================================================
# PARAMETERS
# ============================================================

N = 10  # number of unicycles
n_connections_fraction = 1.0
n_connections_anchor_fraction = 1.0
n_connections_anchor = int(n_connections_anchor_fraction * (N + 1))
n_connections = int(n_connections_fraction * (N))
# Fixed headings (radians)
np.random.seed(42)
theta = np.random.uniform(0, 2*np.pi, N)
e = np.stack([np.cos(theta), np.sin(theta)], axis=1)

# Reference points
p0 = np.random.rand(N, 2) * 2 - 1  # random points in [-1,1]x[-1,1]

# ============================================================
# RANDOM SYMMETRIC STIFFNESS AND REST-LENGTH MATRICES
# ============================================================

# Number of possible connections
num_connections = N*(N-1)//2
stiffnesses_array = np.random.rand(num_connections) * 2 + 0.5  # random K in [0.5, 2.5]
restlengths_array = np.random.rand(num_connections) * 1 + 0.5  # random A in [0.5, 1.5]

K_mat = np.zeros((N, N))
A_mat = np.zeros((N, N))

idx = 0
for i in range(N):
    for j in range(i + 1, N):
        if i == 0:
            if np.abs(i-j-1) < n_connections_anchor:
                K_mat[i, j] = stiffnesses_array[idx]
                K_mat[j, i] = stiffnesses_array[idx]
                A_mat[i, j] = restlengths_array[idx]
                A_mat[j, i] = restlengths_array[idx]
        else:
            if np.abs(i-j-1) < n_connections:
                K_mat[i, j] = stiffnesses_array[idx]
                K_mat[j, i] = stiffnesses_array[idx]
                A_mat[i, j] = restlengths_array[idx]
                A_mat[j, i] = restlengths_array[idx]
        idx += 1
print("Stiffness matrix K:\n", K_mat)
# ============================================================
# GEOMETRY FUNCTIONS
# ============================================================

def positions(s):
    """Compute positions of all unicycles from line parameters s."""
    s_full = np.zeros(N)
    s_full[1:] = s  # fix first unicycle s0=0
    return p0 + s_full[:, None] * e

# ============================================================
# VECTORIAL EQUILIBRIUM FUNCTION
# ============================================================
def equilibrium_equations_loop(s):
    """
    Loop-based equilibrium equations (original version).
    s: (N-1,) array of line parameters for unicycles 1..N-1
    Returns: (N-1,) projected forces
    """
    p = positions(s)  # full positions of all N unicycles
    f = np.zeros(N - 1)

    for i in range(1, N):  # skip first unicycle (fixed)
        Fi = 0.0
        for j in range(N):
            if i == j:
                continue
            d = p[i] - p[j]
            r = np.linalg.norm(d)
            if r == 0:
                continue  # avoid division by zero
            Fi += K_mat[i, j] * (A_mat[i, j] - r) * np.dot(d, e[i]) / r
        f[i - 1] = Fi

    return f

def equilibrium_equations(s):
    """Compute projected equilibrium equations (vectorized)."""
    p = positions(s)  # (N,2)
    d = p[:, None, :] - p[None, :, :]  # (N,N,2)
    r = np.linalg.norm(d, axis=2) + np.eye(N)  # avoid division by zero
    F_contrib = K_mat * (A_mat - r) * np.einsum('ijk,ik->ij', d, e) / r
    F = np.sum(F_contrib, axis=1)
    return F[1:]  # drop first unicycle (fixed)

# ============================================================
# NUMERICAL DERIVATIVES
# ============================================================

def numerical_jacobian(f, s, eps=1e-6):
    n = len(s)
    J = np.zeros((n, n))
    for i in range(n):
        ds = np.zeros(n)
        ds[i] = eps
        J[:, i] = (f(s + ds) - f(s - ds)) / (2 * eps)
    return J

def numerical_hessian(f, s, eps=1e-5):
    n = len(s)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei = np.zeros(n); ei[i] = eps
            ej = np.zeros(n); ej[j] = eps
            H[i, j] = (
                f(s + ei + ej) - f(s + ei - ej)
                - f(s - ei + ej) + f(s - ei - ej)
            ) / (4 * eps**2)
    return H

def reduced_potential(s):
    """Compute reduced potential energy."""
    p = positions(s)
    d = p[:, None, :] - p[None, :, :]
    r = np.linalg.norm(d, axis=2)
    U = 0.5 * np.sum(K_mat * (A_mat - r)**2)
    return U / 2  # divide by 2 to avoid double-counting
#### check two versions ###

s_test = np.random.randn(N-1)

f_loop = equilibrium_equations_loop(s_test)
f_vector = equilibrium_equations(s_test)  # your vectorized version

print("Max difference:", np.max(np.abs(f_loop - f_vector)))

# ============================================================
# FIND EQUILIBRIA (RANDOM RESTARTS)
# ============================================================

solutions = []
n_trials = 10000
# np.random.seed(1000)

for _ in range(n_trials):
    s0 = np.random.randn(N - 1)
    sol = root(equilibrium_equations, s0)
    if sol.success and np.linalg.norm(sol.fun) < 1e-6:
        solutions.append(sol.x)

solutions = np.array(solutions)

# ============================================================
# REMOVE DUPLICATES
# ============================================================

def unique_solutions(solutions, tol=1e-4):
    unique = []
    for s in solutions:
        if not any(np.linalg.norm(s - u) < tol for u in unique):
            unique.append(s)
    return np.array(unique)

unique_eq = unique_solutions(solutions)
print(f"Found {len(unique_eq)} unique equilibria")

# ============================================================
# ANALYZE ALL EQUILIBRIA
# ============================================================

for k, s in enumerate(unique_eq):
    res = np.linalg.norm(equilibrium_equations(s))
    J = numerical_jacobian(equilibrium_equations, s)
    svals = np.linalg.svd(J, compute_uv=False)
    nullity = np.sum(svals < 1e-5)

    H = numerical_hessian(reduced_potential, s)
    eigvals = np.linalg.eigvalsh(H)

    # print(f"\nEquilibrium {k}")
    # print("s* =", s)
    # print("Residual:", res)
    if nullity > 0:
        print(f"Equilibrium {k} has nullity {nullity}")
    # print("Nullity:", nullity)
    # print("Hessian eigenvalues:", eigvals)

# ============================================================
# PLOT ALL EQUILIBRIA OVERLAID
# ============================================================

plt.figure(figsize=(7,7))
t = np.linspace(-2.5, 2.5, 200)
for i in range(N):
    line = p0[i] + t[:, None] * e[i]
    plt.plot(line[:,0], line[:,1], '--', color='gray', alpha=0.4)

colors = plt.cm.tab10(np.linspace(0,1,len(unique_eq)))
markers = ['o','s','^','D','v','P','*','X']

for k, s in enumerate(unique_eq):
    p = positions(s)
    plt.scatter(p[:,0], p[:,1], color=colors[k], marker=markers[k%len(markers)], s=70, label=f"Eq {k}")

plt.axis('equal')
plt.grid(True)
plt.xlabel("x")
plt.ylabel("z")
plt.title("All Equilibria (Overlay)")
plt.legend()
plt.show()

# ============================================================
from sklearn.decomposition import PCA

s_array = np.array(unique_eq)  # shape: (num_eq, N-1)
pca = PCA(n_components=2)
s_2d = pca.fit_transform(s_array)

plt.scatter(s_2d[:,0], s_2d[:,1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Equilibria projected to 2D using PCA")
plt.grid(True)
plt.show()

from pandas.plotting import parallel_coordinates
import pandas as pd

df = pd.DataFrame(s_array, columns=[f"s{i+1}" for i in range(N-1)])
df["eq"] = np.arange(len(s_array))  # just for labeling
parallel_coordinates(df, "eq", colormap=plt.cm.tab10)
plt.show()