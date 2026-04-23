# import numpy as np
# from itertools import combinations
# from copy import deepcopy

# # -----------------------------
# # Interval arithmetic helpers
# # -----------------------------
# def iadd(a, b):
#     return (a[0]+b[0], a[1]+b[1])

# def isub(a, b):
#     return (a[0]-b[1], a[1]-b[0])

# def imul(a, b):
#     vals = [a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1]]
#     return (min(vals), max(vals))

# def idiv(a, b):
#     if b[0] <= 0 <= b[1]:
#         raise ValueError("Division by interval containing zero")
#     vals = [a[0]/b[0], a[0]/b[1], a[1]/b[0], a[1]/b[1]]
#     return (min(vals), max(vals))

# def isquare(a):
#     l, h = a
#     if l >= 0:
#         return (l*l, h*h)
#     elif h <= 0:
#         return (h*h, l*l)
#     else:
#         return (0, max(l*l, h*h))

# def isqrt(a):
#     return (np.sqrt(a[0]), np.sqrt(a[1]))

# # -----------------------------
# # Parameters
# # -----------------------------
# N = 20
# domain_bounds = [-0.5, 0.5]  # small λ-domain for movable units
# connectivity = 6             # number of neighbors
# epsilon = 1e-8               # min r to avoid zero division

# # Random reference points
# p0 = np.random.uniform(-1, 1, (N, 2))

# # Random directions (unit vectors)
# angles = np.random.uniform(0, 2*np.pi, N)
# e = np.stack([np.cos(angles), np.sin(angles)], axis=1)

# # Initialize K and A with connectivity
# K = np.zeros((N, N))
# A = np.zeros((N, N))
# for i in range(N):
#     for offset in range(1, connectivity+1):
#         j = (i + offset) % N
#         K[i,j] = K[j,i] = np.random.uniform(0.5, 2.0)
#         A[i,j] = A[j,i] = np.random.uniform(0.5, 1.5)

# # Initial λ intervals for movable units
# lambda_intervals = [(domain_bounds[0], domain_bounds[1]) for _ in range(1, N)]

# # -----------------------------
# # Hessian interval for anchor
# # -----------------------------
# def compute_hessian_interval_anchor(lambda_ints):
#     H = [[(0.0,0.0) for _ in range(N-1)] for _ in range(N-1)]
    
#     for i,j in combinations(range(N),2):
#         if K[i,j] == 0:
#             continue
        
#         i_anchor = (i == 0)
#         j_anchor = (j == 0)
        
#         li = (0.0, 0.0) if i_anchor else lambda_ints[i-1]
#         lj = (0.0, 0.0) if j_anchor else lambda_ints[j-1]
        
#         rx = isub( (p0[j,0]-p0[i,0], p0[j,0]-p0[i,0]), imul(li,(e[i,0],e[i,0])) )
#         rx = iadd(rx, imul(lj,(e[j,0],e[j,0])))
#         rz = isub( (p0[j,1]-p0[i,1], p0[j,1]-p0[i,1]), imul(li,(e[i,1],e[i,1])) )
#         rz = iadd(rz, imul(lj,(e[j,1],e[j,1])))
        
#         r2 = iadd(isquare(rx), isquare(rz))
#         r = isqrt(r2)
#         r = (max(r[0], epsilon), r[1])
        
#         proj_i = iadd(imul(rx,(e[i,0],e[i,0])), imul(rz,(e[i,1],e[i,1])))
#         proj_j = iadd(imul(rx,(e[j,0],e[j,0])), imul(rz,(e[j,1],e[j,1])))
        
#         Kij = K[i,j]
#         Aij = A[i,j]
        
#         # Only update movable units
#         if not i_anchor and not j_anchor:
#             idx_i, idx_j = i-1, j-1
#             term1_i = imul(idiv(proj_i,r), idiv(proj_i,r))
#             term2_i = imul(isub((1,1), idiv((Aij,Aij),r)), isub((1,1), term1_i))
#             Hii = imul((Kij,Kij), iadd(term1_i, term2_i))
            
#             term1_j = imul(idiv(proj_j,r), idiv(proj_j,r))
#             term2_j = imul(isub((1,1), idiv((Aij,Aij),r)), isub((1,1), term1_j))
#             Hjj = imul((Kij,Kij), iadd(term1_j, term2_j))
            
#             H[idx_i][idx_i] = iadd(H[idx_i][idx_i], Hii)
#             H[idx_j][idx_j] = iadd(H[idx_j][idx_j], Hjj)
            
#             term_off = imul(idiv(proj_i,r), idiv(proj_j,r))
#             term_off2 = imul(isub((1,1), idiv((Aij,Aij),r)), term_off)
#             Hij = imul((-Kij,-Kij), iadd(term_off, term_off2))
#             H[idx_i][idx_j] = iadd(H[idx_i][idx_j], Hij)
#             H[idx_j][idx_i] = iadd(H[idx_j][idx_i], Hij)
#         elif not i_anchor and j_anchor:
#             idx_i = i-1
#             term1_i = imul(idiv(proj_i,r), idiv(proj_i,r))
#             term2_i = imul(isub((1,1), idiv((Aij,Aij),r)), isub((1,1), term1_i))
#             Hii = imul((Kij,Kij), iadd(term1_i, term2_i))
#             H[idx_i][idx_i] = iadd(H[idx_i][idx_i], Hii)
#         elif i_anchor and not j_anchor:
#             idx_j = j-1
#             term1_j = imul(idiv(proj_j,r), idiv(proj_j,r))
#             term2_j = imul(isub((1,1), idiv((Aij,Aij),r)), isub((1,1), term1_j))
#             Hjj = imul((Kij,Kij), iadd(term1_j, term2_j))
#             H[idx_j][idx_j] = iadd(H[idx_j][idx_j], Hjj)
#         # both anchors → skip
#     return H

# # -----------------------------
# # Gershgorin PD check
# # -----------------------------
# def is_PD(H_int):
#     N = len(H_int)
#     for i in range(N):
#         Hii_min = H_int[i][i][0]
#         off_sum_max = sum([abs(H_int[i][j][1]) for j in range(N) if j!=i])
#         if Hii_min - off_sum_max <= 0:
#             return False
#     return True

# # -----------------------------
# # Adaptive interval splitting
# # -----------------------------
# def split_interval_box(box):
#     # Split the longest interval in half
#     lengths = [b[1]-b[0] for b in box]
#     idx = np.argmax(lengths)
#     mid = (box[idx][0]+box[idx][1])/2
#     box1 = deepcopy(box)
#     box2 = deepcopy(box)
#     box1[idx] = (box[idx][0], mid)
#     box2[idx] = (mid, box[idx][1])
#     return box1, box2

# def find_PD_regions(box, depth=3):
#     """Recursively split and find sub-boxes with PD Hessian"""
#     if depth == 0:
#         H_int = compute_hessian_interval_anchor(box)
#         if is_PD(H_int):
#             return [box]
#         else:
#             return []
#     else:
#         box1, box2 = split_interval_box(box)
#         return find_PD_regions(box1, depth-1) + find_PD_regions(box2, depth-1)

# # Run adaptive splitting
# PD_subboxes = find_PD_regions(lambda_intervals, depth=4)
# print(f"Found {len(PD_subboxes)} sub-boxes with guaranteed PD Hessian.")
# for b in PD_subboxes:
#     print(b)

import numpy as np
from z3 import Real, Solver, If, And, Or, Sqrt

# -----------------------------
# Parameters
# -----------------------------
N = 6
connectivity = 3
epsilon = 1e-8
delta = 1e-3  # δ-relaxation for positive definiteness
domain_bounds = [-0.5, 0.5]

# Random reference points
p0 = np.random.uniform(-1, 1, (N, 2))
angles = np.random.uniform(0, 2*np.pi, N)
e = np.stack([np.cos(angles), np.sin(angles)], axis=1)

# Random K and A with limited connectivity
K = np.zeros((N, N))
A = np.zeros((N, N))
for i in range(N):
    for offset in range(1, connectivity+1):
        j = (i + offset) % N
        K[i,j] = K[j,i] = np.random.uniform(0.5, 2.0)
        A[i,j] = A[j,i] = np.random.uniform(0.5, 1.5)

# -----------------------------
# Z3 variables for movable units (λ1,...,λ5)
# -----------------------------
lam = [Real(f'lam{i}') for i in range(1, N)]  # λ0 is anchor

# -----------------------------
# Hessian entries (symbolic)
# -----------------------------
def hessian_diag(i, lam_vars):
    Hii = 0
    for j in range(N):
        if K[i,j] == 0 or i==j:
            continue
        li = 0 if i==0 else lam_vars[i-1]
        lj = 0 if j==0 else lam_vars[j-1]
        dx = (p0[j,0]-p0[i,0]) + lj*e[j,0] - li*e[i,0]
        dz = (p0[j,1]-p0[i,1]) + lj*e[j,1] - li*e[i,1]
        r = Sqrt(dx*dx + dz*dz) + epsilon
        proj_i = dx*e[i,0] + dz*e[i,1]
        Hii += K[i,j]*((proj_i/r)**2 + (1 - A[i,j]/r)*(1 - (proj_i/r)**2))
    return Hii

def hessian_offdiag(i,j, lam_vars):
    li = 0 if i==0 else lam_vars[i-1]
    lj = 0 if j==0 else lam_vars[j-1]
    dx = (p0[j,0]-p0[i,0]) + lj*e[j,0] - li*e[i,0]
    dz = (p0[j,1]-p0[i,1]) + lj*e[j,1] - li*e[i,1]
    r = Sqrt(dx*dx + dz*dz) + epsilon
    proj_i = dx*e[i,0] + dz*e[i,1]
    proj_j = dx*e[j,0] + dz*e[j,1]
    return -K[i,j]*((proj_i*proj_j)/r**2 + (1-A[i,j]/r)*(proj_i*proj_j)/r**2)

# -----------------------------
# Build Hessian matrix (movable units only)
# -----------------------------
H = [[0 for _ in range(N-1)] for _ in range(N-1)]
for i in range(1,N):
    for j in range(1,N):
        if i==j:
            H[i-1][j-1] = hessian_diag(i, lam)
        else:
            H[i-1][j-1] = hessian_offdiag(i,j, lam)

# -----------------------------
# Sylvester criterion with δ-relaxation
# Only first 1x1, 2x2, 3x3 minors for speed
# -----------------------------
constraints = []
for k in range(1,4):
    subH = [[H[i][j] for j in range(k)] for i in range(k)]
    if k==1:
        constraints.append(subH[0][0] > delta)
    elif k==2:
        det2 = subH[0][0]*subH[1][1] - subH[0][1]*subH[1][0]
        constraints.append(det2 > delta)
    elif k==3:
        a,b,c = subH[0]
        d,e,f = subH[1]
        g,h,i_val = subH[2]
        det3 = a*(e*i_val - f*h) - b*(d*i_val - f*g) + c*(d*h - e*g)
        constraints.append(det3 > delta)

# -----------------------------
# Domain bounds for λ
# -----------------------------
for l in lam:
    constraints.append(l >= domain_bounds[0])
    constraints.append(l <= domain_bounds[1])

# -----------------------------
# Solve with Z3
# -----------------------------
solver = Solver()
solver.add(constraints)
if solver.check() == sat:
    model = solver.model()
    print("Found λ values satisfying δ-PD Hessian:")
    for l in lam:
        print(f"{l} = {model[l]}")
else:
    print("No solution found in this domain for δ-PD Hessian.")
