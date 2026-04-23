import numpy as np
from itertools import combinations
from copy import deepcopy

# -----------------------------
# Interval arithmetic helpers
# -----------------------------
def iadd(a, b):
    return (a[0]+b[0], a[1]+b[1])

def isub(a, b):
    return (a[0]-b[1], a[1]-b[0])

def imul(a, b):
    vals = [a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1]]
    return (min(vals), max(vals))

def idiv(a, b):
    if b[0] <= 0 <= b[1]:
        raise ValueError("Division by interval containing zero")
    vals = [a[0]/b[0], a[0]/b[1], a[1]/b[0], a[1]/b[1]]
    return (min(vals), max(vals))

def isquare(a):
    l, h = a
    if l >= 0:
        return (l*l, h*h)
    elif h <= 0:
        return (h*h, l*l)
    else:
        return (0, max(l*l, h*h))

def isqrt(a):
    return (np.sqrt(a[0]), np.sqrt(a[1]))

# -----------------------------
# Parameters
# -----------------------------
N = 6
domain_bounds = [-0.5, 0.5]  # initial λ-domain for movable units
connectivity = 3
epsilon = 1e-8

# Random reference points
p0 = np.random.uniform(-1, 1, (N,2))
# Random directions (unit vectors)
angles = np.random.uniform(0, 2*np.pi, N)
e = np.stack([np.cos(angles), np.sin(angles)], axis=1)

# Random stiffnesses and rest lengths with connectivity
K = np.zeros((N,N))
A = np.zeros((N,N))
for i in range(N):
    for offset in range(1, connectivity+1):
        j = (i + offset) % N
        K[i,j] = K[j,i] = np.random.uniform(0.5, 2.0)
        A[i,j] = A[j,i] = np.random.uniform(0.5, 1.5)

# Initial λ intervals for movable units
lambda_intervals = [(domain_bounds[0], domain_bounds[1]) for _ in range(1, N)]

# -----------------------------
# Hessian interval computation
# -----------------------------
def compute_hessian_interval_anchor(lambda_ints):
    H = [[(0.0,0.0) for _ in range(N-1)] for _ in range(N-1)]
    
    for i,j in combinations(range(N),2):
        if K[i,j]==0:
            continue
        i_anchor = (i==0)
        j_anchor = (j==0)
        li = (0.0,0.0) if i_anchor else lambda_ints[i-1]
        lj = (0.0,0.0) if j_anchor else lambda_ints[j-1]
        
        rx = isub( (p0[j,0]-p0[i,0], p0[j,0]-p0[i,0]), imul(li,(e[i,0],e[i,0])) )
        rx = iadd(rx, imul(lj,(e[j,0],e[j,0])))
        rz = isub( (p0[j,1]-p0[i,1], p0[j,1]-p0[i,1]), imul(li,(e[i,1],e[i,1])) )
        rz = iadd(rz, imul(lj,(e[j,1],e[j,1])))
        
        r2 = iadd(isquare(rx), isquare(rz))
        r = isqrt(r2)
        r = (max(r[0], epsilon), r[1])
        
        proj_i = iadd(imul(rx,(e[i,0],e[i,0])), imul(rz,(e[i,1],e[i,1])))
        proj_j = iadd(imul(rx,(e[j,0],e[j,0])), imul(rz,(e[j,1],e[j,1])))
        
        Kij = K[i,j]
        Aij = A[i,j]
        
        if not i_anchor and not j_anchor:
            idx_i, idx_j = i-1, j-1
            term1_i = imul(idiv(proj_i,r), idiv(proj_i,r))
            term2_i = imul(isub((1,1), idiv((Aij,Aij),r)), isub((1,1), term1_i))
            Hii = imul((Kij,Kij), iadd(term1_i, term2_i))
            
            term1_j = imul(idiv(proj_j,r), idiv(proj_j,r))
            term2_j = imul(isub((1,1), idiv((Aij,Aij),r)), isub((1,1), term1_j))
            Hjj = imul((Kij,Kij), iadd(term1_j, term2_j))
            
            H[idx_i][idx_i] = iadd(H[idx_i][idx_i], Hii)
            H[idx_j][idx_j] = iadd(H[idx_j][idx_j], Hjj)
            
            term_off = imul(idiv(proj_i,r), idiv(proj_j,r))
            term_off2 = imul(isub((1,1), idiv((Aij,Aij),r)), term_off)
            Hij = imul((-Kij,-Kij), iadd(term_off, term_off2))
            H[idx_i][idx_j] = iadd(H[idx_i][idx_j], Hij)
            H[idx_j][idx_i] = iadd(H[idx_j][idx_i], Hij)
        elif not i_anchor and j_anchor:
            idx_i = i-1
            term1_i = imul(idiv(proj_i,r), idiv(proj_i,r))
            term2_i = imul(isub((1,1), idiv((Aij,Aij),r)), isub((1,1), term1_i))
            Hii = imul((Kij,Kij), iadd(term1_i, term2_i))
            H[idx_i][idx_i] = iadd(H[idx_i][idx_i], Hii)
        elif i_anchor and not j_anchor:
            idx_j = j-1
            term1_j = imul(idiv(proj_j,r), idiv(proj_j,r))
            term2_j = imul(isub((1,1), idiv((Aij,Aij),r)), isub((1,1), term1_j))
            Hjj = imul((Kij,Kij), iadd(term1_j, term2_j))
            H[idx_j][idx_j] = iadd(H[idx_j][idx_j], Hjj)
    return H

# -----------------------------
# Gershgorin PD check
# -----------------------------
def is_PD(H_int):
    N = len(H_int)
    for i in range(N):
        Hii_min = H_int[i][i][0]
        off_sum_max = sum([abs(H_int[i][j][1]) for j in range(N) if j!=i])
        if Hii_min - off_sum_max <= 0:
            return False
    return True

# -----------------------------
# Adaptive branch-and-bound search
# -----------------------------
def split_interval_box(box):
    lengths = [b[1]-b[0] for b in box]
    idx = np.argmax(lengths)
    mid = (box[idx][0]+box[idx][1])/2
    box1 = deepcopy(box)
    box2 = deepcopy(box)
    box1[idx] = (box[idx][0], mid)
    box2[idx] = (mid, box[idx][1])
    return box1, box2

def find_PD_regions(box, depth=3):
    if depth == 0:
        H_int = compute_hessian_interval_anchor(box)
        if is_PD(H_int):
            return [box]
        else:
            return []
    else:
        box1, box2 = split_interval_box(box)
        return find_PD_regions(box1, depth-1) + find_PD_regions(box2, depth-1)

# -----------------------------
# Run adaptive PD search
# -----------------------------
PD_subboxes = find_PD_regions(lambda_intervals, depth=20)
print(f"Found {len(PD_subboxes)} sub-boxes with guaranteed PD Hessian:")
for b in PD_subboxes:
    print(b)
