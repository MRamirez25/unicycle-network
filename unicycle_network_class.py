import time
from torch import nn
import torch
import numpy as np
import pdb

class UnicycleNetwork(nn.Module):
    def __init__(self, n_inp, n_units, dt, lin_stiff_min=0.1, lin_stiff_max=0.5, 
                 ang_stiff_min=0.1, ang_stiff_max=0.3, lin_damping_min=0.1, lin_damping_max=0.2,
                 ang_damping_min=0.1, ang_damping_max=0.2, eq_dist_min=0.5, eq_dist_max=1.0, eq_dist_min_ang=0.0,
                 eq_dist_max_ang=np.pi,
                 lin_input_map=None, ang_input_map=None, n_connections=None, n_connections_anchor=2,
                 n_connections_ang=None, n_connections_anchor_ang=2):
        super().__init__()
        self.n_units = n_units
        self.dt = dt
        self.lin_damping = torch.rand(1,n_units, requires_grad=False) * (lin_damping_max - lin_damping_min) + lin_damping_min
        self.ang_damping = torch.rand(1,n_units, requires_grad=False) * (ang_damping_max - ang_damping_min) + ang_damping_min
        # self.dist_ang_coupling = torch.rand(n_units, n_units) * (ang_stiff_max - ang_stiff_min) + ang_stiff_min
        # self.dist_ang_coupling = nn.Parameter(self.dist_ang_coupling, requires_grad=False)
        self.mass_vector = torch.ones(1, n_units, requires_grad=False)
        self.mass_vector[0,0] = 0.
        self.j_vector = torch.ones(1, n_units, requires_grad=False)
        self.j_vector[0,0] = 0.


        stiffnesses_array = (torch.rand(int((n_units**2-n_units)/2),1)) * (lin_stiff_max - lin_stiff_min) + lin_stiff_min
        self.stiffness_coupling_matrix = torch.zeros((n_units, n_units))
        if n_connections is not None:
            self.n_connections = n_connections
        else:
            self.n_connections = n_units
        idx=0
        for i in range(n_units):
            for j in range(i + 1, n_units):
                if i == 0:
                    if np.abs(i-j-1) < n_connections_anchor:
                        self.stiffness_coupling_matrix[i, j] = stiffnesses_array[idx]
                        self.stiffness_coupling_matrix[j,i] = stiffnesses_array[idx]
                else:
                    if np.abs(i-j-1) < n_connections:
                        self.stiffness_coupling_matrix[i, j] = stiffnesses_array[idx]
                        self.stiffness_coupling_matrix[j,i] = stiffnesses_array[idx]
                idx += 1
        self.stiffness_coupling_matrix = nn.Parameter(self.stiffness_coupling_matrix, requires_grad=False)

        stiffnesses_array = (torch.rand(int((n_units**2-n_units)/2),1)) * (ang_stiff_max - ang_stiff_min) + ang_stiff_min
        self.dist_ang_coupling = torch.zeros((n_units, n_units))
        if n_connections_ang is not None:
            self.n_connections_ang = n_connections_ang
        else:
            self.n_connections_ang = n_units
        idx=0
        for i in range(n_units):
            for j in range(i + 1, n_units):
                if i == 0:
                    if np.abs(i-j-1) < n_connections_anchor_ang:
                        self.dist_ang_coupling[i, j] = stiffnesses_array[idx]
                        self.dist_ang_coupling[j,i] = stiffnesses_array[idx]
                else:
                    if np.abs(i-j-1) < n_connections_ang:
                        self.dist_ang_coupling[i, j] = stiffnesses_array[idx]
                        self.dist_ang_coupling[j,i] = stiffnesses_array[idx]
                idx += 1
        self.dist_ang_coupling = nn.Parameter(self.dist_ang_coupling, requires_grad=False)

        eq_distances_array = (torch.rand(int((n_units**2-n_units)/2),1)) * (eq_dist_max - eq_dist_min) + eq_dist_min
        self.eq_distances_matrix = torch.zeros((n_units, n_units))
        idx=0
        for i in range(n_units):
            for j in range(i + 1, n_units):
                self.eq_distances_matrix[i, j] = eq_distances_array[idx]
                self.eq_distances_matrix[j,i] = eq_distances_array[idx]
                idx += 1

        self.eq_distances_matrix = self.eq_distances_matrix.reshape(n_units, n_units, 1)
        self.eq_distances_matrix = nn.Parameter(self.eq_distances_matrix, requires_grad=False)

        # equilibrium matrix for angular coupling
        eq_distances_array = (torch.rand(int((n_units**2-n_units)/2),1)) * (eq_dist_max_ang - eq_dist_min_ang) + eq_dist_min_ang
        self.eq_distances_mat_ang = torch.zeros((n_units, n_units))
        idx=0
        for i in range(n_units):
            for j in range(i + 1, n_units):
                self.eq_distances_mat_ang[i, j] = eq_distances_array[idx]
                self.eq_distances_mat_ang[j,i] = -eq_distances_array[idx]
                idx += 1
        self.eq_distances_mat_ang = self.eq_distances_mat_ang.reshape(n_units, n_units)
        self.eq_distances_mat_ang = nn.Parameter(self.eq_distances_mat_ang, requires_grad=False)

        # if not lin_input_map:
        #     lin_input_map = torch.rand(n_units, n_inp)
        #     self.lin_input_map = nn.Parameter(lin_input_map, requires_grad=False)
        # if not ang_input_map:
        #     ang_input_map = torch.rand(n_units, n_inp)
        #     self.ang_input_map = nn.Parameter(ang_input_map, requires_grad=False)
        
    def forward(self, u_lin, u_ang, x, z, theta, s, omega):
        bs = u_lin.shape[0]
        linear_inp_forces = u_lin
        coords_2d = torch.stack((x, z), dim=-1)  # (b, n_units, 2)  # Stack x and z into a 3D tensor
        theta_unit_vectors = self.angle_to_unit_vector(theta)  # (b, n_units, 2)
        distance_vectors = self.pairwise_differences(coords_2d)  # (b, n_units, n_units, 2)

        # Compute the distance magnitudes batch-wise (torch.norm instead of np.linalg.norm)
        distance_magnitudes = torch.norm(distance_vectors, dim=-1, keepdim=True)  # (b, n_units, n_units, 1)

        # Normalize the distance vectors, avoid division by zero
        distance_vectors_normalized = torch.nan_to_num(distance_vectors / distance_magnitudes)  # (b, n_units, n_units, 2)

        # Forces computation
        forces_before_projection = self.stiffness_coupling_matrix[None, :, :, None] * (self.eq_distances_matrix - distance_magnitudes) * distance_vectors_normalized  # (b, n_units, n_units, 2)

        # Project forces along the theta direction using einsum (b, n_units, n_units, 2) -> (b, n_units)
        projected_forces = torch.einsum('bijk,bik->bi', forces_before_projection, theta_unit_vectors)  # (b, n_units)

        # Ensure projected forces have the correct shape (b, n_units, 1)
        # projected_forces = projected_forces.unsqueeze(-1)

        v_dot = (linear_inp_forces + projected_forces - (s * self.lin_damping)) * self.mass_vector

        s = s + v_dot*self.dt

        inp_term_theta = u_ang
        # Expand theta for pairwise differences
        theta_expanded_1 = theta[:, :, None]   # shape (b, n_units, 1)
        theta_expanded_2 = theta[:, None, :]   # shape (b, 1, n_units)
        ang_distances = theta_expanded_1 - theta_expanded_2
        coupling_term_ang = torch.sum(self.dist_ang_coupling[None, :, :] * (self.eq_distances_mat_ang.repeat(bs,1,1)-ang_distances), dim=2, keepdim=False)  # shape (b, n_units, 1)
        omega_dot = ((inp_term_theta + coupling_term_ang) - omega * self.ang_damping) * self.j_vector
        omega = omega + omega_dot*self.dt

        theta = theta + self.dt*omega
        x = x + torch.cos(theta) * s
        z = z + torch.sin(theta) * s

        return x, z, theta, s, omega
    
    def pairwise_differences(self, arr):
        """
        Computes all pairwise differences between vectors in the given 2D array.
        
        Parameters:
        arr (numpy.ndarray): A 2D array of shape (n, m), where n is the number of vectors
                            and m is the dimension of each vector.
                            
        Returns:
        numpy.ndarray: A 3D array of shape (n, n, m) where the element at [i, j, :] is the 
                    difference between the i-th and j-th vectors.
        """
        b, n, m = arr.shape
        # Expand dimensions to broadcast the subtraction over the pairs within each batch
        expanded_arr1 = arr.unsqueeze(2)  # (b, n_units, 1, 2)
        expanded_arr2 = arr.unsqueeze(1)  # (b, 1, n_units, 2)
        
        # Calculate pairwise differences
        differences = expanded_arr1 - expanded_arr2  # (b, n_units, n_units, 2)
        
        return differences
    
    def angle_to_unit_vector(self, angle):
        cos_angle = torch.cos(angle)  # (b, n_units)
        sin_angle = torch.sin(angle)  # (b, n_units)
    
        # Stack cos and sin to create the unit vector (b, n_units, 2)
        return torch.stack((cos_angle, sin_angle), dim=-1)

class UnicycleReservoir(nn.Module):
    def __init__(self, n_inp, n_units, dt, n_out, lin_stiff_min=0.1, lin_stiff_max=0.5, 
                 ang_stiff_min=0.1, ang_stiff_max=0.3, lin_damping_min=0.1, lin_damping_max=0.2,
                 ang_damping_min=0.1, ang_damping_max=0.2, eq_dist_min=0.5, eq_dist_max=1.0,
                 eq_dist_min_ang=0.0, eq_dist_max_ang=np.pi,
                 lin_input_map=None, ang_input_map=None, n_connections=None, inp_bias=0, n_connections_anchor=2, 
                 n_connections_ang=None, n_connections_anchor_ang=2, n_past_steps_readout=0) -> None:
        super().__init__()
        self.n_inp = n_inp
        self.n_units = n_units
        self.unicycle_network = UnicycleNetwork(n_inp, n_units, dt, lin_stiff_min=lin_stiff_min, lin_stiff_max=lin_stiff_max, 
                 ang_stiff_min=ang_stiff_min, ang_stiff_max=ang_stiff_max, lin_damping_min=lin_damping_min, lin_damping_max=lin_damping_max,
                 ang_damping_min=ang_damping_min, ang_damping_max=ang_damping_max, eq_dist_min=eq_dist_min, eq_dist_max=eq_dist_max,
                 eq_dist_min_ang=eq_dist_min_ang, eq_dist_max_ang=eq_dist_max_ang,
                 lin_input_map=None, ang_input_map=None, n_connections=n_connections, n_connections_anchor=n_connections_anchor, 
                 n_connections_ang=n_connections_ang, n_connections_anchor_ang=n_connections_anchor_ang)
        self.readout = nn.Linear(n_units*5*(n_past_steps_readout+1), n_out)

        self.inp_bias=inp_bias
        self.n_past_steps_readout = n_past_steps_readout

        if lin_input_map is None:
            lin_input_map = torch.rand(n_inp, n_units)
            self.lin_input_map = nn.Parameter(lin_input_map, requires_grad=False)
        else:
            self.lin_input_map = lin_input_map
        if ang_input_map is None:
            ang_input_map = torch.rand(n_inp, n_units)
            self.ang_input_map = nn.Parameter(ang_input_map, requires_grad=False)
        else:
            self.ang_input_map = ang_input_map
    
    def forward(self, u_lin, u_ang):
        #start = time.time()

        x = self.x_init
        z = self.z_init
        theta = self.theta_init
        s = self.s_init
        omega = self.omega_init
        states_list = []

        for t in range(u_lin.size()[1]):
            # linear_input = torch.tanh(u_lin[:, t] +self.inp_bias) @ self.lin_input_map
            linear_input = (u_lin[:, t] +self.inp_bias) @ self.lin_input_map
            angular_input = (u_ang[:, t]) @ self.ang_input_map
            # print(self.lin_input_map)


            # Debug: Check for NaNs or large values
            # print(f"Before unicycle_network loop {t}, linear_input: {linear_input}, angular_input: {angular_input}")

            x, z, theta, s, omega = self.unicycle_network(linear_input, angular_input, x, z, theta, s, omega)

            # Debug: Check outputs
            # print(f"After loop {t}, x: {x}, z: {z}, theta: {theta}, s: {s}, omega: {omega}")

            concatenated_states = torch.hstack((x, z, theta, s, omega))
            states_list.append(concatenated_states)
        #end = time.time()
        #print("elapsed time dynamics", end - start)
        #start = time.time()
            # Check for NaN gradients
            # if t == 1:
        if self.n_past_steps_readout > 0:
            mid_states_idxs = [(int(u_lin.size()[1] / self.n_past_steps_readout) - 1)*k for k in range(1,self.n_past_steps_readout+1)]
            mid_states = torch.hstack(([states_list[idx] for idx in mid_states_idxs]))
            # output = self.readout(torch.hstack((mid_states, x, z, theta, s, omega)))
        else:
            # output = self.readout(torch.hstack((x, z, theta, s, omega)))
            mid_states = states_list[-1]
        output = None
        #end = time.time()
        #print("elapsed time readout", end - start)
        return states_list, output, mid_states
    
    def set_init_states_random(self, bs):
        self.x_init = torch.randn(self.n_units).repeat(bs,1)
        self.z_init = torch.randn(self.n_units).repeat(bs,1)
        self.theta_init = torch.randn(self.n_units).repeat(bs,1)
        self.s_init = torch.randn(self.n_units).repeat(bs,1)
        self.omega_init = torch.randn(self.n_units).repeat(bs,1)

    def set_init_states_grid(self, bs, num_rows, num_cols, spacing):
        """
        Initializes unit positions (x, z) on a grid with specified spacing.
        
        Args:
            bs (int): Batch size.
            num_rows (int): Number of rows in the grid.
            num_cols (int): Number of columns in the grid.
            spacing (tuple): (x_spacing, z_spacing) defining the spacing between units.
        """
        x_spacing, z_spacing = spacing
        
        x_coords = np.tile(np.arange(num_cols) * x_spacing, num_rows)
        z_coords = np.repeat(np.arange(num_rows) * z_spacing, num_cols)
        
        # Ensure correct number of units
        assert len(x_coords) == len(z_coords), "Mismatch in grid coordinates"
        assert len(x_coords) >= self.n_units, "Grid too small for required units"
        
        # Select only required units
        x_coords = x_coords[:self.n_units]
        z_coords = z_coords[:self.n_units]
        
        # Convert to tensors and repeat for batch size
        self.x_init = torch.tensor(x_coords, dtype=torch.float32).repeat(bs, 1)
        self.z_init = torch.tensor(z_coords, dtype=torch.float32).repeat(bs, 1)
        self.theta_init = torch.randn(self.n_units).repeat(bs,1)
        self.s_init = torch.randn(self.n_units).repeat(bs,1)
        self.omega_init = torch.randn(self.n_units).repeat(bs,1)


    def set_init_states(self, bs, x, z, theta, s, omega):
        self.x_init = torch.tensor(x).repeat(bs,1)
        self.z_init = torch.tensor(z).repeat(bs,1)
        self.theta_init = torch.tensor(theta).repeat(bs,1)
        self.s_init = torch.tensor(s).repeat(bs,1)
        self.omega_init = torch.tensor(omega).repeat(bs,1)
