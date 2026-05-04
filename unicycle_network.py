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
                 n_connections_ang=None, n_connections_anchor_ang=2,
                 use_capped_dynamics=False, max_speed=0.15, max_acceleration=1.0,
                 position_noise_std=0.0):
        super().__init__()
        self.n_units = n_units
        self.dt = dt
        self.use_capped_dynamics = use_capped_dynamics
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.position_noise_std = position_noise_std
        self.lin_damping = torch.rand(1,n_units, requires_grad=False) * (lin_damping_max - lin_damping_min) + lin_damping_min
        self.ang_damping = torch.rand(1,n_units, requires_grad=False) * (ang_damping_max - ang_damping_min) + ang_damping_min
        # self.dist_ang_coupling = torch.rand(n_units, n_units) * (ang_stiff_max - ang_stiff_min) + ang_stiff_min
        # self.dist_ang_coupling = nn.Parameter(self.dist_ang_coupling, requires_grad=False)
        self.mass_vector = torch.ones(1, n_units, requires_grad=False)
        self.mass_vector[0,0] = 0.
        self.j_vector = torch.ones(1, n_units, requires_grad=False)
        self.j_vector[0,0] = 0.

        # Create coupling matrices with their equilibrium distances in one pass
        stiff_matrix, eq_dist_matrix = self._create_sparse_coupling_with_eq_distances(
            n_units, n_connections, n_connections_anchor,
            lin_stiff_min, lin_stiff_max, eq_dist_min, eq_dist_max
        )
        self.stiffness_coupling_matrix = stiff_matrix
        self.eq_distances_matrix = nn.Parameter(eq_dist_matrix.reshape(n_units, n_units, 1), requires_grad=False)
        
        ang_matrix, eq_ang_matrix = self._create_sparse_coupling_with_eq_distances(
            n_units, n_connections_ang, n_connections_anchor_ang,
            ang_stiff_min, ang_stiff_max, eq_dist_min_ang, eq_dist_max_ang, 
            antisymmetric=True
        )
        self.dist_ang_coupling = ang_matrix
        self.eq_distances_mat_ang = nn.Parameter(eq_ang_matrix, requires_grad=False)

        # if not lin_input_map:
        #     lin_input_map = torch.rand(n_units, n_inp)
        #     self.lin_input_map = nn.Parameter(lin_input_map, requires_grad=False)
        # if not ang_input_map:
        #     ang_input_map = torch.rand(n_units, n_inp)
        #     self.ang_input_map = nn.Parameter(ang_input_map, requires_grad=False)
    
    def _create_sparse_coupling_with_eq_distances(self, n_units, n_connections, n_connections_anchor, 
                                                   stiff_min, stiff_max, eq_min, eq_max, antisymmetric=False):
        """
        Create sparse coupling matrix and equilibrium distance matrix in a single pass.
        
        Args:
            antisymmetric: If True, equilibrium distances are antisymmetric (negated for j,i)
        
        Returns:
            coupling_matrix: nn.Parameter sparse matrix
            eq_distances: Tensor of equilibrium distances
        """
        coupling = torch.zeros((n_units, n_units))
        eq_dist = torch.zeros((n_units, n_units))
        
        for i in range(n_units):
            for j in range(i + 1, n_units):
                max_conn = n_connections_anchor if i == 0 else n_connections
                if np.abs(i - j - 1) < max_conn:
                    # Generate both coupling stiffness and equilibrium distance
                    stiff = torch.rand(1).item() * (stiff_max - stiff_min) + stiff_min
                    eq = torch.rand(1).item() * (eq_max - eq_min) + eq_min
                    
                    coupling[i, j] = stiff
                    coupling[j, i] = stiff
                    
                    if antisymmetric:
                        eq_dist[i, j] = eq
                        eq_dist[j, i] = -eq
                    else:
                        eq_dist[i, j] = eq
                        eq_dist[j, i] = eq
        
        return nn.Parameter(coupling, requires_grad=False), eq_dist

        
    def forward(self, u_lin, u_ang, x, z, theta, s, omega):
        """
        Forward dynamics with optional velocity and acceleration capping.
        
        If use_capped_dynamics is True, applies max_speed and max_acceleration constraints.
        Otherwise, performs uncapped dynamics.
        
        Args:
            u_lin: Linear input forces (batch_size, n_units)
            u_ang: Angular input forces (batch_size, n_units)
            x, z: Position coordinates (batch_size, n_units)
            theta: Orientation (batch_size, n_units)
            s: Linear velocity (batch_size, n_units)
            omega: Angular velocity (batch_size, n_units)
            
        Returns:
            x, z, theta, s, omega: Updated states
        """
        bs = u_lin.shape[0]
        linear_inp_forces = u_lin
        
        # Compute 2D coordinates and heading vectors
        coords_2d = torch.stack((x, z), dim=-1)  # (b, n_units, 2)
        theta_unit_vectors = self.angle_to_unit_vector(theta)  # (b, n_units, 2)
        
        # Compute pairwise distance vectors
        distance_vectors = self.pairwise_differences(coords_2d)  # (b, n_units, n_units, 2)
        distance_magnitudes = torch.norm(distance_vectors, dim=-1, keepdim=True)  # (b, n_units, n_units, 1)
        distance_vectors_normalized = torch.nan_to_num(distance_vectors / distance_magnitudes)  # (b, n_units, n_units, 2)
        
        # Compute spring forces (before projection)
        forces_before_projection = self.stiffness_coupling_matrix[None, :, :, None] * (
            self.eq_distances_matrix - distance_magnitudes
        ) * distance_vectors_normalized  # (b, n_units, n_units, 2)
        
        # Project forces along the heading direction
        projected_forces = torch.einsum('bijk,bik->bi', forces_before_projection, theta_unit_vectors)  # (b, n_units)
        
        # Compute linear acceleration
        v_dot = (linear_inp_forces + projected_forces - (s * self.lin_damping)) * self.mass_vector
        
        # Apply acceleration capping if enabled
        if self.use_capped_dynamics:
            v_dot_magnitude = torch.abs(v_dot)
            v_dot = torch.where(
                v_dot_magnitude > self.max_acceleration,
                torch.sign(v_dot) * self.max_acceleration,
                v_dot
            )
        
        # Update velocity
        s = s + v_dot * self.dt
        
        # Apply velocity capping if enabled
        if self.use_capped_dynamics:
            s_magnitude = torch.abs(s)
            s = torch.where(
                s_magnitude > self.max_speed,
                torch.sign(s) * self.max_speed,
                s
            )
        
        # Angular dynamics
        inp_term_theta = u_ang
        theta_expanded_1 = theta[:, :, None]  # shape (b, n_units, 1)
        theta_expanded_2 = theta[:, None, :]  # shape (b, 1, n_units)
        ang_distances = theta_expanded_1 - theta_expanded_2
        coupling_term_ang = torch.sum(
            self.dist_ang_coupling[None, :, :] * (self.eq_distances_mat_ang.repeat(bs, 1, 1) - ang_distances),
            dim=2, keepdim=False
        )
        omega_dot = ((inp_term_theta + coupling_term_ang) - omega * self.ang_damping) * self.j_vector
        omega = omega + omega_dot * self.dt
        
        # Update positions and orientation
        theta = theta + self.dt * omega
        x = x + torch.cos(theta) * s * self.dt
        z = z + torch.sin(theta) * s * self.dt
        
        # # Add Gaussian noise to positions if enabled
        # if self.position_noise_std > 0:
        #     x = x + torch.randn_like(x) * self.position_noise_std
        #     z = z + torch.randn_like(z) * self.position_noise_std

        return x, z, theta, s, omega
    
    def get_force_breakdown(self, u_lin, u_ang, x, z, theta, s, omega):
        """
        Compute and return the breakdown of forces acting on each unit.
        
        Returns:
            dict with keys:
                - 'input_force': Force from external input (b, n_units)
                - 'spring_force': Force from inter-unit springs (b, n_units)
                - 'damping_force': Force from damping (b, n_units)
                - 'total_force': Total force (sum of above) (b, n_units)
                - 'input_torque': Torque from external input (b, n_units)
                - 'angular_coupling': Torque from angular coupling (b, n_units)
                - 'angular_damping': Torque from angular damping (b, n_units)
                - 'total_torque': Total torque (b, n_units)
        """
        bs = u_lin.shape[0]
        
        # Linear forces
        input_force = u_lin
        coords_2d = torch.stack((x, z), dim=-1)
        theta_unit_vectors = self.angle_to_unit_vector(theta)
        distance_vectors = self.pairwise_differences(coords_2d)
        distance_magnitudes = torch.norm(distance_vectors, dim=-1, keepdim=True)
        distance_vectors_normalized = torch.nan_to_num(distance_vectors / distance_magnitudes)
        forces_before_projection = self.stiffness_coupling_matrix[None, :, :, None] * (self.eq_distances_matrix - distance_magnitudes) * distance_vectors_normalized
        spring_force = torch.einsum('bijk,bik->bi', forces_before_projection, theta_unit_vectors)
        damping_force = -(s * self.lin_damping)
        total_force = input_force + spring_force + damping_force
        
        # Angular forces (torques)
        input_torque = u_ang
        theta_expanded_1 = theta[:, :, None]
        theta_expanded_2 = theta[:, None, :]
        ang_distances = theta_expanded_1 - theta_expanded_2
        angular_coupling = torch.sum(self.dist_ang_coupling[None, :, :] * (self.eq_distances_mat_ang.repeat(bs,1,1)-ang_distances), dim=2, keepdim=False)
        angular_damping = -(omega * self.ang_damping)
        total_torque = input_torque + angular_coupling + angular_damping
        
        return {
            'input_force': input_force.detach().cpu().numpy(),
            'spring_force': spring_force.detach().cpu().numpy(),
            'damping_force': damping_force.detach().cpu().numpy(),
            'total_force': total_force.detach().cpu().numpy(),
            'input_torque': input_torque.detach().cpu().numpy(),
            'angular_coupling': angular_coupling.detach().cpu().numpy(),
            'angular_damping': angular_damping.detach().cpu().numpy(),
            'total_torque': total_torque.detach().cpu().numpy(),
        }
    
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
    
    def set_eq_distances_from_positions(self, x, z):
        """
        Set equilibrium distances based on actual distances between connected robots.
        Only updates springs that have non-zero stiffness (i.e., actual connections).
        
        Args:
            x: numpy array or torch tensor of x positions, shape (n_units,)
            z: numpy array or torch tensor of z positions, shape (n_units,)
        """
        # Convert to torch tensors if needed
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if isinstance(z, np.ndarray):
            z = torch.tensor(z, dtype=torch.float32)
        
        # Compute pairwise distances
        n_units = len(x)
        for i in range(n_units):
            for j in range(i + 1, n_units):
                # Only update if there's actually a connection (non-zero stiffness)
                if self.stiffness_coupling_matrix[i, j] > 0:
                    # Calculate Euclidean distance
                    dist = torch.sqrt((x[i] - x[j])**2 + (z[i] - z[j])**2)
                    # Update equilibrium distance for this connection
                    self.eq_distances_matrix.data[i, j, 0] = dist
                    self.eq_distances_matrix.data[j, i, 0] = dist
        
        print(f"Updated equilibrium distances based on initial positions")
        print(f"  Distance range: [{self.eq_distances_matrix.data.min():.4f}, {self.eq_distances_matrix.data.max():.4f}]")

class UnicycleReservoir(nn.Module):
    def __init__(self, n_inp, n_units, dt, n_out, lin_stiff_min=0.1, lin_stiff_max=0.5, 
                 ang_stiff_min=0.1, ang_stiff_max=0.3, lin_damping_min=0.1, lin_damping_max=0.2,
                 ang_damping_min=0.1, ang_damping_max=0.2, eq_dist_min=0.5, eq_dist_max=1.0,
                 eq_dist_min_ang=0.0, eq_dist_max_ang=np.pi,
                 lin_input_map=None, ang_input_map=None, n_connections=None, inp_bias=0, n_connections_anchor=2, 
                 n_connections_ang=None, n_connections_anchor_ang=2, n_past_steps_readout=0,
                 use_capped_dynamics=False, max_speed=0.15, max_acceleration=1.0,
                 position_noise_std=0.0) -> None:
        super().__init__()
        self.n_inp = n_inp
        self.n_units = n_units
        self.unicycle_network = UnicycleNetwork(n_inp, n_units, dt, lin_stiff_min=lin_stiff_min, lin_stiff_max=lin_stiff_max, 
                 ang_stiff_min=ang_stiff_min, ang_stiff_max=ang_stiff_max, lin_damping_min=lin_damping_min, lin_damping_max=lin_damping_max,
                 ang_damping_min=ang_damping_min, ang_damping_max=ang_damping_max, eq_dist_min=eq_dist_min, eq_dist_max=eq_dist_max,
                 eq_dist_min_ang=eq_dist_min_ang, eq_dist_max_ang=eq_dist_max_ang,
                 lin_input_map=None, ang_input_map=None, n_connections=n_connections, n_connections_anchor=n_connections_anchor, 
                 n_connections_ang=n_connections_ang, n_connections_anchor_ang=n_connections_anchor_ang,
                 use_capped_dynamics=use_capped_dynamics, max_speed=max_speed, max_acceleration=max_acceleration,
                 position_noise_std=position_noise_std)
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
    
    def forward_with_forces(self, u_lin, u_ang):
        """
        Same as forward() but also returns force breakdown over time.
        
        Returns:
            states_list: List of states at each timestep
            output: Readout output (None in current implementation)
            mid_states: Mid-sequence states for readout
            force_history: List of force breakdown dicts at each timestep
        """
        x = self.x_init
        z = self.z_init
        theta = self.theta_init
        s = self.s_init
        omega = self.omega_init
        states_list = []
        force_history = []

        for t in range(u_lin.size()[1]):
            linear_input = (u_lin[:, t] + self.inp_bias) @ self.lin_input_map
            angular_input = (u_ang[:, t]) @ self.ang_input_map

            # Get force breakdown before updating state
            forces = self.unicycle_network.get_force_breakdown(linear_input, angular_input, x, z, theta, s, omega)
            force_history.append(forces)

            # Update state
            x, z, theta, s, omega = self.unicycle_network(linear_input, angular_input, x, z, theta, s, omega)

            concatenated_states = torch.hstack((x, z, theta, s, omega))
            states_list.append(concatenated_states)

        if self.n_past_steps_readout > 0:
            mid_states_idxs = [(int(u_lin.size()[1] / self.n_past_steps_readout) - 1)*k for k in range(1,self.n_past_steps_readout+1)]
            mid_states = torch.hstack(([states_list[idx] for idx in mid_states_idxs]))
        else:
            mid_states = states_list[-1]
        output = None
        
        return states_list, output, mid_states, force_history
    
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
    
    def set_eq_distances_from_initial_positions(self):
        """
        Set equilibrium distances based on the current initial positions.
        Must be called after set_init_states() or similar initialization.
        """
        if not hasattr(self, 'x_init') or not hasattr(self, 'z_init'):
            raise ValueError("Initial states not set. Call set_init_states() first.")
        
        # Extract first batch (all batches should have same initial positions)
        x = self.x_init[0].detach().cpu()
        z = self.z_init[0].detach().cpu()
        
        # Set equilibrium distances in the underlying network
        self.unicycle_network.set_eq_distances_from_positions(x, z)

