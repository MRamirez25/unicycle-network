import time
from torch import nn
import torch
import numpy as np
from unicycle_network import UnicycleNetwork


class DeepUnicycleReservoir(nn.Module):
    """
    A deep version of UnicycleReservoir with multiple stacked UnicycleNetworks.
    
    The positions (x, z) from one network are passed as input to the next network.
    A randomized linear projection matrix (scaled by linear_transform_scale) transforms 
    the concatenated x,z coordinates to the input dimension of the next layer.
    """
    
    def __init__(self, n_inp, n_units_per_layer, dt, n_out, n_layers=2,
                 lin_stiff_min=0.1, lin_stiff_max=0.5, 
                 ang_stiff_min=0.1, ang_stiff_max=0.3, 
                 lin_damping_min=0.1, lin_damping_max=0.2,
                 ang_damping_min=0.1, ang_damping_max=0.2, 
                 eq_dist_min=0.5, eq_dist_max=1.0,
                 eq_dist_min_ang=0.0, eq_dist_max_ang=np.pi,
                 n_connections=None, inp_bias=0, n_connections_anchor=2, 
                 n_connections_ang=None, n_connections_anchor_ang=2, 
                 n_past_steps_readout=0,
                 lin_input_map=None, ang_input_map=None,
                 use_linear_transform=True, linear_transform_scale=1.0,
                 use_angular_transform=True, angular_transform_scale=1.0,
                 concatenate_all_layers=False) -> None:
        """
        Args:
            n_inp: Number of input dimensions
            n_units_per_layer: List of integers specifying number of units for each layer
                              (e.g., [20, 20] for 2 layers with 20 units each)
            dt: Time step
            n_out: Number of output dimensions
            n_layers: Number of stacked layers (overridden if n_units_per_layer is provided)
            lin_input_map: Optional pre-computed linear input map for first layer
            ang_input_map: Optional pre-computed angular input map for first layer
            use_linear_transform: Whether to apply random projection transformations between layers
            linear_transform_scale: Scaling factor for the random projection matrices between layers
            use_angular_transform: Whether to apply separate transformations for angular states (theta)
            angular_transform_scale: Scaling factor for angular state transformations between layers
            concatenate_all_layers: If True, concatenate activations from all layers for readout;
                                    if False, use only final layer activations
            
            Network parameters can be single values (applied to all layers) or lists (per-layer):
            - lin_stiff_min/max, ang_stiff_min/max: Single value or list of length n_layers
            - lin_damping_min/max, ang_damping_min/max: Single value or list of length n_layers
            - eq_dist_min/max, eq_dist_min_ang/max: Single value or list of length n_layers
            - n_connections, n_connections_anchor: Single value or list of length n_layers
            - n_connections_ang, n_connections_anchor_ang: Single value or list of length n_layers
            
            Example for per-layer parameters with 2 layers:
            - lin_stiff_min=[0.1, 0.15] will use 0.1 for layer 0, 0.15 for layer 1
            - lin_stiff_min=0.1 will use 0.1 for both layers
        """
        super().__init__()
        
        # If n_units_per_layer is a single int, expand it
        if isinstance(n_units_per_layer, int):
            n_units_per_layer = [n_units_per_layer] * n_layers
        
        self.n_inp = n_inp
        self.n_layers = len(n_units_per_layer)
        self.n_units_per_layer = n_units_per_layer
        self.use_linear_transform = use_linear_transform
        self.linear_transform_scale = linear_transform_scale
        self.use_angular_transform = use_angular_transform
        self.angular_transform_scale = angular_transform_scale
        self.inp_bias = inp_bias
        self.n_past_steps_readout = n_past_steps_readout
        self.concatenate_all_layers = concatenate_all_layers
        
        # Helper function to normalize parameters to per-layer lists
        def _ensure_per_layer(param, param_name):
            """Convert single value or list to per-layer list of length n_layers."""
            if isinstance(param, (list, tuple)):
                if len(param) != self.n_layers:
                    raise ValueError(
                        f"Parameter '{param_name}' has length {len(param)} "
                        f"but n_layers={self.n_layers}. Expected {self.n_layers} values."
                    )
                return list(param)
            else:
                return [param] * self.n_layers
        
        # Normalize all parameters to per-layer lists
        lin_stiff_min_list = _ensure_per_layer(lin_stiff_min, 'lin_stiff_min')
        lin_stiff_max_list = _ensure_per_layer(lin_stiff_max, 'lin_stiff_max')
        ang_stiff_min_list = _ensure_per_layer(ang_stiff_min, 'ang_stiff_min')
        ang_stiff_max_list = _ensure_per_layer(ang_stiff_max, 'ang_stiff_max')
        lin_damping_min_list = _ensure_per_layer(lin_damping_min, 'lin_damping_min')
        lin_damping_max_list = _ensure_per_layer(lin_damping_max, 'lin_damping_max')
        ang_damping_min_list = _ensure_per_layer(ang_damping_min, 'ang_damping_min')
        ang_damping_max_list = _ensure_per_layer(ang_damping_max, 'ang_damping_max')
        eq_dist_min_list = _ensure_per_layer(eq_dist_min, 'eq_dist_min')
        eq_dist_max_list = _ensure_per_layer(eq_dist_max, 'eq_dist_max')
        eq_dist_min_ang_list = _ensure_per_layer(eq_dist_min_ang, 'eq_dist_min_ang')
        eq_dist_max_ang_list = _ensure_per_layer(eq_dist_max_ang, 'eq_dist_max_ang')
        n_connections_list = _ensure_per_layer(n_connections, 'n_connections')
        n_connections_anchor_list = _ensure_per_layer(n_connections_anchor, 'n_connections_anchor')
        n_connections_ang_list = _ensure_per_layer(n_connections_ang, 'n_connections_ang')
        n_connections_anchor_ang_list = _ensure_per_layer(n_connections_anchor_ang, 'n_connections_anchor_ang')
        
        # Create or use provided input maps for the first layer
        if lin_input_map is None:
            lin_input_map_layer0 = torch.rand(n_inp, n_units_per_layer[0])
            self.lin_input_map = nn.Parameter(lin_input_map_layer0, requires_grad=False)
        else:
            self.lin_input_map = lin_input_map
        
        if ang_input_map is None:
            ang_input_map_layer0 = torch.rand(n_inp, n_units_per_layer[0])
            self.ang_input_map = nn.Parameter(ang_input_map_layer0, requires_grad=False)
        else:
            self.ang_input_map = ang_input_map
        
        # Create UnicycleNetworks for each layer with per-layer parameters
        self.unicycle_layers = nn.ModuleList()
        for layer_idx in range(self.n_layers):
            unicycle_net = UnicycleNetwork(
                n_inp=n_inp,  # All layers receive the same input dimensions
                n_units=n_units_per_layer[layer_idx],
                dt=dt,
                lin_stiff_min=lin_stiff_min_list[layer_idx],
                lin_stiff_max=lin_stiff_max_list[layer_idx],
                ang_stiff_min=ang_stiff_min_list[layer_idx],
                ang_stiff_max=ang_stiff_max_list[layer_idx],
                lin_damping_min=lin_damping_min_list[layer_idx],
                lin_damping_max=lin_damping_max_list[layer_idx],
                ang_damping_min=ang_damping_min_list[layer_idx],
                ang_damping_max=ang_damping_max_list[layer_idx],
                eq_dist_min=eq_dist_min_list[layer_idx],
                eq_dist_max=eq_dist_max_list[layer_idx],
                eq_dist_min_ang=eq_dist_min_ang_list[layer_idx],
                eq_dist_max_ang=eq_dist_max_ang_list[layer_idx],
                n_connections=n_connections_list[layer_idx],
                n_connections_anchor=n_connections_anchor_list[layer_idx],
                n_connections_ang=n_connections_ang_list[layer_idx],
                n_connections_anchor_ang=n_connections_anchor_ang_list[layer_idx]
            )
            self.unicycle_layers.append(unicycle_net)
        
        # Create position transformation matrices between network layers
        if use_linear_transform and self.n_layers > 1:
            self.position_transforms = nn.ParameterList()
            for layer_idx in range(self.n_layers - 1):
                # Transform from layer i to layer i+1
                # Each layer outputs positions: x (n_units) and z (n_units)
                # Total position dimension: 2 * n_units_per_layer[layer_idx]
                in_dim = 2 * n_units_per_layer[layer_idx]  # x and z coordinates
                out_dim = n_units_per_layer[layer_idx + 1]
                
                # Create a random projection matrix scaled by linear_transform_scale
                transform_matrix = torch.randn(in_dim, out_dim) * linear_transform_scale
                
                self.position_transforms.append(nn.Parameter(transform_matrix, requires_grad=False))
        
        # Create angular transformation matrices between network layers
        # Angular states (theta) from one layer become angular input to the next
        if use_angular_transform and self.n_layers > 1:
            self.angular_transforms = nn.ParameterList()
            for layer_idx in range(self.n_layers - 1):
                # Transform from layer i to layer i+1
                # Each layer outputs angular state: theta (n_units)
                in_dim = n_units_per_layer[layer_idx]  # theta only
                out_dim = n_units_per_layer[layer_idx + 1]
                
                # Create a random projection matrix scaled by angular_transform_scale
                transform_matrix = torch.randn(in_dim, out_dim) * angular_transform_scale
                
                self.angular_transforms.append(nn.Parameter(transform_matrix, requires_grad=False))
        
        # Output readout layer - compute input size based on concatenate_all_layers flag
        if concatenate_all_layers:
            # Concatenate activations from all layers (each has 5 components: x, z, theta, s, omega)
            readout_input_size = sum(n_units * 5 for n_units in n_units_per_layer) * (n_past_steps_readout + 1)
        else:
            # Use only final layer activations
            readout_input_size = n_units_per_layer[-1] * 5 * (n_past_steps_readout + 1)
        
        self.readout = nn.Linear(readout_input_size, n_out)
    
    def forward(self, u_lin, u_ang):
        """
        Forward pass through the deep network.
        
        Args:
            u_lin: Linear input forces, shape (batch_size, time_steps, n_inp)
            u_ang: Angular input, shape (batch_size, time_steps, n_inp)
        
        Returns:
            states_list: List of state concatenations at each timestep
                        - If concatenate_all_layers is False: states from final layer only
                        - If concatenate_all_layers is True: states concatenated from all layers
            output: Readout output (None if not computing loss)
            mid_states: Mid-layer states for readout
        """
        batch_size = u_lin.size(0)
        time_steps = u_lin.size(1)
        
        # Initialize states for all layers
        layer_states = []
        for layer_idx in range(self.n_layers):
            layer_state = {
                'x': self.x_init_per_layer[layer_idx],
                'z': self.z_init_per_layer[layer_idx],
                'theta': self.theta_init_per_layer[layer_idx],
                's': self.s_init_per_layer[layer_idx],
                'omega': self.omega_init_per_layer[layer_idx],
            }
            layer_states.append(layer_state)
        
        final_states_list = []
        
        for t in range(time_steps):
            # Process through each layer in sequence
            layer_inputs = []
            
            # First layer gets the original input
            lin_input_layer0 = (u_lin[:, t] + self.inp_bias) @ self.lin_input_map
            ang_input_layer0 = (u_ang[:, t]) @ self.ang_input_map
            layer_inputs.append((lin_input_layer0, ang_input_layer0))
            
            # Process first layer
            state = layer_states[0]
            x, z, theta, s, omega = self.unicycle_layers[0](
                lin_input_layer0, ang_input_layer0,
                state['x'], state['z'], state['theta'], state['s'], state['omega']
            )
            layer_states[0]['x'] = x
            layer_states[0]['z'] = z
            layer_states[0]['theta'] = theta
            layer_states[0]['s'] = s
            layer_states[0]['omega'] = omega
            
            # Process subsequent layers
            for layer_idx in range(1, self.n_layers):
                # Extract output from previous layer: positions (x, z) and angles
                prev_layer_state = layer_states[layer_idx - 1]
                
                # Concatenate x and z positions for linear input
                layer_output = torch.cat([prev_layer_state['x'], prev_layer_state['z']], dim=-1)
                
                # Apply position transformation (random projection)
                if self.use_linear_transform:
                    lin_input = layer_output @ self.position_transforms[layer_idx - 1]
                else:
                    lin_input = layer_output
                
                # Apply angular transformation to theta for angular input
                if self.use_angular_transform:
                    ang_input = prev_layer_state['theta'] @ self.angular_transforms[layer_idx - 1]
                else:
                    ang_input = lin_input  # Fallback to linear input if no angular transform
                
                # Process through this layer
                state = layer_states[layer_idx]
                x, z, theta, s, omega = self.unicycle_layers[layer_idx](
                    lin_input, ang_input,
                    state['x'], state['z'], state['theta'], state['s'], state['omega']
                )
                layer_states[layer_idx]['x'] = x
                layer_states[layer_idx]['z'] = z
                layer_states[layer_idx]['theta'] = theta
                layer_states[layer_idx]['s'] = s
                layer_states[layer_idx]['omega'] = omega
            
            # Collect states based on concatenate_all_layers flag
            if self.concatenate_all_layers:
                # Concatenate activations from all layers
                all_activations = []
                for layer_idx in range(self.n_layers):
                    layer_state = layer_states[layer_idx]
                    concatenated = torch.hstack((
                        layer_state['x'], layer_state['z'], layer_state['theta'],
                        layer_state['s'], layer_state['omega']
                    ))
                    all_activations.append(concatenated)
                concatenated_states = torch.hstack(all_activations)
            else:
                # Use only final layer state (original behavior)
                final_state = layer_states[-1]
                concatenated_states = torch.hstack((
                    final_state['x'], final_state['z'], final_state['theta'],
                    final_state['s'], final_state['omega']
                ))
            
            final_states_list.append(concatenated_states)
        
        # Prepare output
        if self.n_past_steps_readout > 0:
            mid_states_idxs = [
                (int(time_steps / self.n_past_steps_readout) - 1) * k 
                for k in range(1, self.n_past_steps_readout + 1)
            ]
            mid_states = torch.hstack([final_states_list[idx] for idx in mid_states_idxs])
        else:
            mid_states = final_states_list[-1]
        
        output = None
        
        return final_states_list, output, mid_states
    
    def set_init_states_random(self, bs):
        """Initialize all layers with random states."""
        self.x_init_per_layer = []
        self.z_init_per_layer = []
        self.theta_init_per_layer = []
        self.s_init_per_layer = []
        self.omega_init_per_layer = []
        
        for layer_idx in range(self.n_layers):
            n_units = self.n_units_per_layer[layer_idx]
            self.x_init_per_layer.append(torch.randn(n_units).repeat(bs, 1))
            self.z_init_per_layer.append(torch.randn(n_units).repeat(bs, 1))
            self.theta_init_per_layer.append(torch.randn(n_units).repeat(bs, 1))
            self.s_init_per_layer.append(torch.randn(n_units).repeat(bs, 1))
            self.omega_init_per_layer.append(torch.randn(n_units).repeat(bs, 1))
    
    def set_init_states_grid(self, bs, num_rows, num_cols, spacing):
        """
        Initialize unit positions on a grid for all layers.
        
        Args:
            bs: Batch size
            num_rows: Number of rows in the grid
            num_cols: Number of columns in the grid
            spacing: Tuple of (x_spacing, z_spacing)
        """
        x_spacing, z_spacing = spacing
        
        x_coords = np.tile(np.arange(num_cols) * x_spacing, num_rows)
        z_coords = np.repeat(np.arange(num_rows) * z_spacing, num_cols)
        
        self.x_init_per_layer = []
        self.z_init_per_layer = []
        self.theta_init_per_layer = []
        self.s_init_per_layer = []
        self.omega_init_per_layer = []
        
        for layer_idx in range(self.n_layers):
            n_units = self.n_units_per_layer[layer_idx]
            
            # Ensure we have enough grid points
            assert len(x_coords) >= n_units, f"Grid too small for layer {layer_idx} with {n_units} units"
            
            layer_x_coords = x_coords[:n_units]
            layer_z_coords = z_coords[:n_units]
            
            self.x_init_per_layer.append(
                torch.tensor(layer_x_coords, dtype=torch.float32).repeat(bs, 1)
            )
            self.z_init_per_layer.append(
                torch.tensor(layer_z_coords, dtype=torch.float32).repeat(bs, 1)
            )
            self.theta_init_per_layer.append(torch.randn(n_units).repeat(bs, 1))
            self.s_init_per_layer.append(torch.randn(n_units).repeat(bs, 1))
            self.omega_init_per_layer.append(torch.randn(n_units).repeat(bs, 1))
    
    def set_init_states(self, bs, x_per_layer, z_per_layer, theta_per_layer, s_per_layer, omega_per_layer):
        """
        Set initial states for all layers.
        
        Args:
            bs: Batch size
            x_per_layer: List of x coordinate arrays for each layer
            z_per_layer: List of z coordinate arrays for each layer
            theta_per_layer: List of theta arrays for each layer
            s_per_layer: List of s (velocity) arrays for each layer
            omega_per_layer: List of omega arrays for each layer
        """
        self.x_init_per_layer = []
        self.z_init_per_layer = []
        self.theta_init_per_layer = []
        self.s_init_per_layer = []
        self.omega_init_per_layer = []
        
        for layer_idx in range(self.n_layers):
            self.x_init_per_layer.append(torch.tensor(x_per_layer[layer_idx]).repeat(bs, 1))
            self.z_init_per_layer.append(torch.tensor(z_per_layer[layer_idx]).repeat(bs, 1))
            self.theta_init_per_layer.append(torch.tensor(theta_per_layer[layer_idx]).repeat(bs, 1))
            self.s_init_per_layer.append(torch.tensor(s_per_layer[layer_idx]).repeat(bs, 1))
            self.omega_init_per_layer.append(torch.tensor(omega_per_layer[layer_idx]).repeat(bs, 1))
    
    def set_eq_distances_from_initial_positions(self):
        """
        Set equilibrium distances based on initial positions for all layers.
        """
        for layer_idx in range(self.n_layers):
            x = self.x_init_per_layer[layer_idx][0].detach().cpu()
            z = self.z_init_per_layer[layer_idx][0].detach().cpu()
            self.unicycle_layers[layer_idx].set_eq_distances_from_positions(x, z)
