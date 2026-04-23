#!/usr/bin/env python3
"""
Unicycle oscillator dynamics WITH non-holonomic constraint.

Each oscillator has state: [x, z, theta]
- x, z: 2D position
- theta: orientation angle

Non-holonomic constraint: velocity must be along heading direction
- dx/dt = v * cos(theta)
- dz/dt = v * sin(theta)
- dtheta/dt = omega

Where v is forward velocity control and omega is angular velocity control.

No potential coupling: no springs connecting oscillators
"""

import numpy as np
from scipy.integrate import odeint
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class UnicycleParams:
    """Parameters for a single unicycle oscillator."""
    mass: float = 1.0              # Effective mass
    damping: float = 0.5           # Linear damping on velocity
    angular_damping: float = 0.5   # Rotational damping on omega
    max_velocity: float = 5.0      # Maximum forward velocity


class UnicycleOscillatorNetwork:
    """
    Network of unicycle oscillators with non-holonomic constraint.
    
    State vector: [x1, z1, theta1, v1, omega1, x2, z2, theta2, v2, omega2, ...]
    
    Constraint: dx/dt = v*cos(theta), dz/dt = v*sin(theta)
    """
    
    def __init__(self, 
                 n_oscillators: int,
                 params: Optional[UnicycleParams] = None,
                 forcing_fn: Optional[Callable] = None):
        """
        Initialize the network.
        
        Args:
            n_oscillators: Number of oscillators
            params: Parameters for each oscillator
            forcing_fn: Function(t, oscillator_idx) -> (v_control, omega_control)
                       Returns desired velocity and angular velocity
        """
        self.n_oscillators = n_oscillators
        self.params = params or UnicycleParams()
        self.forcing_fn = forcing_fn
        
        # State: [x, z, theta, v, omega] for each oscillator
        self.state_dim = 5
        self.total_state_dim = n_oscillators * self.state_dim
    
    def state_to_positions_and_angles(self, state: np.ndarray) -> tuple:
        """Extract positions and angles from state."""
        positions = []
        angles = []
        for i in range(self.n_oscillators):
            idx = i * self.state_dim
            positions.append(np.array([state[idx], state[idx + 1]]))
            angles.append(state[idx + 2])
        return np.array(positions), np.array(angles)
    
    def dynamics(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        Compute state derivatives with non-holonomic constraint.
        
        Args:
            state: Current state vector [x, z, theta, v, omega, ...]
            t: Current time
            
        Returns:
            State derivatives
        """
        state_dot = np.zeros_like(state)
        
        for i in range(self.n_oscillators):
            idx = i * self.state_dim
            
            # Current state
            x, z, theta, v, omega = state[idx:idx+5]
            
            # Non-holonomic constraint: velocity along heading
            # dx/dt = v * cos(theta)
            # dz/dt = v * sin(theta)
            state_dot[idx] = v * np.cos(theta)
            state_dot[idx + 1] = v * np.sin(theta)
            state_dot[idx + 2] = omega
            
            # Get velocity control inputs
            if self.forcing_fn is not None:
                v_control, omega_control = self.forcing_fn(t, i)
            else:
                v_control, omega_control = 0.0, 0.0
            
            # First-order dynamics for velocity and angular velocity
            # with damping: dv/dt = -damping*v + control
            # Do dynamics as differential equations to smooth the response
            dv_dt = -self.params.damping * v + v_control
            domega_dt = -self.params.angular_damping * omega + omega_control
            
            state_dot[idx + 3] = dv_dt
            state_dot[idx + 4] = domega_dt
        
        return state_dot
    
    def initialize_state(self, 
                        positions: np.ndarray,
                        angles: np.ndarray,
                        velocities: Optional[np.ndarray] = None,
                        angular_velocities: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Initialize state vector.
        
        Args:
            positions: (n_oscillators, 2) array of [x, z] positions
            angles: (n_oscillators,) array of theta values
            velocities: (n_oscillators,) array of v values (default: zeros)
            angular_velocities: (n_oscillators,) array of omega values (default: zeros)
            
        Returns:
            State vector
        """
        if velocities is None:
            velocities = np.zeros(self.n_oscillators)
        if angular_velocities is None:
            angular_velocities = np.zeros(self.n_oscillators)
        
        state = np.zeros(self.total_state_dim)
        for i in range(self.n_oscillators):
            idx = i * self.state_dim
            state[idx:idx+2] = positions[i]
            state[idx+2] = angles[i]
            state[idx+3] = velocities[i]
            state[idx+4] = angular_velocities[i]
        
        return state
    
    def simulate(self, 
                 initial_state: np.ndarray,
                 t_span: np.ndarray) -> dict:
        """
        Simulate the network dynamics.
        
        Args:
            initial_state: Initial state vector
            t_span: Time points for simulation
            
        Returns:
            Dictionary with:
                't': time points
                'state': full state trajectory
                'positions': (n_time, n_oscillators, 2) positions
                'angles': (n_time, n_oscillators) angles
                'velocities': (n_time, n_oscillators) forward velocities
                'angular_velocities': (n_time, n_oscillators) angular velocities
        """
        # Use odeint for compatibility
        solution = odeint(self.dynamics, initial_state, t_span, full_output=False)
        
        # Extract positions, angles, and velocities
        positions = []
        angles = []
        velocities = []
        angular_velocities = []
        
        for state in solution:
            pos, ang = self.state_to_positions_and_angles(state)
            positions.append(pos)
            angles.append(ang)
            
            # Extract velocities
            vel = []
            ang_vel = []
            for i in range(self.n_oscillators):
                idx = i * self.state_dim
                vel.append(state[idx+3])
                ang_vel.append(state[idx+4])
            velocities.append(np.array(vel))
            angular_velocities.append(np.array(ang_vel))
        
        return {
            't': t_span,
            'state': solution,
            'positions': np.array(positions),
            'angles': np.array(angles),
            'velocities': np.array(velocities),
            'angular_velocities': np.array(angular_velocities)
        }


def create_circular_forcing(amplitude: float = 1.0, frequency: float = 1.0):
    """
    Create a forcing function for circular motion.
    
    Args:
        amplitude: Force amplitude
        frequency: Forcing frequency
        
    Returns:
        Forcing function that returns (v_control, omega_control)
    """
    def forcing_fn(t: float, oscillator_idx: int) -> tuple:
        # Phase offset for different oscillators
        phase_offset = (oscillator_idx * 2 * np.pi) / 10
        
        # Sinusoidal velocity and angular velocity
        v_control = amplitude * np.cos(2 * np.pi * frequency * t + phase_offset)
        omega_control = amplitude * 0.5 * np.sin(2 * np.pi * frequency * t + phase_offset)
        
        return v_control, omega_control
    
    return forcing_fn


def create_sinusoidal_forcing(amplitude: float = 1.0, frequency: float = 1.0):
    """
    Create a forcing function with sinusoidal velocity control.
    
    Args:
        amplitude: Force amplitude
        frequency: Forcing frequency
        
    Returns:
        Forcing function that returns (v_control, omega_control)
    """
    def forcing_fn(t: float, oscillator_idx: int) -> tuple:
        phase_offset = (oscillator_idx * 2 * np.pi) / 10
        
        v_control = amplitude * np.sin(2 * np.pi * frequency * t + phase_offset)
        omega_control = amplitude * 0.5 * np.cos(2 * np.pi * frequency * t + phase_offset)
        
        return v_control, omega_control
    
    return forcing_fn


if __name__ == '__main__':
    # Example: simulate 10 unicycle oscillators
    n_oscillators = 10
    
    # Create network
    params = UnicycleParams(
        mass=1.0,
        damping=0.5,
        angular_damping=0.3
    )
    
    forcing = create_circular_forcing(amplitude=1.0, frequency=0.5)
    network = UnicycleOscillatorNetwork(n_oscillators, params, forcing)
    
    # Initialize state (circle of oscillators)
    radius = 2.0
    angles_init = np.linspace(0, 2*np.pi, n_oscillators, endpoint=False)
    positions_init = np.array([[radius * np.cos(a), radius * np.sin(a)] for a in angles_init])
    angles_init = angles_init + np.pi/2  # Face outward
    
    initial_state = network.initialize_state(positions_init, angles_init)
    
    # Simulate
    t_span = np.linspace(0, 10, 1000)
    print("Simulating unicycle oscillator network...")
    result = network.simulate(initial_state, t_span)
    
    print(f"Simulation complete!")
    print(f"  Time points: {len(result['t'])}")
    print(f"  Oscillators: {n_oscillators}")
    print(f"  Final positions shape: {result['positions'].shape}")
    print(f"  Final angles shape: {result['angles'].shape}")
