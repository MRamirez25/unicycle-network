#!/usr/bin/env python3
"""
Generic oscillator dynamics WITHOUT non-holonomic constraint.

Each oscillator has state: [x, z, theta]
- x, z: 2D position (can move freely in any direction)
- theta: orientation angle

No non-holonomic constraint: can move sideways
No potential coupling: no springs connecting oscillators
"""

import numpy as np
from scipy.integrate import odeint
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class OscillatorParams:
    """Parameters for a single oscillator."""
    mass: float = 1.0              # Effective mass
    damping: float = 0.5           # Linear damping coefficient
    angular_damping: float = 0.5   # Rotational damping
    forcing_amplitude: float = 1.0 # Amplitude of forcing
    max_velocity: float = 5.0      # Maximum velocity magnitude


class GenericOscillatorNetwork:
    """
    Network of generic oscillators without non-holonomic constraints.
    
    State vector: [x1, z1, theta1, vx1, vz1, omega1, x2, z2, theta2, vx2, vz2, omega2, ...]
    """
    
    def __init__(self, 
                 n_oscillators: int,
                 params: Optional[OscillatorParams] = None,
                 forcing_fn: Optional[Callable] = None):
        """
        Initialize the network.
        
        Args:
            n_oscillators: Number of oscillators
            params: Parameters for each oscillator (same for all if single)
            forcing_fn: Function(t, oscillator_idx) -> (force_x, force_z, torque)
                       If None, no forcing is applied
        """
        self.n_oscillators = n_oscillators
        self.params = params or OscillatorParams()
        self.forcing_fn = forcing_fn
        
        # State: [x, z, theta, vx, vz, omega] for each oscillator
        self.state_dim = 6
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
        Compute state derivatives.
        
        Args:
            state: Current state vector
            t: Current time
            
        Returns:
            State derivatives
        """
        state_dot = np.zeros_like(state)
        
        for i in range(self.n_oscillators):
            idx = i * self.state_dim
            
            # Current state
            x, z, theta = state[idx:idx+3]
            vx, vz, omega = state[idx+3:idx+6]
            
            # Position derivatives (velocity)
            state_dot[idx] = vx
            state_dot[idx + 1] = vz
            state_dot[idx + 2] = omega
            
            # Get forcing
            if self.forcing_fn is not None:
                force_x, force_z, torque = self.forcing_fn(t, i)
            else:
                force_x, force_z, torque = 0.0, 0.0, 0.0
            
            # Velocity derivatives (acceleration)
            # F = ma - damping
            ax = (force_x - self.params.damping * vx) / self.params.mass
            az = (force_z - self.params.damping * vz) / self.params.mass
            
            # Angular acceleration
            alpha = (torque - self.params.angular_damping * omega)
            
            state_dot[idx + 3] = ax
            state_dot[idx + 4] = az
            state_dot[idx + 5] = alpha
        
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
            velocities: (n_oscillators, 2) array of [vx, vz] velocities (default: zeros)
            angular_velocities: (n_oscillators,) array of omega values (default: zeros)
            
        Returns:
            State vector
        """
        if velocities is None:
            velocities = np.zeros_like(positions)
        if angular_velocities is None:
            angular_velocities = np.zeros(self.n_oscillators)
        
        state = np.zeros(self.total_state_dim)
        for i in range(self.n_oscillators):
            idx = i * self.state_dim
            state[idx:idx+2] = positions[i]
            state[idx+2] = angles[i]
            state[idx+3:idx+5] = velocities[i]
            state[idx+5] = angular_velocities[i]
        
        return state
    
    def simulate(self, 
                 initial_state: np.ndarray,
                 t_span: np.ndarray,
                 method: str = 'RK45') -> dict:
        """
        Simulate the network dynamics.
        
        Args:
            initial_state: Initial state vector
            t_span: Time points for simulation
            method: Integration method (see scipy.integrate.odeint)
            
        Returns:
            Dictionary with:
                't': time points
                'state': full state trajectory
                'positions': (n_time, n_oscillators, 2) positions
                'angles': (n_time, n_oscillators) angles
                'velocities': (n_time, n_oscillators, 2) velocities
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
                vel.append(state[idx+3:idx+5])
                ang_vel.append(state[idx+5])
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
    Create a forcing function that applies circular motion forces.
    
    Args:
        amplitude: Force amplitude
        frequency: Forcing frequency
        
    Returns:
        Forcing function
    """
    def forcing_fn(t: float, oscillator_idx: int) -> tuple:
        # Phase offset for different oscillators
        phase_offset = (oscillator_idx * 2 * np.pi) / 10  # Distribute phases
        
        # Circular forcing in x-z plane
        force_x = amplitude * np.cos(2 * np.pi * frequency * t + phase_offset)
        force_z = amplitude * np.sin(2 * np.pi * frequency * t + phase_offset)
        
        # Oscillating torque
        torque = amplitude * 0.5 * np.sin(2 * np.pi * frequency * t + phase_offset)
        
        return force_x, force_z, torque
    
    return forcing_fn


def create_sinusoidal_forcing(amplitude: float = 1.0, frequency: float = 1.0):
    """
    Create a forcing function with sinusoidal x-forcing.
    
    Args:
        amplitude: Force amplitude
        frequency: Forcing frequency
        
    Returns:
        Forcing function
    """
    def forcing_fn(t: float, oscillator_idx: int) -> tuple:
        phase_offset = (oscillator_idx * 2 * np.pi) / 10
        
        force_x = amplitude * np.sin(2 * np.pi * frequency * t + phase_offset)
        force_z = 0.0
        torque = 0.5 * amplitude * np.cos(2 * np.pi * frequency * t + phase_offset)
        
        return force_x, force_z, torque
    
    return forcing_fn


if __name__ == '__main__':
    # Example: simulate 10 oscillators
    n_oscillators = 10
    
    # Create network
    params = OscillatorParams(
        mass=1.0,
        damping=0.5,
        angular_damping=0.3,
        forcing_amplitude=1.0
    )
    
    forcing = create_circular_forcing(amplitude=1.0, frequency=0.5)
    network = GenericOscillatorNetwork(n_oscillators, params, forcing)
    
    # Initialize state (circle of oscillators)
    radius = 2.0
    angles_init = np.linspace(0, 2*np.pi, n_oscillators, endpoint=False)
    positions_init = np.array([[radius * np.cos(a), radius * np.sin(a)] for a in angles_init])
    angles_init = angles_init + np.pi/2  # Face outward
    
    initial_state = network.initialize_state(positions_init, angles_init)
    
    # Simulate
    t_span = np.linspace(0, 10, 1000)
    print("Simulating generic oscillator network...")
    result = network.simulate(initial_state, t_span)
    
    print(f"Simulation complete!")
    print(f"  Time points: {len(result['t'])}")
    print(f"  Oscillators: {n_oscillators}")
    print(f"  Final positions shape: {result['positions'].shape}")
    print(f"  Final angles shape: {result['angles'].shape}")
