#!/usr/bin/env python3
"""
Compare and visualize generic oscillators vs unicycle oscillators.

Shows the evolution of both systems with the same forcing in 3D space (x, z, theta).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Add parent directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from oscillator_dynamics import (
    GenericOscillatorNetwork, OscillatorParams, create_circular_forcing
)
from unicycle_dynamics import (
    UnicycleOscillatorNetwork, UnicycleParams
)


def run_comparison(n_oscillators=10, duration=10.0, n_frames=1000, amplitude=1.0, frequency=0.5):
    """
    Run both simulations with the same forcing and return results.
    
    Args:
        n_oscillators: Number of oscillators
        duration: Simulation duration in seconds
        n_frames: Number of time steps
        amplitude: Forcing amplitude
        frequency: Forcing frequency
        
    Returns:
        Dictionary with results from both systems
    """
    
    print("="*80)
    print("COMPARING GENERIC vs UNICYCLE OSCILLATORS")
    print("="*80)
    
    # Time vector
    t_span = np.linspace(0, duration, n_frames)
    
    # Initial conditions (same for both)
    radius = 2.0
    angles_init = np.linspace(0, 2*np.pi, n_oscillators, endpoint=False)
    positions_init = np.array([[radius * np.cos(a), radius * np.sin(a)] for a in angles_init])
    angles_init = angles_init + np.pi/2  # Face outward
    
    print(f"\nInitialization:")
    print(f"  Oscillators: {n_oscillators}")
    print(f"  Duration: {duration}s")
    print(f"  Time steps: {n_frames}")
    print(f"  Initial radius: {radius}m")
    print(f"  Forcing: circular, amplitude={amplitude}, frequency={frequency}")
    
    # Create forcing function
    # We need to create a special forcing that works for both systems
    def generic_forcing_fn(t, idx):
        """Forcing for generic oscillators (x, z, torque)."""
        phase_offset = (idx * 2 * np.pi) / n_oscillators
        force_x = amplitude * np.cos(2 * np.pi * frequency * t + phase_offset)
        force_z = amplitude * np.sin(2 * np.pi * frequency * t + phase_offset)
        torque = amplitude * 0.5 * np.sin(2 * np.pi * frequency * t + phase_offset)
        return force_x, force_z, torque
    
    def unicycle_forcing_fn(t, idx):
        """Forcing for unicycle oscillators (v, omega)."""
        phase_offset = (idx * 2 * np.pi) / n_oscillators
        v_control = amplitude * np.cos(2 * np.pi * frequency * t + phase_offset)
        omega_control = amplitude * 0.5 * np.sin(2 * np.pi * frequency * t + phase_offset)
        return v_control, omega_control
    
    # Generic oscillators
    print("\nSimulating generic oscillators (unconstrained)...")
    generic_params = OscillatorParams(
        mass=1.0,
        damping=0.5,
        angular_damping=0.3,
        forcing_amplitude=amplitude
    )
    generic_network = GenericOscillatorNetwork(n_oscillators, generic_params, generic_forcing_fn)
    generic_state_init = generic_network.initialize_state(positions_init, angles_init)
    generic_result = generic_network.simulate(generic_state_init, t_span)
    print(f"  ✓ Generic simulation complete")
    
    # Unicycle oscillators
    print("\nSimulating unicycle oscillators (non-holonomic constraint)...")
    unicycle_params = UnicycleParams(
        mass=1.0,
        damping=0.5,
        angular_damping=0.3
    )
    unicycle_network = UnicycleOscillatorNetwork(n_oscillators, unicycle_params, unicycle_forcing_fn)
    unicycle_state_init = unicycle_network.initialize_state(positions_init, angles_init)
    unicycle_result = unicycle_network.simulate(unicycle_state_init, t_span)
    print(f"  ✓ Unicycle simulation complete")
    
    return {
        'generic': generic_result,
        'unicycle': unicycle_result,
        'n_oscillators': n_oscillators,
        'duration': duration,
    }


def plot_static_comparison(results, oscillator_indices=[0, 1, 2]):
    """
    Create static plots comparing the two systems.
    
    Args:
        results: Results dictionary from run_comparison
        oscillator_indices: Which oscillators to plot
    """
    
    generic_result = results['generic']
    unicycle_result = results['unicycle']
    
    fig = plt.figure(figsize=(16, 12))
    
    for plot_idx, osc_idx in enumerate(oscillator_indices):
        # 3D trajectory plot
        ax = fig.add_subplot(3, len(oscillator_indices), plot_idx + 1, projection='3d')
        
        # Generic trajectory
        gen_pos = generic_result['positions'][:, osc_idx, :]
        gen_angle = generic_result['angles'][:, osc_idx]
        ax.plot(gen_pos[:, 0], gen_pos[:, 1], gen_angle, 'b-', label='Generic', alpha=0.7, linewidth=2)
        ax.scatter(gen_pos[0, 0], gen_pos[0, 1], gen_angle[0], color='blue', s=100, marker='o', label='Generic start')
        ax.scatter(gen_pos[-1, 0], gen_pos[-1, 1], gen_angle[-1], color='blue', s=100, marker='s')
        
        # Unicycle trajectory
        uni_pos = unicycle_result['positions'][:, osc_idx, :]
        uni_angle = unicycle_result['angles'][:, osc_idx]
        ax.plot(uni_pos[:, 0], uni_pos[:, 1], uni_angle, 'r-', label='Unicycle', alpha=0.7, linewidth=2)
        ax.scatter(uni_pos[0, 0], uni_pos[0, 1], uni_angle[0], color='red', s=100, marker='o', label='Unicycle start')
        ax.scatter(uni_pos[-1, 0], uni_pos[-1, 1], uni_angle[-1], color='red', s=100, marker='s')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_zlabel('θ (rad)')
        ax.set_title(f'Oscillator {osc_idx} in (x, z, θ) space')
        if plot_idx == 0:
            ax.legend(loc='upper left', fontsize=8)
        
        # X-Z position plot
        ax = fig.add_subplot(3, len(oscillator_indices), len(oscillator_indices) + plot_idx + 1)
        ax.plot(gen_pos[:, 0], gen_pos[:, 1], 'b-', label='Generic', alpha=0.7, linewidth=2)
        ax.plot(uni_pos[:, 0], uni_pos[:, 1], 'r-', label='Unicycle', alpha=0.7, linewidth=2)
        ax.scatter(gen_pos[0, 0], gen_pos[0, 1], color='blue', s=100, marker='o')
        ax.scatter(uni_pos[0, 0], uni_pos[0, 1], color='red', s=100, marker='o')
        ax.scatter(gen_pos[-1, 0], gen_pos[-1, 1], color='blue', s=100, marker='s')
        ax.scatter(uni_pos[-1, 0], uni_pos[-1, 1], color='red', s=100, marker='s')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_title(f'Oscillator {osc_idx} XZ-plane')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        if plot_idx == 0:
            ax.legend(fontsize=8)
        
        # Angle evolution
        ax = fig.add_subplot(3, len(oscillator_indices), 2*len(oscillator_indices) + plot_idx + 1)
        ax.plot(generic_result['t'], gen_angle, 'b-', label='Generic', alpha=0.7, linewidth=2)
        ax.plot(unicycle_result['t'], uni_angle, 'r-', label='Unicycle', alpha=0.7, linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('θ (rad)')
        ax.set_title(f'Oscillator {osc_idx} orientation')
        ax.grid(True, alpha=0.3)
        if plot_idx == 0:
            ax.legend(fontsize=8)
    
    plt.tight_layout()
    return fig


def plot_system_overview(results):
    """
    Create overview plots of the full system evolution.
    
    Args:
        results: Results dictionary from run_comparison
    """
    
    generic_result = results['generic']
    unicycle_result = results['unicycle']
    n_osc = results['n_oscillators']
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Generic XZ trajectories
    ax = axes[0, 0]
    for i in range(n_osc):
        pos = generic_result['positions'][:, i, :]
        ax.plot(pos[:, 0], pos[:, 1], 'b-', alpha=0.3)
    gen_pos_final = generic_result['positions'][-1, :, :]
    ax.scatter(gen_pos_final[:, 0], gen_pos_final[:, 1], color='blue', s=50, label='Generic final')
    gen_pos_init = generic_result['positions'][0, :, :]
    ax.scatter(gen_pos_init[:, 0], gen_pos_init[:, 1], color='lightblue', s=50, marker='x', label='Generic init')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('Generic Oscillators - XZ Plane')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend()
    
    # Unicycle XZ trajectories
    ax = axes[0, 1]
    for i in range(n_osc):
        pos = unicycle_result['positions'][:, i, :]
        ax.plot(pos[:, 0], pos[:, 1], 'r-', alpha=0.3)
    uni_pos_final = unicycle_result['positions'][-1, :, :]
    ax.scatter(uni_pos_final[:, 0], uni_pos_final[:, 1], color='red', s=50, label='Unicycle final')
    uni_pos_init = unicycle_result['positions'][0, :, :]
    ax.scatter(uni_pos_init[:, 0], uni_pos_init[:, 1], color='lightcoral', s=50, marker='x', label='Unicycle init')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z (m)')
    ax.set_title('Unicycle Oscillators - XZ Plane')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend()
    
    # Distance from origin over time
    ax = axes[0, 2]
    gen_distances = np.sqrt(np.sum(generic_result['positions']**2, axis=2))
    uni_distances = np.sqrt(np.sum(unicycle_result['positions']**2, axis=2))
    ax.plot(generic_result['t'], np.mean(gen_distances, axis=1), 'b-', linewidth=2, label='Generic (mean)')
    ax.plot(unicycle_result['t'], np.mean(uni_distances, axis=1), 'r-', linewidth=2, label='Unicycle (mean)')
    ax.fill_between(generic_result['t'], np.min(gen_distances, axis=1), np.max(gen_distances, axis=1), 
                     color='blue', alpha=0.2)
    ax.fill_between(unicycle_result['t'], np.min(uni_distances, axis=1), np.max(uni_distances, axis=1), 
                     color='red', alpha=0.2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance from origin (m)')
    ax.set_title('System Spread from Origin')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Generic angle evolution
    ax = axes[1, 0]
    for i in range(n_osc):
        ax.plot(generic_result['t'], generic_result['angles'][:, i], 'b-', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('θ (rad)')
    ax.set_title('Generic Oscillators - Orientations')
    ax.grid(True, alpha=0.3)
    
    # Unicycle angle evolution
    ax = axes[1, 1]
    for i in range(n_osc):
        ax.plot(unicycle_result['t'], unicycle_result['angles'][:, i], 'r-', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('θ (rad)')
    ax.set_title('Unicycle Oscillators - Orientations')
    ax.grid(True, alpha=0.3)
    
    # Velocity comparison
    ax = axes[1, 2]
    gen_vels = np.sqrt(np.sum(generic_result['velocities']**2, axis=2))
    uni_vels = np.abs(unicycle_result['velocities'])
    ax.plot(generic_result['t'], np.mean(gen_vels, axis=1), 'b-', linewidth=2, label='Generic (mean speed)')
    ax.plot(unicycle_result['t'], np.mean(uni_vels, axis=1), 'r-', linewidth=2, label='Unicycle (mean v)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity magnitude')
    ax.set_title('Mean Velocity Over Time')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig


def create_animation(results, output_file='comparison_animation.mp4'):
    """
    Create an animation comparing the two systems.
    
    Args:
        results: Results dictionary from run_comparison
        output_file: Output video file name
    """
    
    generic_result = results['generic']
    unicycle_result = results['unicycle']
    n_osc = results['n_oscillators']
    
    fig = plt.figure(figsize=(14, 6))
    
    # Generic system
    ax1 = fig.add_subplot(121)
    ax1.set_xlim(-4, 4)
    ax1.set_ylim(-4, 4)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Z (m)')
    ax1.set_title('Generic Oscillators (Unconstrained)')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    gen_scatter = ax1.scatter([], [], c='blue', s=100, alpha=0.6, zorder=3)
    gen_arrows = ax1.quiver([], [], [], [], scale=25, scale_units='inches', alpha=0.7, color='blue')
    gen_trails = [ax1.plot([], [], 'b-', alpha=0.3)[0] for _ in range(n_osc)]
    
    # Unicycle system
    ax2 = fig.add_subplot(122)
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('Unicycle Oscillators (Non-holonomic)')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    uni_scatter = ax2.scatter([], [], c='red', s=100, alpha=0.6, zorder=3)
    uni_arrows = ax2.quiver([], [], [], [], scale=25, scale_units='inches', alpha=0.7, color='red')
    uni_trails = [ax2.plot([], [], 'r-', alpha=0.3)[0] for _ in range(n_osc)]
    
    time_text = fig.text(0.5, 0.95, '', ha='center', fontsize=12)
    
    # Trail length for animation
    trail_len = 50
    
    def animate(frame):
        # Generic system
        gen_pos = generic_result['positions'][frame, :, :]
        gen_angle = generic_result['angles'][frame, :]
        gen_vel = generic_result['velocities'][frame, :]
        
        gen_scatter.set_offsets(gen_pos)
        
        # Update arrows (velocity direction)
        gen_vel_mag = np.sqrt(np.sum(gen_vel**2, axis=1))
        gen_vel_norm = gen_vel.copy()
        for i in range(n_osc):
            if gen_vel_mag[i] > 0.01:
                gen_vel_norm[i] = gen_vel[i] / (gen_vel_mag[i] + 0.01)
        
        gen_arrows.set_offsets(gen_pos)
        gen_arrows.set_UVC(gen_vel_norm[:, 0], gen_vel_norm[:, 1])
        
        # Update trails
        start_frame = max(0, frame - trail_len)
        for i in range(n_osc):
            trail_pos = generic_result['positions'][start_frame:frame+1, i, :]
            gen_trails[i].set_data(trail_pos[:, 0], trail_pos[:, 1])
        
        # Unicycle system
        uni_pos = unicycle_result['positions'][frame, :, :]
        uni_angle = unicycle_result['angles'][frame, :]
        uni_vel = unicycle_result['velocities'][frame, :]
        
        uni_scatter.set_offsets(uni_pos)
        
        # Update arrows (heading direction * velocity magnitude)
        uni_arrow_x = np.cos(uni_angle) * np.abs(uni_vel)
        uni_arrow_z = np.sin(uni_angle) * np.abs(uni_vel)
        
        uni_arrows.set_offsets(uni_pos)
        uni_arrows.set_UVC(uni_arrow_x, uni_arrow_z)
        
        # Update trails
        for i in range(n_osc):
            trail_pos = unicycle_result['positions'][start_frame:frame+1, i, :]
            uni_trails[i].set_data(trail_pos[:, 0], trail_pos[:, 1])
        
        time_text.set_text(f'Time: {generic_result["t"][frame]:.2f}s')
        
        return [gen_scatter, gen_arrows, uni_scatter, uni_arrows, time_text] + gen_trails + uni_trails
    
    anim = FuncAnimation(fig, animate, frames=len(generic_result['t']), 
                        interval=50, blit=True, repeat=True)
    
    print(f"\nCreating animation: {output_file}")
    try:
        anim.save(output_file, writer='ffmpeg', fps=20)
        print(f"  ✓ Animation saved to {output_file}")
    except Exception as e:
        print(f"  ✗ Could not save animation: {e}")
        print(f"    (Make sure ffmpeg is installed)")
    
    return fig, anim


if __name__ == '__main__':
    # Run comparison
    results = run_comparison(
        n_oscillators=10,
        duration=10.0,
        n_frames=1000,
        amplitude=1.0,
        frequency=0.5
    )
    
    # Create static plots
    print("\nGenerating static plots...")
    fig1 = plot_static_comparison(results, oscillator_indices=[0, 3, 6])
    fig1.savefig('comparison_trajectories.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: comparison_trajectories.png")
    
    fig2 = plot_system_overview(results)
    fig2.savefig('comparison_overview.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: comparison_overview.png")
    
    # Create animation
    try:
        fig3, anim = create_animation(results, 'comparison_animation.mp4')
        fig3.savefig('comparison_animation_frame.png', dpi=150, bbox_inches='tight')
    except Exception as e:
        print(f"\nSkipping animation: {e}")
    
    print("\n" + "="*80)
    print("Comparison complete!")
    print("="*80)
    
    plt.show()
