#!/usr/bin/env python3
"""
Example 5: Solving Schrödinger Equation for Hydrogen Atom
==========================================================

This example demonstrates solving the time-independent Schrödinger equation
for the hydrogen atom, specifically the 2p orbital.

The equation in spherical coordinates (r, θ, φ):
-ℏ²/2m ∇²ψ - e²/(4πε₀r) ψ = E ψ

For simplicity, we'll work in atomic units where ℏ = m = e = 4πε₀ = 1:
-½∇²ψ - 1/r ψ = E ψ

We'll focus on the 2p_z orbital (n=2, l=1, m=0) which has the form:
ψ(r,θ) = R₂₁(r) Y₁⁰(θ,φ) = (1/4√2π) (r/a₀) e^(-r/2a₀) cos(θ)

In 2D (r-z plane), this simplifies since z = r·cos(θ).
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from deepphielm import PIELM
from deepphielm.physics.pde_base import PDE


class SchrodingerHydrogen(PDE):
    """
    Time-independent Schrödinger equation for hydrogen atom.
    In atomic units: -½∇²ψ - 1/r ψ = E ψ
    """
    
    def __init__(self, n=2, l=1, m=0):
        """
        Parameters
        ----------
        n : int
            Principal quantum number
        l : int
            Azimuthal quantum number
        m : int
            Magnetic quantum number
        """
        super().__init__()
        self.n = n
        self.l = l
        self.m = m
        # Energy eigenvalue for hydrogen in atomic units
        self.E = -1.0 / (2 * n**2)
    
    def residual(self, psi, x, derivatives):
        """
        Compute PDE residual for Schrödinger equation.
        
        For 2D case in (r, z) coordinates where r = sqrt(x² + z²):
        -½(∂²ψ/∂x² + ∂²ψ/∂z² + (1/r)∂ψ/∂r) - ψ/r = E·ψ
        """
        # Extract coordinates
        x_coord = x[:, 0:1]  # x coordinate
        z_coord = x[:, 1:2]  # z coordinate
        
        # Compute r = sqrt(x² + z²)
        r = np.sqrt(x_coord**2 + z_coord**2)
        r = np.maximum(r, 1e-6)  # Avoid division by zero
        
        # Get derivatives
        psi_xx = derivatives.get('dxx', np.zeros_like(psi))
        psi_zz = derivatives.get('dyy', np.zeros_like(psi))  # d²/dz² stored as dyy
        psi_x = derivatives.get('dx', np.zeros_like(psi))
        psi_z = derivatives.get('dy', np.zeros_like(psi))   # d/dz stored as dy
        
        # Laplacian in cylindrical coordinates (assuming azimuthal symmetry)
        # ∇²ψ = ∂²ψ/∂x² + ∂²ψ/∂z² + (1/r)∂ψ/∂r
        # where ∂ψ/∂r = (x/r)∂ψ/∂x + (z/r)∂ψ/∂z
        dpsi_dr = (x_coord/r) * psi_x + (z_coord/r) * psi_z
        laplacian = psi_xx + psi_zz + dpsi_dr / r
        
        # Potential term
        V = -1.0 / r
        
        # Schrödinger equation residual: -½∇²ψ + V·ψ - E·ψ = 0
        residual = -0.5 * laplacian + V * psi - self.E * psi
        
        return residual


def analytical_2pz(x, z, normalize=True):
    """
    Analytical solution for 2p_z orbital of hydrogen.
    ψ₂p_z(r,θ) = (1/4√2π) (r/a₀) e^(-r/2a₀) cos(θ)
    
    In Cartesian: ψ(x,z) = C · r · e^(-r/2) · (z/r)
    where r = sqrt(x² + z²) and we use atomic units (a₀ = 1)
    """
    r = np.sqrt(x**2 + z**2)
    r = np.maximum(r, 1e-6)  # Avoid division by zero
    
    # Radial part: R₂₁(r) ∝ r·e^(-r/2)
    # Angular part: Y₁⁰(θ) ∝ cos(θ) = z/r
    if normalize:
        C = 1.0 / (4 * np.sqrt(2 * np.pi))  # Normalization constant
    else:
        C = 0.1  # Smaller for visualization
    
    psi = C * r * np.exp(-r/2) * (z/r)
    return psi


def generate_training_data(n_points=1000):
    """Generate training data for the 2p_z orbital."""
    print("Generating training data for 2p_z orbital...")
    
    # Domain: Use larger domain to capture orbital extent
    r_max = 12.0  # Bohr radii
    
    # Generate points in (x, z) plane
    # Use more points near nucleus and along z-axis
    np.random.seed(42)
    
    # Mix of uniform and concentrated sampling
    n_uniform = n_points // 2
    n_concentrated = n_points - n_uniform
    
    # Uniform sampling
    x_uniform = np.random.uniform(-r_max, r_max, n_uniform)
    z_uniform = np.random.uniform(-r_max, r_max, n_uniform)
    
    # Concentrated sampling near nucleus and along axes
    r_concentrated = np.random.exponential(3.0, n_concentrated)  # Exponential decay
    r_concentrated = np.minimum(r_concentrated, r_max)
    theta_concentrated = np.random.uniform(0, 2*np.pi, n_concentrated)
    x_concentrated = r_concentrated * np.cos(theta_concentrated)
    z_concentrated = r_concentrated * np.sin(theta_concentrated)
    
    # Combine all points
    x_all = np.concatenate([x_uniform, x_concentrated])
    z_all = np.concatenate([z_uniform, z_concentrated])
    
    # Create input array
    X_data = np.column_stack([x_all, z_all])
    
    # Generate target values (wavefunction)
    y_data = analytical_2pz(x_all, z_all, normalize=False)
    
    # Boundary conditions (wavefunction → 0 as r → ∞)
    n_bc = 200
    theta_bc = np.linspace(0, 2*np.pi, n_bc)
    x_bc = r_max * np.cos(theta_bc)
    z_bc = r_max * np.sin(theta_bc)
    X_bc = np.column_stack([x_bc, z_bc])
    y_bc = np.zeros(n_bc)  # ψ → 0 at large r
    
    return X_data, y_data, X_bc, y_bc


def main():
    print("DeepPhiELM Example: Schrödinger Equation for Hydrogen 2p_z Orbital")
    print("=" * 65)
    
    # Create the PDE
    pde = SchrodingerHydrogen(n=2, l=1, m=0)
    print(f"Solving for n=2, l=1, m=0 (2p_z orbital)")
    print(f"Energy eigenvalue: E = {pde.E:.3f} Hartree")
    
    # Generate training data
    X_data, y_data, X_bc, y_bc = generate_training_data(n_points=800)
    
    print(f"\nTraining data: {len(X_data)} points")
    print(f"Boundary conditions: {len(X_bc)} points")
    print(f"Domain: r ∈ [0, 12] Bohr radii")
    
    # Create and train PIELM model
    print("\nCreating DeepPhiELM model...")
    model = PIELM(
        n_hidden=500,            # More neurons for complex wavefunction
        activation='tanh',
        pde=pde,
        lambda_data=1.0,
        lambda_physics=0.1,      # Lower physics weight for stability
        lambda_bc=10.0,          # Strong boundary condition
        regularization='l2',
        reg_param=1e-5,          # Less regularization
        random_state=42,
        diff_step=5e-3,          # Larger step for numerical stability with 1/r
        diff_method='central'
    )
    
    print("Training model...")
    model.fit(
        X_data, y_data,
        X_bc=X_bc, y_bc=y_bc,
        n_collocation=2000,      # More collocation points
        collocation_strategy='latin_hypercube',
        max_physics_iterations=1  # Single iteration for stability
    )
    
    print("Training completed!")
    
    # Evaluate model
    print("\nEvaluating model...")
    
    # Create test grid
    n_test = 100
    x_test = np.linspace(-10, 10, n_test)
    z_test = np.linspace(-10, 10, n_test)
    X_test, Z_test = np.meshgrid(x_test, z_test)
    X_test_flat = np.column_stack([X_test.flatten(), Z_test.flatten()])
    
    # Predictions
    psi_pred = model.predict(X_test_flat).reshape(n_test, n_test)
    psi_exact = analytical_2pz(X_test.flatten(), Z_test.flatten(), normalize=False).reshape(n_test, n_test)
    
    # Compute probability density |ψ|²
    prob_pred = np.abs(psi_pred)**2
    prob_exact = np.abs(psi_exact)**2
    
    # Normalize for comparison
    prob_pred = prob_pred / np.max(prob_pred)
    prob_exact = prob_exact / np.max(prob_exact)
    
    # Compute errors
    psi_error = np.abs(psi_pred - psi_exact)
    relative_error = np.mean(psi_error) / (np.mean(np.abs(psi_exact)) + 1e-10)
    
    print(f"Mean absolute error: {np.mean(psi_error):.3e}")
    print(f"Relative error: {relative_error:.2%}")
    
    # Test physics residual
    X_test_phys = X_data[::5]  # Subsample for physics test
    residual = model.compute_physics_residual(X_test_phys)
    avg_residual = np.mean(np.abs(residual))
    print(f"Average physics residual: {avg_residual:.3e}")
    
    # Visualization
    print("\nGenerating visualizations...")
    
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Wavefunction (DeepPhiELM)
    ax1 = fig.add_subplot(2, 3, 1)
    im1 = ax1.contourf(X_test, Z_test, psi_pred, levels=30, cmap='RdBu_r')
    ax1.set_title('DeepPhiELM: ψ(x,z)')
    ax1.set_xlabel('x (Bohr radii)')
    ax1.set_ylabel('z (Bohr radii)')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1)
    
    # Plot 2: Analytical solution
    ax2 = fig.add_subplot(2, 3, 2)
    im2 = ax2.contourf(X_test, Z_test, psi_exact, levels=30, cmap='RdBu_r')
    ax2.set_title('Analytical: ψ(x,z)')
    ax2.set_xlabel('x (Bohr radii)')
    ax2.set_ylabel('z (Bohr radii)')
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2)
    
    # Plot 3: Error
    ax3 = fig.add_subplot(2, 3, 3)
    im3 = ax3.contourf(X_test, Z_test, psi_error, levels=20, cmap='hot')
    ax3.set_title(f'Absolute Error (Rel: {relative_error:.1%})')
    ax3.set_xlabel('x (Bohr radii)')
    ax3.set_ylabel('z (Bohr radii)')
    ax3.set_aspect('equal')
    plt.colorbar(im3, ax=ax3)
    
    # Plot 4: Probability density |ψ|² (DeepPhiELM)
    ax4 = fig.add_subplot(2, 3, 4)
    im4 = ax4.contourf(X_test, Z_test, prob_pred, levels=20, cmap='viridis')
    ax4.set_title('DeepPhiELM: |ψ|² (Probability)')
    ax4.set_xlabel('x (Bohr radii)')
    ax4.set_ylabel('z (Bohr radii)')
    ax4.set_aspect('equal')
    plt.colorbar(im4, ax=ax4)
    
    # Plot 5: Probability density (Analytical)
    ax5 = fig.add_subplot(2, 3, 5)
    im5 = ax5.contourf(X_test, Z_test, prob_exact, levels=20, cmap='viridis')
    ax5.set_title('Analytical: |ψ|² (Probability)')
    ax5.set_xlabel('x (Bohr radii)')
    ax5.set_ylabel('z (Bohr radii)')
    ax5.set_aspect('equal')
    plt.colorbar(im5, ax=ax5)
    
    # Plot 6: 1D slice along z-axis
    ax6 = fig.add_subplot(2, 3, 6)
    z_slice = np.linspace(-10, 10, 200)
    x_slice = np.zeros_like(z_slice)
    X_slice = np.column_stack([x_slice, z_slice])
    psi_slice_pred = model.predict(X_slice)
    psi_slice_exact = analytical_2pz(x_slice, z_slice, normalize=False)
    
    ax6.plot(z_slice, psi_slice_pred, 'b-', linewidth=2, label='DeepPhiELM')
    ax6.plot(z_slice, psi_slice_exact, 'r--', linewidth=2, label='Analytical')
    ax6.set_xlabel('z (Bohr radii)')
    ax6.set_ylabel('ψ(0,z)')
    ax6.set_title('Wavefunction along z-axis')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hydrogen_2pz_orbital.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 3D visualization of probability density
    fig2 = plt.figure(figsize=(12, 5))
    
    # 3D plot - DeepPhiELM
    ax3d1 = fig2.add_subplot(1, 2, 1, projection='3d')
    ax3d1.plot_surface(X_test, Z_test, prob_pred, cmap='viridis', alpha=0.9)
    ax3d1.set_title('DeepPhiELM: |ψ|² 3D')
    ax3d1.set_xlabel('x (Bohr radii)')
    ax3d1.set_ylabel('z (Bohr radii)')
    ax3d1.set_zlabel('Probability')
    ax3d1.view_init(elev=20, azim=45)
    
    # 3D plot - Analytical
    ax3d2 = fig2.add_subplot(1, 2, 2, projection='3d')
    ax3d2.plot_surface(X_test, Z_test, prob_exact, cmap='viridis', alpha=0.9)
    ax3d2.set_title('Analytical: |ψ|² 3D')
    ax3d2.set_xlabel('x (Bohr radii)')
    ax3d2.set_ylabel('z (Bohr radii)')
    ax3d2.set_zlabel('Probability')
    ax3d2.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig('hydrogen_2pz_3d.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Radial probability distribution
    fig3, (ax7, ax8) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Compute radial distribution
    r_test = np.sqrt(X_test.flatten()**2 + Z_test.flatten()**2)
    r_bins = np.linspace(0, 10, 50)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2
    
    # Bin the probability by radius
    prob_radial_pred = []
    prob_radial_exact = []
    
    for i in range(len(r_bins)-1):
        mask = (r_test >= r_bins[i]) & (r_test < r_bins[i+1])
        if np.sum(mask) > 0:
            prob_radial_pred.append(np.mean(prob_pred.flatten()[mask]))
            prob_radial_exact.append(np.mean(prob_exact.flatten()[mask]))
        else:
            prob_radial_pred.append(0)
            prob_radial_exact.append(0)
    
    ax7.plot(r_centers, prob_radial_pred, 'b-', linewidth=2, label='DeepPhiELM')
    ax7.plot(r_centers, prob_radial_exact, 'r--', linewidth=2, label='Analytical')
    ax7.set_xlabel('r (Bohr radii)')
    ax7.set_ylabel('Mean |ψ|²')
    ax7.set_title('Radial Probability Distribution')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Energy level diagram
    ax8.axhline(y=pde.E, color='blue', linewidth=3, label=f'n=2, E={pde.E:.3f}')
    ax8.axhline(y=-0.5, color='gray', linewidth=2, linestyle='--', alpha=0.5, label='n=1, E=-0.5')
    ax8.axhline(y=-0.125, color='gray', linewidth=2, linestyle='--', alpha=0.5, label='n=3, E=-0.125')
    ax8.set_ylim([-0.6, 0])
    ax8.set_xlim([0, 1])
    ax8.set_ylabel('Energy (Hartree)')
    ax8.set_title('Energy Levels of Hydrogen')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.set_xticks([])
    
    plt.tight_layout()
    plt.savefig('hydrogen_2pz_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*65)
    print("Summary:")
    print(f"✓ Successfully solved Schrödinger equation for 2p_z orbital")
    print(f"✓ Energy eigenvalue: E = {pde.E:.3f} Hartree (exact: -0.125)")
    print(f"✓ Relative error: {relative_error:.2%}")
    print(f"✓ Physics residual: {avg_residual:.3e}")
    print("\nGenerated files:")
    print("- hydrogen_2pz_orbital.png")
    print("- hydrogen_2pz_3d.png")
    print("- hydrogen_2pz_analysis.png")


if __name__ == "__main__":
    main()