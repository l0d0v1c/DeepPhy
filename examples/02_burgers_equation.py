#!/usr/bin/env python3
"""
Example 2: Solving Viscous Burgers Equation with DeepPhiELM
===========================================================

This example demonstrates solving the nonlinear Burgers equation:
∂u/∂t + u∂u/∂x = ν∂²u/∂x²

This is a good test case for nonlinear PDEs with shock formation.
Uses numerical differentiation for computing derivatives.
"""

import numpy as np
import matplotlib.pyplot as plt
from deepphielm import PIELM
from deepphielm.physics.equations import BurgersEquation
from deepphielm.utils import Visualizer
from deepphielm.optimization import HyperparameterOptimizer


def main():
    print("DeepPhiELM Example: Viscous Burgers Equation")
    print("=" * 45)
    
    # Problem parameters
    nu = 0.01  # Viscosity
    L = 2.0    # Domain length  
    T = 2.0    # Final time
    
    # Define the PDE
    pde = BurgersEquation(nu=nu)
    print(f"PDE: ∂u/∂t + u∂u/∂x = {nu} ∂²u/∂x²")
    
    # Initial condition: u(x,0) = -sin(πx)
    def initial_condition(x):
        return -np.sin(np.pi * x)
    
    # Boundary conditions: u(0,t) = u(2,t) = 0
    def boundary_condition(t):
        return 0.0
    
    print(f"Initial condition: u(x,0) = -sin(πx)")
    print(f"Boundary conditions: u(0,t) = u({L},t) = 0")
    print(f"Domain: x ∈ [0, {L}], t ∈ [0, {T}]")
    
    # Generate training data
    print("\nGenerating training data...")
    
    # Initial condition data
    n_ic = 100
    x_ic = np.linspace(0, L, n_ic)
    t_ic = np.zeros(n_ic)
    X_ic = np.column_stack([x_ic, t_ic])
    y_ic = initial_condition(x_ic)
    
    # Boundary condition data  
    n_bc = 50
    t_bc = np.linspace(0, T, n_bc)
    
    # Left boundary
    X_bc_left = np.column_stack([np.zeros(n_bc), t_bc])
    y_bc_left = np.zeros(n_bc)
    
    # Right boundary
    X_bc_right = np.column_stack([np.full(n_bc, L), t_bc])
    y_bc_right = np.zeros(n_bc)
    
    # Combine boundary data
    X_bc = np.vstack([X_bc_left, X_bc_right])
    y_bc = np.hstack([y_bc_left, y_bc_right])
    
    # Interior data (sparse)
    n_interior = 200
    np.random.seed(42)\n    x_interior = np.random.uniform(0, L, n_interior)
    t_interior = np.random.uniform(0, T, n_interior)
    X_interior = np.column_stack([x_interior, t_interior])
    
    # For interior points, we don't have exact solution, so we'll use sparse data
    # In practice, you might have experimental measurements here
    y_interior = initial_condition(x_interior) * np.exp(-0.1 * t_interior)  # Rough approximation
    
    # Combine all training data
    X_data = np.vstack([X_ic, X_interior])
    y_data = np.hstack([y_ic, y_interior])
    
    print(f"Initial condition points: {len(X_ic)}")
    print(f"Boundary condition points: {len(X_bc)}")
    print(f"Interior data points: {len(X_interior)}")
    print(f"Total training data: {len(X_data)}")
    
    # Hyperparameter optimization
    print("\nOptimizing hyperparameters...")
    
    param_bounds = {
        'n_hidden': (100, 400),
        'lambda_physics': (1.0, 100.0),
        'lambda_bc': (1.0, 200.0),
        'lambda_ic': (1.0, 50.0)
    }
    
    fixed_params = {
        'activation': 'tanh',
        'pde': pde,
        'lambda_data': 1.0,
        'regularization': 'l2',
        'reg_param': 1e-6,
        'random_state': 42
    }
    
    optimizer = HyperparameterOptimizer(
        PIELM,
        param_bounds,
        fixed_params
    )
    
    # Generate collocation points for optimization
    bounds = {'x': (0, L), 't': (0, T)}
    from deepphielm.utils.sampling import SamplingStrategy
    sampler = SamplingStrategy()
    X_collocation = sampler.generate_points(2000, bounds, 'latin_hypercube')
    
    best_params = optimizer.optimize(
        X_data, y_data, X_collocation,
        method='random',  # Using random for speed, could use 'bayesian'
        n_trials=20,
        cv_folds=3,
        scoring='combined'
    )
    
    print("Best hyperparameters:")
    for param, value in best_params.items():
        if param != 'pde':
            print(f"  {param}: {value}")
    
    # Train final model
    print("\nTraining final model...")
    
    model = PIELM(**best_params)
    
    # Train with all constraints
    model.fit(
        X_data, y_data, 
        X_collocation=X_collocation,
        X_bc=X_bc,
        y_bc=y_bc,
        X_ic=X_ic,
        y_ic=y_ic
    )
    
    print("Training completed!")
    
    # Evaluate model
    print("\nEvaluating model...")
    
    # Physics residual
    residual = model.compute_physics_residual(X_collocation)
    avg_residual = np.mean(np.abs(residual))
    max_residual = np.max(np.abs(residual))
    
    print(f"Average physics residual: {avg_residual:.2e}")
    print(f"Maximum physics residual: {max_residual:.2e}")
    
    # Boundary condition compliance
    bc_pred = model.predict(X_bc)
    bc_error = np.mean(np.abs(bc_pred - y_bc))
    print(f"Boundary condition error: {bc_error:.2e}")
    
    # Initial condition compliance
    ic_pred = model.predict(X_ic)
    ic_error = np.mean(np.abs(ic_pred - y_ic))
    print(f"Initial condition error: {ic_error:.2e}")
    
    # Visualization
    print("\nGenerating visualizations...")
    
    # Create test grid
    x_test = np.linspace(0, L, 100)
    t_test = np.linspace(0, T, 50)
    X_mesh, T_mesh = np.meshgrid(x_test, t_test)
    X_test_full = np.column_stack([X_mesh.flatten(), T_mesh.flatten()])
    
    # Predictions
    u_pred = model.predict(X_test_full).reshape(len(t_test), len(x_test))
    
    # Plot 1: Solution evolution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Contour plot
    cs = axes[0, 0].contourf(X_mesh, T_mesh, u_pred, levels=20, cmap='RdBu_r')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('t')
    axes[0, 0].set_title('Burgers Equation Solution')
    plt.colorbar(cs, ax=axes[0, 0])
    
    # Time snapshots
    times = [0.0, 0.5, 1.0, 1.5]
    colors = ['blue', 'green', 'red', 'purple']
    
    for i, t_snap in enumerate(times):
        t_idx = np.argmin(np.abs(t_test - t_snap))
        axes[0, 1].plot(x_test, u_pred[t_idx, :], color=colors[i], 
                       linewidth=2, label=f't = {t_snap}')
    
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('u(x,t)')
    axes[0, 1].set_title('Solution Snapshots')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Physics residual
    residual_2d = model.compute_physics_residual(X_test_full).reshape(len(t_test), len(x_test))
    cs2 = axes[1, 0].contourf(X_mesh, T_mesh, np.log10(np.abs(residual_2d) + 1e-16), 
                             levels=20, cmap='hot')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('t')
    axes[1, 0].set_title('log₁₀(|Physics Residual|)')
    plt.colorbar(cs2, ax=axes[1, 0])
    
    # Training data distribution
    axes[1, 1].scatter(X_data[:, 0], X_data[:, 1], c='blue', s=5, alpha=0.6, label='Data')
    axes[1, 1].scatter(X_bc[:, 0], X_bc[:, 1], c='red', s=10, alpha=0.8, label='BC')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('t')
    axes[1, 1].set_title('Training Data Distribution')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('burgers_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Conservation analysis
    print("\nAnalyzing conservation properties...")
    
    # For Burgers equation, we can check energy conservation
    # E(t) = ∫ u²/2 dx
    energy = np.zeros(len(t_test))
    
    for i, t_snap in enumerate(t_test):
        u_slice = u_pred[i, :]
        # Numerical integration using trapezoidal rule
        energy[i] = np.trapz(u_slice**2, x_test) / 2
    
    # Plot energy evolution
    plt.figure(figsize=(10, 6))
    plt.plot(t_test, energy, 'b-', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Energy = ∫ u²/2 dx')
    plt.title('Energy Evolution (Should Decrease for Viscous Flow)')
    plt.grid(True, alpha=0.3)
    plt.savefig('burgers_energy.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Shock detection
    print("\nAnalyzing shock formation...")
    
    # Compute spatial gradient to detect shocks
    dudx = np.gradient(u_pred, x_test, axis=1)
    
    # Find maximum gradient at each time
    max_gradient = np.max(np.abs(dudx), axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(t_test, max_gradient, 'r-', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Max |∂u/∂x|')
    plt.title('Shock Formation (Steep Gradients)')
    plt.grid(True, alpha=0.3)
    plt.savefig('burgers_shock.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary
    print("\nSummary:")
    print(f"Successfully solved viscous Burgers equation with ν = {nu}")
    print(f"Physics residual: {avg_residual:.2e}")
    print(f"Boundary compliance: {bc_error:.2e}")
    print(f"Initial condition compliance: {ic_error:.2e}")
    print(f"Energy decreases over time: {energy[-1] < energy[0]}")
    print(f"Maximum shock gradient: {np.max(max_gradient):.2f}")
    
    print("\nExample completed successfully!")
    print("Generated files:")
    print("- burgers_results.png")
    print("- burgers_energy.png")
    print("- burgers_shock.png")


if __name__ == "__main__":
    main()