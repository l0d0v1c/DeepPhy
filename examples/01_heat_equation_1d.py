#!/usr/bin/env python3
"""
Example 1: Solving 1D Heat Equation with DeepPhiELM
====================================================

This example demonstrates how to solve the 1D heat equation:
∂u/∂t = α ∂²u/∂x²

with initial condition u(x,0) = sin(πx) and boundary conditions u(0,t) = u(1,t) = 0
Uses numerical differentiation instead of automatic differentiation.
"""

import numpy as np
import matplotlib.pyplot as plt
from deepphielm import PIELM
from deepphielm.physics.equations import HeatEquation1D
from deepphielm.utils import Visualizer


def main():
    print("DeepPhiELM Example: 1D Heat Equation")
    print("=" * 40)
    
    # Problem parameters
    alpha = 1.0  # Thermal diffusivity
    L = 1.0      # Domain length
    T_final = 0.1  # Final time
    
    # Define the PDE
    pde = HeatEquation1D(alpha=alpha)
    print(f"PDE: ∂u/∂t = {alpha} ∂²u/∂x²")
    
    # Generate training data (random sampling instead of regular grid)
    print("\nGenerating training data...")
    n_data = 500  # Random sampling instead of grid
    
    np.random.seed(42)
    x_data = np.random.uniform(0, L, n_data)
    t_data = np.random.uniform(0, T_final, n_data)
    X_data = np.column_stack([x_data, t_data])
    
    # Analytical solution: u(x,t) = sin(πx) * exp(-π²αt)
    def analytical_solution(x, t):
        return np.sin(np.pi * x) * np.exp(-np.pi**2 * alpha * t)
    
    y_data = analytical_solution(X_data[:, 0], X_data[:, 1])
    
    # Generate boundary conditions u(0,t) = u(1,t) = 0
    print("Adding boundary conditions...")
    n_bc = 100
    t_bc = np.random.uniform(0, T_final, n_bc)
    X_bc_left = np.column_stack([np.zeros(n_bc//2), t_bc[:n_bc//2]])
    X_bc_right = np.column_stack([np.ones(n_bc//2), t_bc[n_bc//2:]])
    X_bc = np.vstack([X_bc_left, X_bc_right])
    y_bc = np.zeros(n_bc)
    
    # Generate initial condition u(x,0) = sin(πx)
    print("Adding initial condition...")
    n_ic = 100
    x_ic = np.random.uniform(0, L, n_ic)
    X_ic = np.column_stack([x_ic, np.zeros(n_ic)])
    y_ic = analytical_solution(x_ic, 0)
    
    print(f"Training data: {len(X_data)} points")
    print(f"Boundary conditions: {len(X_bc)} points")
    print(f"Initial conditions: {len(X_ic)} points")
    print(f"Domain: x ∈ [0, {L}], t ∈ [0, {T_final}]")
    
    # Create and train PIELM model
    print("\nCreating DeepPhiELM model...")
    model = PIELM(
        n_hidden=200,           # Balanced number of neurons
        activation='tanh',
        pde=pde,
        lambda_data=1.0,        # Data fitting weight
        lambda_physics=1.0,     # Moderate physics constraint weight
        lambda_bc=10.0,         # Moderate boundary condition weight
        lambda_ic=10.0,         # Moderate initial condition weight
        regularization='l2',
        reg_param=1e-4,         # Balanced regularization
        random_state=42,
        diff_step=1e-4,         # Balanced step size
        diff_method='central'   # Central finite differences
    )
    
    print("Training model...")
    model.fit(
        X_data, y_data,
        X_bc=X_bc, y_bc=y_bc,          # Add boundary conditions
        X_ic=X_ic, y_ic=y_ic,          # Add initial condition
        n_collocation=1000,            # Balanced collocation points
        collocation_strategy='latin_hypercube',
        max_physics_iterations=2       # Limited iterations for stability
    )
    
    print("Training completed!")
    
    # Evaluate model performance
    print("\nEvaluating model...")
    
    # Test data
    x_test = np.linspace(0, L, 100)
    t_test = 0.05  # Middle of time domain
    X_test = np.column_stack([x_test, np.full(len(x_test), t_test)])
    
    # Predictions
    u_pred = model.predict(X_test)
    u_exact = analytical_solution(x_test, t_test)
    
    # Compute errors
    l2_error = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
    max_error = np.max(np.abs(u_pred - u_exact))
    
    print(f"Relative L2 error: {l2_error:.2e}")
    print(f"Maximum error: {max_error:.2e}")
    
    # Compute physics residual
    X_collocation_test = model.sampling_strategy.generate_points(
        500, {'x': (0, L), 't': (0, T_final)}, 'uniform'
    )
    residual = model.compute_physics_residual(X_collocation_test)
    avg_residual = np.mean(np.abs(residual))
    
    print(f"Average physics residual: {avg_residual:.2e}")
    
    # Visualization
    print("\nGenerating visualizations...")
    
    # Plot 1: Solution at t=0.05
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(x_test, u_pred, 'b-', linewidth=2, label='DeepPhiELM')
    plt.plot(x_test, u_exact, 'r--', linewidth=2, label='Exact')
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title(f'Solution at t={t_test}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Error
    plt.subplot(1, 3, 2)
    error = np.abs(u_pred - u_exact)
    plt.plot(x_test, error, 'g-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('|Error|')
    plt.title('Absolute Error')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Physics residual spatial distribution
    plt.subplot(1, 3, 3)
    plt.scatter(X_collocation_test[:, 0], X_collocation_test[:, 1], 
                c=np.abs(residual), cmap='hot', s=10)
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Physics Residual')
    plt.colorbar(label='|Residual|')
    
    plt.tight_layout()
    plt.savefig('heat_1d_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Time evolution animation
    print("\nCreating time evolution plot...")
    
    # Multiple time snapshots
    t_snapshots = np.linspace(0, T_final, 5)
    colors = plt.cm.viridis(np.linspace(0, 1, len(t_snapshots)))
    
    plt.figure(figsize=(10, 6))
    
    for i, t_snap in enumerate(t_snapshots):
        X_snap = np.column_stack([x_test, np.full(len(x_test), t_snap)])
        u_pred_snap = model.predict(X_snap)
        u_exact_snap = analytical_solution(x_test, t_snap)
        
        plt.plot(x_test, u_pred_snap, '-', color=colors[i], 
                linewidth=2, label=f'PIELM t={t_snap:.2f}')
        plt.plot(x_test, u_exact_snap, '--', color=colors[i], 
                linewidth=1, alpha=0.7)
    
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title('Heat Equation: Time Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('heat_1d_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Performance analysis
    print("\nPerformance Analysis:")
    print(f"Model parameters: {model.n_hidden} hidden neurons")
    print(f"Collocation points: 1000")
    print(f"Training data points: {len(X_data)}")
    print(f"Final L2 error: {l2_error:.2e}")
    print(f"Physics residual: {avg_residual:.2e}")
    
    # Compare with different numbers of hidden neurons
    print("\nTesting different model sizes...")
    hidden_sizes = [50, 100, 200, 300]
    errors = []
    
    for n_hidden in hidden_sizes:
        print(f"Testing n_hidden = {n_hidden}")
        
        test_model = PIELM(
            n_hidden=n_hidden,
            activation='tanh',
            pde=pde,
            lambda_data=1.0,
            lambda_physics=10.0,
            random_state=42
        )
        
        test_model.fit(X_data, y_data, n_collocation=1000)
        u_test = test_model.predict(X_test)
        error = np.linalg.norm(u_test - u_exact) / np.linalg.norm(u_exact)
        errors.append(error)
        print(f"  Error: {error:.2e}")
    
    # Plot size vs error
    plt.figure(figsize=(8, 5))
    plt.loglog(hidden_sizes, errors, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Hidden Neurons')
    plt.ylabel('Relative L2 Error')
    plt.title('Model Size vs Accuracy')
    plt.grid(True, alpha=0.3)
    plt.savefig('heat_1d_size_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nExample completed successfully!")
    print("Generated files:")
    print("- heat_1d_results.png")
    print("- heat_1d_evolution.png") 
    print("- heat_1d_size_analysis.png")


if __name__ == "__main__":
    main()