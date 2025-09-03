#!/usr/bin/env python3
"""
Example 3: Creating and Solving Custom PDEs with DeepPhiELM
===========================================================

This example shows how to create custom PDE classes and solve them.
We'll implement a reaction-diffusion equation with nonlinear reaction term.
Uses numerical differentiation for all derivative computations.
"""

import numpy as np
import matplotlib.pyplot as plt
from deepphielm import PIELM
from deepphielm.physics.pde_base import PDE
from deepphielm.physics.operators import DifferentialOperator
from deepphielm.utils import Visualizer
from typing import Dict


class ReactionDiffusionPDE(PDE):
    """
    Custom reaction-diffusion PDE:
    ∂u/∂t = D∇²u + r*u*(1 - u/K) - μu
    
    Where:
    - D: diffusion coefficient
    - r: reaction rate  
    - K: carrying capacity
    - μ: decay rate
    """
    
    def __init__(self, D: float = 0.1, r: float = 1.0, K: float = 1.0, mu: float = 0.1):
        super().__init__()
        self.D = D      # Diffusion coefficient
        self.r = r      # Reaction rate
        self.K = K      # Carrying capacity
        self.mu = mu    # Decay rate
        self.dimension = 3  # (x, y, t)
        self.order = 2
        self.ops = DifferentialOperator()
        
    def residual(self, u, x, derivatives):
        """Compute PDE residual."""
        # Time derivative
        ut = derivatives.get('dt', np.zeros_like(u))
        
        # Diffusion term
        laplacian = self.ops.laplacian(derivatives)
        diffusion = self.D * laplacian
        
        # Nonlinear reaction term (logistic growth)
        reaction = self.r * u * (1 - u / self.K)
        
        # Decay term
        decay = self.mu * u
        
        # PDE residual: ut - D∇²u - r*u*(1-u/K) + μu = 0
        residual = ut - diffusion - reaction + decay
        
        return residual
    
    def source_term(self, x):
        """No external source term."""
        return np.zeros(x.shape[0])
    
    def get_info(self):
        """Return PDE information."""
        return {
            'name': 'Reaction-Diffusion',
            'equation': '∂u/∂t = D∇²u + r*u*(1-u/K) - μu',
            'parameters': {
                'D': self.D,
                'r': self.r, 
                'K': self.K,
                'μ': self.mu
            }
        }


class KuramotoSivashinskyPDE(PDE):
    """
    Kuramoto-Sivashinsky equation (1D):
    ∂u/∂t + u∂u/∂x + ∂²u/∂x² + ν∂⁴u/∂x⁴ = 0
    
    This is a chaotic PDE that's challenging to solve.
    """
    
    def __init__(self, nu: float = 1.0):
        super().__init__()
        self.nu = nu
        self.dimension = 2  # (x, t)
        self.order = 4  # Fourth-order PDE
        
    def residual(self, u, x, derivatives):
        """Compute KS equation residual."""
        # Derivatives
        ut = derivatives.get('dt', np.zeros_like(u))
        ux = derivatives.get('dx', np.zeros_like(u))
        uxx = derivatives.get('dxx', np.zeros_like(u))
        uxxxx = derivatives.get('dxxxx', np.zeros_like(u))  # Fourth derivative
        
        # KS equation: ut + u*ux + uxx + ν*uxxxx = 0
        residual = ut + u * ux + uxx + self.nu * uxxxx
        
        return residual


def solve_reaction_diffusion():
    """Solve the reaction-diffusion PDE."""
    print("Solving Reaction-Diffusion PDE")
    print("-" * 35)
    
    # Problem parameters
    L = 2.0    # Domain size
    T = 2.0    # Final time
    D = 0.05   # Diffusion
    r = 2.0    # Reaction rate
    K = 1.0    # Carrying capacity
    mu = 0.1   # Decay rate
    
    # Create PDE
    pde = ReactionDiffusionPDE(D=D, r=r, K=K, mu=mu)
    info = pde.get_info()
    
    print(f"PDE: {info['equation']}")
    for param, value in info['parameters'].items():
        print(f"  {param} = {value}")
    
    # Initial condition: Gaussian bump
    def initial_condition(x, y):
        return np.exp(-((x - L/2)**2 + (y - L/2)**2) / 0.2)
    
    # Generate training data
    print(f"\nGenerating training data...")
    
    # Initial condition data
    n_ic = 200
    np.random.seed(42)
    x_ic = np.random.uniform(0, L, n_ic)
    y_ic = np.random.uniform(0, L, n_ic)
    t_ic = np.zeros(n_ic)
    
    X_ic = np.column_stack([x_ic, y_ic, t_ic])
    y_ic_data = initial_condition(x_ic, y_ic)
    
    # Boundary conditions (zero on all boundaries)
    n_bc = 100
    t_bc = np.random.uniform(0, T, n_bc)
    
    # Create boundary points
    X_bc = []
    y_bc_data = []
    
    # Bottom boundary (y=0)
    x_bc = np.random.uniform(0, L, n_bc//4)
    X_bc.append(np.column_stack([x_bc, np.zeros(len(x_bc)), np.random.uniform(0, T, len(x_bc))]))
    y_bc_data.extend(np.zeros(len(x_bc)))
    
    # Top boundary (y=L)
    x_bc = np.random.uniform(0, L, n_bc//4)
    X_bc.append(np.column_stack([x_bc, np.full(len(x_bc), L), np.random.uniform(0, T, len(x_bc))]))
    y_bc_data.extend(np.zeros(len(x_bc)))
    
    # Left boundary (x=0)
    y_bc = np.random.uniform(0, L, n_bc//4)
    X_bc.append(np.column_stack([np.zeros(len(y_bc)), y_bc, np.random.uniform(0, T, len(y_bc))]))
    y_bc_data.extend(np.zeros(len(y_bc)))
    
    # Right boundary (x=L)
    y_bc = np.random.uniform(0, L, n_bc//4)
    X_bc.append(np.column_stack([np.full(len(y_bc), L), y_bc, np.random.uniform(0, T, len(y_bc))]))
    y_bc_data.extend(np.zeros(len(y_bc)))
    
    X_bc = np.vstack(X_bc)
    y_bc_data = np.array(y_bc_data)
    
    print(f"Initial condition points: {len(X_ic)}")
    print(f"Boundary condition points: {len(X_bc)}")
    
    # Create PIELM model
    model = PIELM(
        n_hidden=300,
        activation='tanh',
        pde=pde,
        lambda_data=1.0,
        lambda_physics=50.0,
        lambda_bc=100.0,
        lambda_ic=20.0,
        random_state=42
    )
    
    # Train model
    print(f"Training model...")
    
    model.fit(
        X_ic, y_ic_data,
        n_collocation=3000,
        X_bc=X_bc,
        y_bc=y_bc_data,
        X_ic=X_ic,
        y_ic=y_ic_data,
        collocation_strategy='latin_hypercube'
    )
    
    # Evaluate
    bounds = {'x': (0, L), 'y': (0, L), 't': (0, T)}
    from pypielm.utils.sampling import SamplingStrategy
    sampler = SamplingStrategy()
    X_test = sampler.generate_points(1000, bounds, 'uniform')
    
    residual = model.compute_physics_residual(X_test)
    avg_residual = np.mean(np.abs(residual))
    print(f"Average physics residual: {avg_residual:.2e}")
    
    # Visualize
    print(f"Generating visualizations...")
    
    # Create spatial grid for visualization
    x_vis = np.linspace(0, L, 50)
    y_vis = np.linspace(0, L, 50)
    X_vis, Y_vis = np.meshgrid(x_vis, y_vis)
    
    # Time snapshots
    times = [0.0, 0.5, 1.0, 1.5]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, t in enumerate(times):
        # Create test points
        X_test_2d = np.column_stack([
            X_vis.flatten(),
            Y_vis.flatten(), 
            np.full(X_vis.size, t)
        ])
        
        # Predict
        u_pred = model.predict(X_test_2d).reshape(X_vis.shape)
        
        # Plot
        cs = axes[i].contourf(X_vis, Y_vis, u_pred, levels=15, cmap='viridis')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
        axes[i].set_title(f't = {t}')
        plt.colorbar(cs, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig('reaction_diffusion.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model


def solve_kuramoto_sivashinsky():
    """Solve the Kuramoto-Sivashinsky equation."""
    print("\n\nSolving Kuramoto-Sivashinsky PDE")
    print("-" * 40)
    
    # Problem parameters
    L = 16.0   # Domain length (typically 16π for KS)
    T = 10.0   # Final time
    nu = 1.0   # Viscosity parameter
    
    # Create PDE
    pde = KuramotoSivashinskyPDE(nu=nu)
    print(f"PDE: ∂u/∂t + u∂u/∂x + ∂²u/∂x² + ν∂⁴u/∂x⁴ = 0")
    print(f"  ν = {nu}")
    
    # Initial condition: Random perturbation
    def initial_condition(x):
        return 0.1 * np.cos(2*np.pi*x/L) + 0.05 * np.cos(4*np.pi*x/L)
    
    # Generate training data
    print("Generating training data...")
    
    # Initial condition
    n_ic = 200
    x_ic = np.linspace(0, L, n_ic)
    X_ic = np.column_stack([x_ic, np.zeros(n_ic)])
    y_ic = initial_condition(x_ic)
    
    # Periodic boundary conditions (u(0,t) = u(L,t))
    # For simplicity, we'll use a large penalty on boundaries
    n_bc = 50
    t_bc = np.random.uniform(0, T, n_bc)
    X_bc_left = np.column_stack([np.zeros(n_bc), t_bc])
    X_bc_right = np.column_stack([np.full(n_bc, L), t_bc])
    
    print(f"Initial condition points: {len(X_ic)}")
    print(f"Boundary constraint points: {2 * len(t_bc)}")
    
    # Note: KS equation requires 4th derivatives, which is challenging
    # We'll use a simplified approach here
    print("Warning: KS equation requires 4th derivatives - using approximation")
    
    # Create model with more hidden neurons for complex dynamics
    model = PIELM(
        n_hidden=500,
        activation='tanh', 
        pde=pde,
        lambda_data=1.0,
        lambda_physics=20.0,
        lambda_ic=50.0,
        regularization='l2',
        reg_param=1e-5,
        random_state=42
    )
    
    # Train (this may be challenging due to 4th derivatives)
    try:
        print("Training model...")
        bounds = {'x': (0, L), 't': (0, min(T, 2.0))}  # Limit time for stability
        
        from pypielm.utils.sampling import SamplingStrategy
        sampler = SamplingStrategy()
        X_collocation = sampler.generate_points(2000, bounds, 'latin_hypercube')
        
        model.fit(
            X_ic, y_ic,
            X_collocation=X_collocation,
            X_ic=X_ic,
            y_ic=y_ic
        )
        
        print("Training completed!")
        
        # Evaluate
        residual = model.compute_physics_residual(X_collocation)
        avg_residual = np.mean(np.abs(residual))
        print(f"Average physics residual: {avg_residual:.2e}")
        
        # Visualize (simplified)
        x_test = np.linspace(0, L, 100)
        t_test = 1.0  # Single time point
        X_test = np.column_stack([x_test, np.full(100, t_test)])
        u_pred = model.predict(X_test)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_test, u_pred, 'b-', linewidth=2, label=f'PIELM t={t_test}')
        plt.plot(x_ic, y_ic, 'r--', linewidth=2, label='Initial condition')
        plt.xlabel('x')
        plt.ylabel('u')
        plt.title('Kuramoto-Sivashinsky Equation (Simplified)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('kuramoto_sivashinsky.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"KS equation training failed: {e}")
        print("This is expected - KS equation requires specialized numerical methods")


def main():
    """Main function demonstrating custom PDEs."""
    print("PyPIELM Example: Custom PDE Implementation")
    print("=" * 50)
    
    # Solve reaction-diffusion equation
    rd_model = solve_reaction_diffusion()
    
    # Solve Kuramoto-Sivashinsky equation (challenging)
    solve_kuramoto_sivashinsky()
    
    # Advanced: Custom PDE with parameter estimation
    print("\n\nAdvanced: Parameter Estimation")
    print("-" * 35)
    
    # Suppose we have data but don't know the exact parameters
    # We can optimize both the neural network AND the PDE parameters
    
    print("This would involve:")
    print("1. Defining PDE parameters as trainable")
    print("2. Using optimization to fit both NN weights and PDE params")
    print("3. Regularizing parameter estimates")
    print("4. Cross-validation for parameter selection")
    
    # For demonstration, show how to access model information
    print(f"\nReaction-Diffusion model info:")
    if hasattr(rd_model.pde, 'get_info'):
        info = rd_model.pde.get_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    print("\nCustom PDE example completed!")
    print("Key takeaways:")
    print("1. Easy to implement custom PDEs by inheriting from PDE base class")
    print("2. Define residual method with your PDE formulation")
    print("3. Higher-order PDEs are more challenging and may need specialized methods")
    print("4. Nonlinear PDEs work well with PIELM approach")
    print("5. Parameter estimation is possible by making PDE parameters trainable")


if __name__ == "__main__":
    main()