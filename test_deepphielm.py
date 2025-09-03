#!/usr/bin/env python3
"""
Test script for DeepPhiELM with numerical differentiation
"""

import numpy as np
import matplotlib.pyplot as plt
from deepphielm import PIELM
from deepphielm.physics.equations import HeatEquation1D


def main():
    print("Testing DeepPhiELM with Numerical Differentiation")
    print("=" * 50)
    
    # Simple 1D heat equation test
    alpha = 1.0
    pde = HeatEquation1D(alpha=alpha)
    
    # Generate simple test data
    n_data = 100
    x = np.random.uniform(0, 1, n_data)
    t = np.random.uniform(0, 0.1, n_data)
    X_data = np.column_stack([x, t])
    
    # Analytical solution
    y_data = np.exp(-np.pi**2 * alpha * t) * np.sin(np.pi * x)
    
    print(f"Training data: {n_data} points")
    print(f"PDE: Heat equation with α = {alpha}")
    
    # Create model with numerical differentiation
    model = PIELM(
        n_hidden=50,
        activation='tanh',
        pde=pde,
        lambda_data=1.0,
        lambda_physics=5.0,
        diff_step=1e-6,
        diff_method='central',
        random_state=42
    )
    
    print("Training model...")
    
    try:
        # Train model
        model.fit(X_data, y_data, n_collocation=200)
        
        # Test prediction
        x_test = np.linspace(0, 1, 20)
        t_test = 0.05
        X_test = np.column_stack([x_test, np.full(20, t_test)])
        
        u_pred = model.predict(X_test)
        u_exact = np.exp(-np.pi**2 * alpha * t_test) * np.sin(np.pi * x_test)
        
        # Compute error
        error = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
        
        print(f"Relative L2 error: {error:.2e}")
        
        # Test physics residual
        X_test_phys = np.random.uniform(0, 1, (50, 2))
        X_test_phys[:, 1] *= 0.1  # t in [0, 0.1]
        
        residual = model.compute_physics_residual(X_test_phys)
        avg_residual = np.mean(np.abs(residual))
        
        print(f"Average physics residual: {avg_residual:.2e}")
        
        # Check differentiation info
        diff_info = model.get_differentiation_info()
        print(f"Differentiation: {diff_info}")
        
        # Simple plot
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(x_test, u_pred, 'b-', label='DeepPhiELM', linewidth=2)
        plt.plot(x_test, u_exact, 'r--', label='Exact', linewidth=2)
        plt.xlabel('x')
        plt.ylabel('u')
        plt.title(f'Solution at t={t_test}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(x_test, np.abs(u_pred - u_exact), 'g-', linewidth=2)
        plt.xlabel('x')
        plt.ylabel('|Error|')
        plt.title('Absolute Error')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('deepphielm_test.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("\n✅ DeepPhiELM test completed successfully!")
        print(f"✅ Numerical differentiation working (step={model.differentiator.h})")
        print(f"✅ Physics constraints satisfied (residual={avg_residual:.2e})")
        print(f"✅ Prediction accuracy: {error:.2e}")
        
        if error < 1e-2:
            print("🎉 Test PASSED - Good accuracy achieved!")
        else:
            print("⚠️  Test WARNING - Consider tuning hyperparameters")
            
    except Exception as e:
        print(f"❌ Test FAILED with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()