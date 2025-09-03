#!/usr/bin/env python3
"""
Debug script pour identifier les problèmes avec l'équation de chaleur
"""

import numpy as np
import matplotlib.pyplot as plt
from deepphielm import PIELM
from deepphielm.physics.equations import HeatEquation1D

def main():
    print("=== DEBUG: Heat Equation 1D ===")
    
    # Paramètres très simples
    alpha = 1.0
    pde = HeatEquation1D(alpha=alpha)
    
    # Solution analytique: u(x,t) = sin(πx) * exp(-π²t)
    def u_exact(x, t):
        return np.sin(np.pi * x) * np.exp(-np.pi**2 * t)
    
    # Très peu de données pour debug
    np.random.seed(42)
    n_data = 100
    x_data = np.random.uniform(0.1, 0.9, n_data)  # Éviter les bords
    t_data = np.random.uniform(0.01, 0.05, n_data)  # Temps courts
    X_data = np.column_stack([x_data, t_data])
    y_data = u_exact(x_data, t_data)
    
    # Conditions aux limites
    n_bc = 50
    t_bc = np.random.uniform(0, 0.05, n_bc)
    X_bc = np.vstack([
        np.column_stack([np.zeros(n_bc//2), t_bc[:n_bc//2]]),
        np.column_stack([np.ones(n_bc//2), t_bc[n_bc//2:]])
    ])
    y_bc = np.zeros(n_bc)
    
    # Condition initiale
    n_ic = 50
    x_ic = np.linspace(0, 1, n_ic)
    X_ic = np.column_stack([x_ic, np.zeros(n_ic)])
    y_ic = u_exact(x_ic, 0)
    
    print(f"Data: {n_data}, BC: {n_bc}, IC: {n_ic}")
    
    # Test avec différentes configurations
    configs = [
        {"n_hidden": 100, "lambda_physics": 0.0, "name": "No Physics"},
        {"n_hidden": 100, "lambda_physics": 1.0, "name": "Weak Physics"},
        {"n_hidden": 100, "lambda_physics": 10.0, "name": "Strong Physics"},
        {"n_hidden": 200, "lambda_physics": 10.0, "name": "More Neurons"},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n--- Testing: {config['name']} ---")
        
        model = PIELM(
            n_hidden=config["n_hidden"],
            activation='tanh',
            pde=pde,
            lambda_data=1.0,
            lambda_physics=config["lambda_physics"],
            lambda_bc=10.0,
            lambda_ic=10.0,
            reg_param=1e-4,
            diff_step=1e-4,
            random_state=42
        )
        
        # Training
        model.fit(
            X_data, y_data,
            X_bc=X_bc, y_bc=y_bc,
            X_ic=X_ic, y_ic=y_ic,
            n_collocation=300,
            max_physics_iterations=1
        )
        
        # Test à t=0.03
        x_test = np.linspace(0, 1, 50)
        t_test = 0.03
        X_test = np.column_stack([x_test, np.full(50, t_test)])
        
        u_pred = model.predict(X_test)
        u_true = u_exact(x_test, t_test)
        
        error = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
        
        print(f"L2 Error: {error:.3f}")
        
        # Vérifier la physique
        if config["lambda_physics"] > 0:
            residual = model.compute_physics_residual(X_test)
            avg_residual = np.mean(np.abs(residual))
            print(f"Physics residual: {avg_residual:.3f}")
        
        results.append({
            'name': config['name'],
            'x': x_test,
            'pred': u_pred,
            'true': u_true,
            'error': error
        })
    
    # Visualisation comparative
    plt.figure(figsize=(15, 5))
    
    for i, result in enumerate(results):
        plt.subplot(1, 4, i+1)
        plt.plot(result['x'], result['pred'], 'b-', linewidth=2, label='PIELM')
        plt.plot(result['x'], result['true'], 'r--', linewidth=2, label='Exact')
        plt.title(f"{result['name']}\nError: {result['error']:.3f}")
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('debug_heat_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n=== SUMMARY ===")
    for result in results:
        print(f"{result['name']}: Error = {result['error']:.3f}")

if __name__ == "__main__":
    main()