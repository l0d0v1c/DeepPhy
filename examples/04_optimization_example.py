#!/usr/bin/env python3
"""
Example 4: Advanced Optimization Techniques with DeepPhiELM
===========================================================

This example demonstrates various optimization strategies:
1. Hyperparameter optimization
2. Adaptive training
3. Cross-validation
4. Multi-scale training

Uses numerical differentiation throughout.
"""

import numpy as np
import matplotlib.pyplot as plt
from deepphielm import PIELM
from deepphielm.physics.equations import HeatEquation2D
from deepphielm.optimization import HyperparameterOptimizer, AdaptiveTrainer, CrossValidator
from deepphielm.utils import Visualizer
import time


def generate_2d_heat_data():
    """Generate synthetic 2D heat equation data."""
    print("Generating 2D heat equation data...")
    
    # Domain
    L = 1.0
    T = 0.2
    alpha = 0.1
    
    # Analytical solution: u(x,y,t) = exp(-2π²αt) * sin(πx) * sin(πy)
    def analytical_solution(x, y, t):
        return np.exp(-2 * np.pi**2 * alpha * t) * np.sin(np.pi * x) * np.sin(np.pi * y)
    
    # Generate training data
    n_points = 500
    np.random.seed(42)
    
    x = np.random.uniform(0, L, n_points)
    y = np.random.uniform(0, L, n_points)
    t = np.random.uniform(0, T, n_points)
    
    X_data = np.column_stack([x, y, t])
    y_data = analytical_solution(x, y, t)
    
    # Boundary conditions (zero on all boundaries)
    n_bc = 200
    X_bc = []
    y_bc = []
    
    # Generate boundary points
    for _ in range(n_bc):
        t_rand = np.random.uniform(0, T)
        
        # Choose random boundary
        boundary = np.random.choice(['left', 'right', 'bottom', 'top'])
        
        if boundary == 'left':
            point = [0, np.random.uniform(0, L), t_rand]
        elif boundary == 'right':
            point = [L, np.random.uniform(0, L), t_rand]
        elif boundary == 'bottom':
            point = [np.random.uniform(0, L), 0, t_rand]
        else:  # top
            point = [np.random.uniform(0, L), L, t_rand]
        
        X_bc.append(point)
        y_bc.append(0.0)
    
    X_bc = np.array(X_bc)
    y_bc = np.array(y_bc)
    
    # Initial condition
    n_ic = 300
    x_ic = np.random.uniform(0, L, n_ic)
    y_ic = np.random.uniform(0, L, n_ic)
    X_ic = np.column_stack([x_ic, y_ic, np.zeros(n_ic)])
    y_ic_data = np.sin(np.pi * x_ic) * np.sin(np.pi * y_ic)
    
    bounds = {'x': (0, L), 'y': (0, L), 't': (0, T)}
    
    return X_data, y_data, X_bc, y_bc, X_ic, y_ic_data, bounds, analytical_solution


def hyperparameter_optimization_demo():
    """Demonstrate hyperparameter optimization."""
    print("\n" + "="*60)
    print("HYPERPARAMETER OPTIMIZATION DEMO")
    print("="*60)
    
    # Get data
    X_data, y_data, X_bc, y_bc, X_ic, y_ic, bounds, _ = generate_2d_heat_data()
    
    # Define PDE
    pde = HeatEquation2D(alpha=0.1)
    
    # Define parameter search space
    param_bounds = {
        'n_hidden': (100, 400),
        'lambda_physics': (1.0, 100.0),
        'lambda_bc': (10.0, 200.0),
        'lambda_ic': (5.0, 50.0),
        'reg_param': (1e-8, 1e-4)
    }
    
    fixed_params = {
        'activation': 'tanh',
        'pde': pde,
        'lambda_data': 1.0,
        'random_state': 42
    }
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(PIELM, param_bounds, fixed_params)
    
    # Generate collocation points
    from pypielm.utils.sampling import SamplingStrategy
    sampler = SamplingStrategy()
    X_collocation = sampler.generate_points(2000, bounds, 'latin_hypercube')
    
    print("Starting Bayesian optimization...")
    print(f"Search space: {len(param_bounds)} parameters")
    print(f"Parameter bounds: {param_bounds}")
    
    # Run optimization
    start_time = time.time()
    
    try:
        best_params = optimizer.optimize(
            X_data, y_data, X_collocation,
            method='random',  # Use 'bayesian' if scikit-optimize is available
            n_trials=15,
            cv_folds=3,
            scoring='combined'
        )
    except ImportError:
        print("Scikit-optimize not available, using random search...")
        best_params = optimizer.optimize(
            X_data, y_data, X_collocation,
            method='random',
            n_trials=15,
            cv_folds=3,
            scoring='combined'
        )
    
    opt_time = time.time() - start_time
    
    print(f"\\nOptimization completed in {opt_time:.1f} seconds")
    print("Best parameters:")
    for param, value in best_params.items():
        if param != 'pde':
            print(f"  {param}: {value}")
    
    print(f"Best score: {optimizer.best_score:.2e}")
    
    # Plot optimization history
    if optimizer.optimization_history:
        optimizer.plot_optimization_history()
    
    # Train final model with best parameters
    print("\\nTraining model with optimized parameters...")
    
    optimal_model = PIELM(**best_params)
    optimal_model.fit(
        X_data, y_data,
        X_collocation=X_collocation,
        X_bc=X_bc, y_bc=y_bc,
        X_ic=X_ic, y_ic=y_ic
    )
    
    # Evaluate
    residual = optimal_model.compute_physics_residual(X_collocation)
    avg_residual = np.mean(np.abs(residual))
    print(f"Optimized model physics residual: {avg_residual:.2e}")
    
    return optimal_model, best_params


def adaptive_training_demo():
    """Demonstrate adaptive training strategies."""
    print("\\n" + "="*60)
    print("ADAPTIVE TRAINING DEMO")
    print("="*60)
    
    # Get data
    X_data, y_data, X_bc, y_bc, X_ic, y_ic, bounds, _ = generate_2d_heat_data()
    
    # Define PDE
    pde = HeatEquation2D(alpha=0.1)
    
    # Create adaptive trainer
    initial_params = {
        'n_hidden': 200,
        'activation': 'tanh',
        'pde': pde,
        'lambda_data': 1.0,
        'lambda_physics': 20.0,
        'lambda_bc': 50.0,
        'lambda_ic': 10.0,
        'random_state': 42
    }
    
    trainer = AdaptiveTrainer(PIELM, initial_params)
    
    # Strategy 1: Adaptive refinement
    print("Strategy 1: Adaptive Refinement Training")
    print("-" * 45)
    
    start_time = time.time()
    
    refined_model = trainer.adaptive_refinement_training(
        X_data, y_data, bounds,
        max_iterations=5,
        initial_collocation=1000,
        refinement_factor=1.3,
        tolerance=1e-5
    )
    
    refine_time = time.time() - start_time
    
    print(f"Adaptive refinement completed in {refine_time:.1f} seconds")
    
    # Plot training history
    if trainer.training_history:
        trainer.plot_training_history()
    
    # Strategy 2: Progressive training with changing weights
    print("\\nStrategy 2: Progressive Training")
    print("-" * 35)
    
    # Define training stages with different emphasis
    lambda_schedule = [
        # Stage 1: Focus on data fitting
        {'lambda_data': 10.0, 'lambda_physics': 1.0, 'lambda_bc': 1.0, 'lambda_ic': 5.0},
        # Stage 2: Increase physics weight
        {'lambda_data': 5.0, 'lambda_physics': 10.0, 'lambda_bc': 5.0, 'lambda_ic': 5.0},
        # Stage 3: Balance all terms
        {'lambda_data': 1.0, 'lambda_physics': 20.0, 'lambda_bc': 50.0, 'lambda_ic': 10.0}
    ]
    
    # Generate collocation points for progressive training
    from pypielm.utils.sampling import SamplingStrategy
    sampler = SamplingStrategy()
    X_collocation = sampler.generate_points(2000, bounds, 'latin_hypercube')
    
    trainer_prog = AdaptiveTrainer(PIELM, initial_params)
    
    progressive_model = trainer_prog.progressive_training(
        X_data, y_data, X_collocation,
        lambda_schedule,
        n_epochs_per_stage=3
    )
    
    print("Progressive training completed!")
    
    # Strategy 3: Multi-scale training
    print("\\nStrategy 3: Multi-scale Training")
    print("-" * 35)
    
    scales = [20, 30, 40]  # Grid resolutions
    
    trainer_multi = AdaptiveTrainer(PIELM, initial_params)
    
    multiscale_model = trainer_multi.multi_scale_training(
        X_data, y_data, bounds,
        scales,
        transfer_weights=True
    )
    
    print("Multi-scale training completed!")
    
    # Compare all models
    print("\\nComparing adaptive training strategies...")
    
    models = {
        'Adaptive Refinement': refined_model,
        'Progressive Training': progressive_model,
        'Multi-scale Training': multiscale_model
    }
    
    # Test points for comparison
    x_test = np.linspace(0.1, 0.9, 20)
    y_test = np.linspace(0.1, 0.9, 20)
    X_test, Y_test = np.meshgrid(x_test, y_test)
    X_test_flat = np.column_stack([
        X_test.flatten(),
        Y_test.flatten(),
        np.full(X_test.size, 0.1)  # t = 0.1
    ])
    
    # Analytical solution
    _, _, _, _, _, _, _, analytical = generate_2d_heat_data()
    u_exact = analytical(X_test.flatten(), Y_test.flatten(), 0.1)
    
    comparison_results = {}
    
    for name, model in models.items():
        if model is not None:
            u_pred = model.predict(X_test_flat)
            error = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
            
            # Physics residual
            sampler = SamplingStrategy()
            X_phys_test = sampler.generate_points(500, bounds, 'uniform')
            residual = model.compute_physics_residual(X_phys_test)
            avg_residual = np.mean(np.abs(residual))
            
            comparison_results[name] = {
                'L2_error': error,
                'Physics_residual': avg_residual
            }
            
            print(f"{name}:")
            print(f"  L2 error: {error:.2e}")
            print(f"  Physics residual: {avg_residual:.2e}")
    
    # Visualize comparison
    if comparison_results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        methods = list(comparison_results.keys())
        l2_errors = [comparison_results[m]['L2_error'] for m in methods]
        phys_residuals = [comparison_results[m]['Physics_residual'] for m in methods]
        
        axes[0].bar(methods, l2_errors)
        axes[0].set_ylabel('L2 Error')
        axes[0].set_title('Prediction Accuracy')
        axes[0].tick_params(axis='x', rotation=45)
        
        axes[1].bar(methods, phys_residuals)
        axes[1].set_ylabel('Physics Residual')
        axes[1].set_title('Physics Compliance')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('adaptive_training_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    return refined_model


def cross_validation_demo():
    """Demonstrate cross-validation techniques."""
    print("\\n" + "="*60)
    print("CROSS-VALIDATION DEMO")
    print("="*60)
    
    # Get data
    X_data, y_data, X_bc, y_bc, X_ic, y_ic, bounds, _ = generate_2d_heat_data()
    
    # Define PDE and model parameters
    pde = HeatEquation2D(alpha=0.1)
    model_params = {
        'n_hidden': 200,
        'activation': 'tanh',
        'pde': pde,
        'lambda_data': 1.0,
        'lambda_physics': 20.0,
        'lambda_bc': 50.0,
        'random_state': 42
    }
    
    # Generate collocation points
    from pypielm.utils.sampling import SamplingStrategy
    sampler = SamplingStrategy()
    X_collocation = sampler.generate_points(2000, bounds, 'latin_hypercube')
    
    # Create cross-validator
    cv = CrossValidator()
    
    # Strategy 1: Physics-aware cross-validation
    print("Physics-aware K-fold cross-validation...")
    
    cv_results = cv.physics_aware_cv(
        PIELM,
        model_params,
        X_data, y_data,
        X_collocation,
        n_folds=5,
        validation_metrics=['mse', 'physics_residual', 'r2']
    )
    
    print("Cross-validation results:")
    for metric, value in cv_results.items():
        print(f"  {metric}: {value:.2e}")
    
    # Strategy 2: Temporal cross-validation (if time-dependent data)
    print("\\nTemporal cross-validation...")
    
    # Sort data by time for temporal CV
    time_sorted_idx = np.argsort(X_data[:, 2])  # Sort by time dimension
    X_temporal = X_data[time_sorted_idx]
    y_temporal = y_data[time_sorted_idx]
    
    temporal_results = cv.temporal_validation(
        PIELM,
        model_params,
        X_temporal, y_temporal,
        X_collocation,
        n_splits=4
    )
    
    print("Temporal validation results:")
    for metric, value in temporal_results.items():
        print(f"  {metric}: {value:.2e}")
    
    # Strategy 3: Spatial cross-validation
    print("\\nSpatial cross-validation...")
    
    spatial_results = cv.spatial_validation(
        PIELM,
        model_params,
        X_data, y_data,
        X_collocation,
        spatial_dims=[0, 1],  # x and y dimensions
        n_regions=4
    )
    
    print("Spatial validation results:")
    for metric, value in spatial_results.items():
        print(f"  {metric}: {value:.2e}")
    
    # Strategy 4: Bootstrap validation
    print("\\nBootstrap validation...")
    
    bootstrap_results = cv.bootstrap_validation(
        PIELM,
        model_params,
        X_data, y_data,
        X_collocation,
        n_bootstrap=20,
        bootstrap_ratio=0.8
    )
    
    print("Bootstrap validation results:")
    for metric, value in bootstrap_results.items():
        if isinstance(value, tuple):
            print(f"  {metric}: {value}")
        else:
            print(f"  {metric}: {value:.2e}")
    
    # Visualize validation results
    validation_types = ['Physics-aware', 'Temporal', 'Spatial', 'Bootstrap']
    mse_means = [
        cv_results['mse_mean'],
        temporal_results['mse_mean'],
        spatial_results['mse_mean'],
        bootstrap_results['mse_mean']
    ]
    
    mse_stds = [
        cv_results['mse_std'],
        temporal_results['mse_std'],
        spatial_results['mse_std'],
        bootstrap_results['mse_std']
    ]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(validation_types, mse_means, yerr=mse_stds, 
                fmt='bo-', linewidth=2, markersize=8, capsize=5)
    plt.ylabel('MSE')
    plt.title('Cross-Validation Results Comparison')
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('cross_validation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main optimization example."""
    print("PyPIELM Advanced Optimization Examples")
    print("=" * 50)
    
    # Run hyperparameter optimization
    optimal_model, best_params = hyperparameter_optimization_demo()
    
    # Run adaptive training demos
    adaptive_model = adaptive_training_demo()
    
    # Run cross-validation demos
    cross_validation_demo()
    
    # Summary
    print("\\n" + "="*60)
    print("OPTIMIZATION SUMMARY")
    print("="*60)
    
    print("Key optimization strategies demonstrated:")
    print("1. Hyperparameter Optimization:")
    print("   - Bayesian/Random search over parameter space")
    print("   - Cross-validation for robust evaluation")
    print("   - Automatic parameter selection")
    
    print("\\n2. Adaptive Training:")
    print("   - Adaptive refinement with residual-based sampling")
    print("   - Progressive training with changing loss weights")
    print("   - Multi-scale training from coarse to fine")
    
    print("\\n3. Cross-Validation:")
    print("   - Physics-aware K-fold validation")
    print("   - Temporal validation for time-series data")
    print("   - Spatial validation for distributed phenomena")
    print("   - Bootstrap validation for uncertainty estimation")
    
    print("\\nBest practices:")
    print("- Always validate with physics constraints")
    print("- Use adaptive sampling for complex domains")
    print("- Progressive training for difficult problems")
    print("- Multiple validation strategies for robustness")
    
    print("\\nOptimization example completed!")
    print("Generated files:")
    print("- adaptive_training_comparison.png")
    print("- cross_validation_comparison.png")


if __name__ == "__main__":
    main()