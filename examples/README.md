# PyPIELM Examples

This directory contains comprehensive examples demonstrating various features and capabilities of PyPIELM.

## 📁 Example Files

### 1. `01_heat_equation_1d.py` - Basic Heat Equation
- **What it demonstrates**: Basic PIELM usage for solving 1D heat equation
- **Key concepts**: 
  - Simple PDE setup
  - Model training and evaluation
  - Visualization of results
  - Comparison with analytical solution
- **Run time**: ~30 seconds
- **Difficulty**: Beginner ⭐

### 2. `02_burgers_equation.py` - Nonlinear Burgers Equation  
- **What it demonstrates**: Handling nonlinear PDEs with shock formation
- **Key concepts**:
  - Nonlinear PDE formulation
  - Boundary and initial conditions
  - Hyperparameter optimization
  - Shock detection and analysis
  - Energy conservation monitoring
- **Run time**: ~2 minutes
- **Difficulty**: Intermediate ⭐⭐

### 3. `03_custom_pde.py` - Custom PDE Implementation
- **What it demonstrates**: Creating and solving custom PDEs
- **Key concepts**:
  - Implementing custom PDE classes
  - Reaction-diffusion equations
  - Complex nonlinear terms
  - Parameter estimation techniques
- **Run time**: ~1 minute
- **Difficulty**: Intermediate ⭐⭐

### 4. `04_optimization_example.py` - Advanced Optimization
- **What it demonstrates**: Comprehensive optimization strategies
- **Key concepts**:
  - Hyperparameter optimization (Bayesian, random search)
  - Adaptive training techniques
  - Cross-validation strategies
  - Multi-scale training
- **Run time**: ~5 minutes
- **Difficulty**: Advanced ⭐⭐⭐

## 🚀 Getting Started

### Prerequisites
```bash
pip install pypielm
# or install from source:
# pip install -e .
```

### Running Examples

Each example is self-contained and can be run independently:

```bash
cd examples/
python 01_heat_equation_1d.py
```

### Expected Output
Each example will:
1. Print progress information to console
2. Generate plots showing results
3. Save visualization images to disk
4. Provide performance metrics and analysis

## 📊 Example Outputs

### Heat Equation (Example 1)
- `heat_1d_results.png`: Solution, error, and residual plots
- `heat_1d_evolution.png`: Time evolution of solution
- `heat_1d_size_analysis.png`: Model size vs accuracy analysis

### Burgers Equation (Example 2)  
- `burgers_results.png`: Solution evolution and residual analysis
- `burgers_energy.png`: Energy conservation plot
- `burgers_shock.png`: Shock formation analysis

### Custom PDE (Example 3)
- `reaction_diffusion.png`: 2D reaction-diffusion solution snapshots
- `kuramoto_sivashinsky.png`: KS equation results (if successful)

### Optimization (Example 4)
- `adaptive_training_comparison.png`: Comparison of training strategies
- `cross_validation_comparison.png`: Cross-validation results

## 🛠️ Customization Tips

### Modifying Examples

1. **Change PDE parameters**:
```python
pde = HeatEquation1D(alpha=0.5)  # Different diffusion coefficient
```

2. **Adjust model architecture**:
```python
model = PIELM(
    n_hidden=300,        # More neurons
    activation='swish',  # Different activation
    lambda_physics=50.0  # Stronger physics constraint
)
```

3. **Use different sampling strategies**:
```python
model.fit(
    X_data, y_data,
    n_collocation=2000,
    collocation_strategy='sobol'  # Quasi-random sampling
)
```

### Creating New Examples

Template for new examples:
```python
#!/usr/bin/env python3
"""
Your Example: Description
========================
"""

import numpy as np
from pypielm import PIELM
from pypielm.physics.equations import YourPDE

def main():
    # 1. Define problem
    pde = YourPDE(parameters)
    
    # 2. Generate/load data
    X_data, y_data = generate_data()
    
    # 3. Create and train model
    model = PIELM(pde=pde, ...)
    model.fit(X_data, y_data)
    
    # 4. Evaluate and visualize
    evaluate_model(model)
    
if __name__ == "__main__":
    main()
```

## 🐛 Troubleshooting

### Common Issues

1. **Memory errors with large problems**:
   - Reduce `n_collocation` points
   - Use iterative solvers: `solver_method='cg'`
   - Decrease `n_hidden` neurons

2. **Poor physics compliance**:
   - Increase `lambda_physics` weight
   - Add more collocation points
   - Check PDE implementation for bugs

3. **Training instability**:
   - Add regularization: `reg_param=1e-5`
   - Use different activation: `activation='tanh'`
   - Reduce learning rate equivalent (lower lambda values)

4. **Slow training**:
   - Use optimized linear solvers
   - Reduce problem size
   - Enable multi-threading if available

### Performance Tips

1. **For large problems**: Use adaptive sampling
2. **For complex PDEs**: Use progressive training
3. **For high accuracy**: Use hyperparameter optimization
4. **For debugging**: Start with simple 1D problems

## 📚 Next Steps

After running these examples:

1. **Try your own PDE**: Implement a custom equation
2. **Experiment with parameters**: See how they affect results
3. **Compare methods**: Test against traditional numerical methods
4. **Scale up**: Apply to larger, more complex problems

## 🤝 Contributing Examples

We welcome new examples! Please:

1. Follow the existing code style
2. Include comprehensive documentation
3. Add error handling and validation
4. Test on different problem sizes
5. Submit via pull request

For questions or suggestions, please open an issue on GitHub.

---

Happy solving with PyPIELM! 🚀