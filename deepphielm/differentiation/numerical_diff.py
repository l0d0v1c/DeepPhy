"""Numerical differentiation for ELM activation functions."""

import numpy as np
from scipy.spatial.distance import cdist
from typing import Dict, Optional, Callable, Tuple


class NumericalDifferentiator:
    """
    Numerical differentiation for ELM networks without automatic differentiation.
    
    Uses finite differences to compute derivatives of network outputs with
    respect to inputs, avoiding the need for PyTorch or other AD frameworks.
    """
    
    def __init__(self, h: float = 1e-6, method: str = 'central'):
        """
        Initialize numerical differentiator.
        
        Parameters
        ----------
        h : float
            Step size for finite differences
        method : str
            Differentiation method ('central', 'forward', 'backward')
        """
        self.h = h
        self.method = method
        
    def compute_derivatives(
        self,
        model,
        X: np.ndarray,
        order: int = 2
    ) -> Dict[str, np.ndarray]:
        """
        Compute derivatives of model output w.r.t inputs using finite differences.
        
        Parameters
        ----------
        model : ELM model
            Model with predict method
        X : np.ndarray
            Input points of shape (n_points, n_dims)
        order : int
            Maximum order of derivatives to compute
            
        Returns
        -------
        dict
            Dictionary of derivatives
        """
        n_points, n_dims = X.shape
        derivatives = {}
        
        # Get base predictions
        u0 = model.predict(X)
        if len(u0.shape) == 1:
            u0 = u0.reshape(-1, 1)
        n_outputs = u0.shape[1]
        
        # First-order derivatives
        if order >= 1:
            for dim in range(n_dims):
                dim_name = self._get_dim_name(dim, n_dims)
                
                if self.method == 'central':
                    # Central difference
                    X_plus = X.copy()
                    X_minus = X.copy()
                    X_plus[:, dim] += self.h
                    X_minus[:, dim] -= self.h
                    
                    u_plus = model.predict(X_plus)
                    u_minus = model.predict(X_minus)
                    
                    if len(u_plus.shape) == 1:
                        u_plus = u_plus.reshape(-1, 1)
                        u_minus = u_minus.reshape(-1, 1)
                    
                    deriv = (u_plus - u_minus) / (2 * self.h)
                    
                elif self.method == 'forward':
                    # Forward difference
                    X_plus = X.copy()
                    X_plus[:, dim] += self.h
                    u_plus = model.predict(X_plus)
                    
                    if len(u_plus.shape) == 1:
                        u_plus = u_plus.reshape(-1, 1)
                    
                    deriv = (u_plus - u0) / self.h
                    
                else:  # backward
                    # Backward difference
                    X_minus = X.copy()
                    X_minus[:, dim] -= self.h
                    u_minus = model.predict(X_minus)
                    
                    if len(u_minus.shape) == 1:
                        u_minus = u_minus.reshape(-1, 1)
                    
                    deriv = (u0 - u_minus) / self.h
                
                derivatives[f'd{dim_name}'] = deriv
        
        # Second-order derivatives
        if order >= 2:
            for dim in range(n_dims):
                dim_name = self._get_dim_name(dim, n_dims)
                
                # Pure second derivatives
                X_plus = X.copy()
                X_minus = X.copy()
                X_plus[:, dim] += self.h
                X_minus[:, dim] -= self.h
                
                u_plus = model.predict(X_plus)
                u_minus = model.predict(X_minus)
                
                if len(u_plus.shape) == 1:
                    u_plus = u_plus.reshape(-1, 1)
                    u_minus = u_minus.reshape(-1, 1)
                
                # Second derivative: f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
                deriv_2 = (u_plus - 2*u0 + u_minus) / (self.h**2)
                derivatives[f'd{dim_name}{dim_name}'] = deriv_2
                
            # Mixed second derivatives (for 2D+ problems)
            if n_dims >= 2:
                for i in range(n_dims):
                    for j in range(i+1, n_dims):
                        dim_i = self._get_dim_name(i, n_dims)
                        dim_j = self._get_dim_name(j, n_dims)
                        
                        # Mixed derivative: ∂²u/∂x∂y
                        X_pp = X.copy()  # +h, +h
                        X_pm = X.copy()  # +h, -h
                        X_mp = X.copy()  # -h, +h
                        X_mm = X.copy()  # -h, -h
                        
                        X_pp[:, i] += self.h
                        X_pp[:, j] += self.h
                        
                        X_pm[:, i] += self.h
                        X_pm[:, j] -= self.h
                        
                        X_mp[:, i] -= self.h
                        X_mp[:, j] += self.h
                        
                        X_mm[:, i] -= self.h
                        X_mm[:, j] -= self.h
                        
                        u_pp = model.predict(X_pp)
                        u_pm = model.predict(X_pm)
                        u_mp = model.predict(X_mp)
                        u_mm = model.predict(X_mm)
                        
                        if len(u_pp.shape) == 1:
                            u_pp = u_pp.reshape(-1, 1)
                            u_pm = u_pm.reshape(-1, 1)
                            u_mp = u_mp.reshape(-1, 1)
                            u_mm = u_mm.reshape(-1, 1)
                        
                        # Mixed derivative formula
                        deriv_mixed = (u_pp - u_pm - u_mp + u_mm) / (4 * self.h**2)
                        derivatives[f'd{dim_i}{dim_j}'] = deriv_mixed
        
        return derivatives
    
    def _get_dim_name(self, dim: int, n_dims: int) -> str:
        """Get dimension name (x, y, z, t, etc.)."""
        if n_dims == 2:
            names = ['x', 't']
        elif n_dims == 3:
            names = ['x', 'y', 't']
        elif n_dims == 4:
            names = ['x', 'y', 'z', 't']
        else:
            names = [f'x{i}' for i in range(n_dims)]
        
        return names[dim] if dim < len(names) else f'x{dim}'
    
    def compute_gradient(
        self,
        model,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Compute gradient (first derivatives) at given points.
        
        Parameters
        ----------
        model : ELM model
            Model with predict method
        X : np.ndarray
            Input points
            
        Returns
        -------
        np.ndarray
            Gradient matrix of shape (n_points, n_dims, n_outputs)
        """
        derivatives = self.compute_derivatives(model, X, order=1)
        
        n_points, n_dims = X.shape
        u0 = model.predict(X)
        if len(u0.shape) == 1:
            u0 = u0.reshape(-1, 1)
        n_outputs = u0.shape[1]
        
        gradient = np.zeros((n_points, n_dims, n_outputs))
        
        for dim in range(n_dims):
            dim_name = self._get_dim_name(dim, n_dims)
            key = f'd{dim_name}'
            if key in derivatives:
                gradient[:, dim, :] = derivatives[key]
        
        return gradient
    
    def compute_laplacian(
        self,
        model,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Compute Laplacian (sum of second derivatives).
        
        Parameters
        ----------
        model : ELM model
            Model with predict method
        X : np.ndarray
            Input points
            
        Returns
        -------
        np.ndarray
            Laplacian values
        """
        derivatives = self.compute_derivatives(model, X, order=2)
        
        n_points, n_dims = X.shape
        u0 = model.predict(X)
        if len(u0.shape) == 1:
            u0 = u0.reshape(-1, 1)
        n_outputs = u0.shape[1]
        
        laplacian = np.zeros((n_points, n_outputs))
        
        # Sum pure second derivatives
        for dim in range(n_dims):
            dim_name = self._get_dim_name(dim, n_dims)
            key = f'd{dim_name}{dim_name}'
            if key in derivatives:
                laplacian += derivatives[key]
        
        return laplacian
    
    def set_step_size(self, h: float):
        """Update step size for finite differences."""
        self.h = h
    
    def adaptive_step_size(
        self,
        model,
        X: np.ndarray,
        target_accuracy: float = 1e-6
    ) -> float:
        """
        Adaptively determine optimal step size.
        
        Uses Richardson extrapolation to estimate optimal h.
        """
        h_values = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
        errors = []
        
        # Compute derivatives with different step sizes
        base_h = self.h
        
        for h in h_values:
            self.h = h
            try:
                deriv = self.compute_derivatives(model, X[:5], order=1)  # Test on subset
                # Use first derivative as test
                first_deriv_key = list(deriv.keys())[0]
                test_deriv = deriv[first_deriv_key]
                
                # Estimate error using Richardson extrapolation
                if len(errors) > 0:
                    # Compare with previous result
                    error = np.mean(np.abs(test_deriv - prev_deriv))
                    errors.append(error)
                
                prev_deriv = test_deriv
                
            except Exception as e:
                errors.append(1e6)  # Large error for failed computation
        
        # Find optimal step size
        if errors:
            min_error_idx = np.argmin(errors[1:]) + 1  # Skip first (no comparison)
            optimal_h = h_values[min_error_idx]
        else:
            optimal_h = 1e-6  # Default
        
        self.h = optimal_h
        return optimal_h