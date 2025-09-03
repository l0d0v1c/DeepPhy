"""Finite difference operators for structured grids."""

import numpy as np
from typing import Tuple, Optional


class FiniteDifference:
    """
    Finite difference operators for computing derivatives on structured grids.
    
    This class provides methods for computing derivatives using finite difference
    stencils, which can be useful for validation and comparison with numerical
    differentiation of neural networks.
    """
    
    def __init__(self, dx: float = 1.0, dy: float = 1.0, dt: float = 1.0):
        """
        Initialize finite difference operator.
        
        Parameters
        ----------
        dx : float
            Grid spacing in x direction
        dy : float
            Grid spacing in y direction
        dt : float
            Time step size
        """
        self.dx = dx
        self.dy = dy
        self.dt = dt
    
    def gradient_1d(
        self,
        u: np.ndarray,
        axis: int = 0,
        method: str = 'central'
    ) -> np.ndarray:
        """
        Compute 1D gradient using finite differences.
        
        Parameters
        ----------
        u : np.ndarray
            Field values
        axis : int
            Axis along which to compute gradient
        method : str
            Finite difference method
            
        Returns
        -------
        np.ndarray
            Gradient values
        """
        if method == 'central':
            # Central difference: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
            grad = np.zeros_like(u)
            
            if axis == 0:
                h = self.dx
                grad[1:-1] = (u[2:] - u[:-2]) / (2 * h)
                # Boundaries: forward/backward difference
                grad[0] = (u[1] - u[0]) / h
                grad[-1] = (u[-1] - u[-2]) / h
                
            elif axis == 1 and u.ndim > 1:
                h = self.dy
                grad[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * h)
                grad[:, 0] = (u[:, 1] - u[:, 0]) / h
                grad[:, -1] = (u[:, -1] - u[:, -2]) / h
                
        elif method == 'forward':
            # Forward difference: f'(x) ≈ (f(x+h) - f(x)) / h
            grad = np.zeros_like(u)
            
            if axis == 0:
                h = self.dx
                grad[:-1] = (u[1:] - u[:-1]) / h
                grad[-1] = grad[-2]  # Extrapolate last point
                
            elif axis == 1 and u.ndim > 1:
                h = self.dy
                grad[:, :-1] = (u[:, 1:] - u[:, :-1]) / h
                grad[:, -1] = grad[:, -2]
                
        elif method == 'backward':
            # Backward difference: f'(x) ≈ (f(x) - f(x-h)) / h
            grad = np.zeros_like(u)
            
            if axis == 0:
                h = self.dx
                grad[1:] = (u[1:] - u[:-1]) / h
                grad[0] = grad[1]
                
            elif axis == 1 and u.ndim > 1:
                h = self.dy
                grad[:, 1:] = (u[:, 1:] - u[:, :-1]) / h
                grad[:, 0] = grad[:, 1]
        
        return grad
    
    def laplacian_2d(
        self,
        u: np.ndarray,
        method: str = 'central'
    ) -> np.ndarray:
        """
        Compute 2D Laplacian using finite differences.
        
        ∇²u = ∂²u/∂x² + ∂²u/∂y²
        
        Parameters
        ----------
        u : np.ndarray
            2D field values of shape (nx, ny)
        method : str
            Finite difference method
            
        Returns
        -------
        np.ndarray
            Laplacian values
        """
        if u.ndim != 2:
            raise ValueError("Input must be 2D array")
        
        laplacian = np.zeros_like(u)
        
        if method == 'central':
            # Central difference for second derivatives
            # ∂²u/∂x² ≈ (u(x+h) - 2u(x) + u(x-h)) / h²
            
            # Interior points
            laplacian[1:-1, 1:-1] = (
                (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / self.dx**2 +
                (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / self.dy**2
            )
            
            # Boundary conditions (Neumann: ∂u/∂n = 0)
            # Left and right boundaries
            laplacian[0, :] = laplacian[1, :]
            laplacian[-1, :] = laplacian[-2, :]
            
            # Top and bottom boundaries
            laplacian[:, 0] = laplacian[:, 1]
            laplacian[:, -1] = laplacian[:, -2]
        
        return laplacian
    
    def gradient_2d(
        self,
        u: np.ndarray,
        method: str = 'central'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute 2D gradient components.
        
        Parameters
        ----------
        u : np.ndarray
            2D field values
        method : str
            Finite difference method
            
        Returns
        -------
        tuple
            (du_dx, du_dy) gradient components
        """
        du_dx = self.gradient_1d(u, axis=0, method=method)
        du_dy = self.gradient_1d(u, axis=1, method=method)
        
        return du_dx, du_dy
    
    def divergence_2d(
        self,
        u: np.ndarray,
        v: np.ndarray,
        method: str = 'central'
    ) -> np.ndarray:
        """
        Compute 2D divergence: ∇·F = ∂u/∂x + ∂v/∂y
        
        Parameters
        ----------
        u : np.ndarray
            x-component of vector field
        v : np.ndarray
            y-component of vector field
        method : str
            Finite difference method
            
        Returns
        -------
        np.ndarray
            Divergence values
        """
        du_dx = self.gradient_1d(u, axis=0, method=method)
        dv_dy = self.gradient_1d(v, axis=1, method=method)
        
        return du_dx + dv_dy
    
    def curl_2d(
        self,
        u: np.ndarray,
        v: np.ndarray,
        method: str = 'central'
    ) -> np.ndarray:
        """
        Compute 2D curl (z-component): curl·z = ∂v/∂x - ∂u/∂y
        
        Parameters
        ----------
        u : np.ndarray
            x-component of vector field
        v : np.ndarray
            y-component of vector field
        method : str
            Finite difference method
            
        Returns
        -------
        np.ndarray
            Curl z-component values
        """
        dv_dx = self.gradient_1d(v, axis=0, method=method)
        du_dy = self.gradient_1d(u, axis=1, method=method)
        
        return dv_dx - du_dy
    
    def time_derivative(
        self,
        u_current: np.ndarray,
        u_previous: np.ndarray,
        method: str = 'backward'
    ) -> np.ndarray:
        """
        Compute time derivative using finite differences.
        
        Parameters
        ----------
        u_current : np.ndarray
            Current time level
        u_previous : np.ndarray
            Previous time level
        method : str
            Time differencing method
            
        Returns
        -------
        np.ndarray
            Time derivative
        """
        if method == 'backward':
            # Backward Euler: ∂u/∂t ≈ (u^n - u^{n-1}) / Δt
            return (u_current - u_previous) / self.dt
            
        elif method == 'forward':
            # Forward Euler (would need future time step)
            # For now, same as backward
            return (u_current - u_previous) / self.dt
            
        else:
            raise ValueError(f"Unknown time method: {method}")
    
    def high_order_derivative(
        self,
        u: np.ndarray,
        order: int,
        axis: int = 0
    ) -> np.ndarray:
        """
        Compute high-order derivatives using finite differences.
        
        Parameters
        ----------
        u : np.ndarray
            Field values
        order : int
            Derivative order
        axis : int
            Axis for derivative
            
        Returns
        -------
        np.ndarray
            High-order derivative
        """
        if order == 1:
            return self.gradient_1d(u, axis)
            
        elif order == 2:
            # Second derivative: f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
            deriv = np.zeros_like(u)
            
            if axis == 0:
                h = self.dx
                if u.ndim == 1:
                    deriv[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / h**2
                    # Boundaries
                    deriv[0] = deriv[1]
                    deriv[-1] = deriv[-2]
                elif u.ndim == 2:
                    deriv[1:-1, :] = (u[2:, :] - 2*u[1:-1, :] + u[:-2, :]) / h**2
                    deriv[0, :] = deriv[1, :]
                    deriv[-1, :] = deriv[-2, :]
                    
            elif axis == 1:
                h = self.dy
                if u.ndim == 2:
                    deriv[:, 1:-1] = (u[:, 2:] - 2*u[:, 1:-1] + u[:, :-2]) / h**2
                    deriv[:, 0] = deriv[:, 1]
                    deriv[:, -1] = deriv[:, -2]
            
            return deriv
            
        elif order == 4:
            # Fourth derivative using 5-point stencil
            deriv = np.zeros_like(u)
            
            if axis == 0:
                h = self.dx
                if u.ndim == 1:
                    # 5-point stencil: (u[i+2] - 4u[i+1] + 6u[i] - 4u[i-1] + u[i-2]) / h⁴
                    deriv[2:-2] = (u[4:] - 4*u[3:-1] + 6*u[2:-2] - 4*u[1:-3] + u[:-4]) / h**4
                    # Boundaries (simplified)
                    deriv[:2] = deriv[2]
                    deriv[-2:] = deriv[-3]
            
            return deriv
        
        else:
            raise NotImplementedError(f"Order {order} derivatives not implemented")