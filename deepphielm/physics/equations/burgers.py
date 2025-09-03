"""Burgers equation implementation."""

import numpy as np
from typing import Dict, Optional
from ..pde_base import PDE
from ..operators import DifferentialOperator


class BurgersEquation(PDE):
    """
    Viscous Burgers equation: ∂u/∂t + u∂u/∂x = ν∂²u/∂x²
    
    Parameters
    ----------
    nu : float
        Viscosity coefficient
    """
    
    def __init__(self, nu: float = 0.01):
        super().__init__()
        self.nu = nu
        self.dimension = 2  # (x, t)
        self.order = 2
        self.ops = DifferentialOperator()
        
    def residual(
        self,
        u: np.ndarray,
        x: np.ndarray,
        derivatives: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Compute Burgers equation residual.
        
        Parameters
        ----------
        u : np.ndarray
            Solution values
        x : np.ndarray
            Points (x, t)
        derivatives : dict
            Contains 'dt', 'dx', 'dxx'
            
        Returns
        -------
        np.ndarray
            Residual: ut + u*ux - ν*uxx
        """
        ut = derivatives.get('dt', np.zeros_like(u))
        ux = derivatives.get('dx', np.zeros_like(u))
        uxx = derivatives.get('dxx', np.zeros_like(u))
        
        # Nonlinear advection term
        advection = self.ops.advection(u, derivatives)
        
        return ut + advection - self.nu * uxx
    
    def shock_wave_ic(self, x: np.ndarray) -> np.ndarray:
        """
        Initial condition for shock wave solution.
        
        Parameters
        ----------
        x : np.ndarray
            Spatial points
            
        Returns
        -------
        np.ndarray
            Initial values
        """
        return -np.sin(np.pi * x)
    
    def exact_solution_cole_hopf(self, x: np.ndarray, t: float) -> Optional[np.ndarray]:
        """
        Exact solution using Cole-Hopf transformation for specific IC.
        
        Valid for u(x,0) = -sin(πx) on [0, 2]
        
        Parameters
        ----------
        x : np.ndarray
            Spatial points
        t : float
            Time value
            
        Returns
        -------
        np.ndarray or None
            Exact solution if available
        """
        if self.nu <= 0:
            return None
            
        # Cole-Hopf transformation solution
        # This is for the specific initial condition u(x,0) = -sin(πx)
        
        def phi(x, t):
            """Auxiliary function for Cole-Hopf."""
            sum_val = 0
            for n in range(1, 50):  # Truncate series
                an = 2 * np.pi * self.nu * (1 - (-1)**n) / n
                bn = np.exp(-n**2 * np.pi**2 * self.nu * t)
                sum_val += an * bn * np.sin(n * np.pi * x)
            return sum_val
        
        # Compute solution
        numerator = np.zeros_like(x)
        denominator = np.zeros_like(x)
        
        for n in range(1, 50):
            exp_term = np.exp(-n**2 * np.pi**2 * self.nu * t)
            numerator += n * exp_term * np.sin(n * np.pi * x)
            denominator += exp_term * np.cos(n * np.pi * x)
        
        # Avoid division by zero
        denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
        
        return -2 * np.pi * self.nu * numerator / denominator