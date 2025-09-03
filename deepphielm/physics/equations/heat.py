"""Heat equation implementations."""

import numpy as np
from typing import Dict, Optional
from ..pde_base import PDE
from ..operators import DifferentialOperator


class HeatEquation1D(PDE):
    """
    1D Heat equation: ∂u/∂t = α ∂²u/∂x²
    
    Parameters
    ----------
    alpha : float
        Thermal diffusivity coefficient
    """
    
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.dimension = 2  # (x, t)
        self.order = 2
        
    def residual(
        self,
        u: np.ndarray,
        x: np.ndarray,
        derivatives: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Compute heat equation residual.
        
        Parameters
        ----------
        u : np.ndarray
            Temperature values
        x : np.ndarray
            Points (x, t)
        derivatives : dict
            Contains 'dt' and 'dxx'
            
        Returns
        -------
        np.ndarray
            Residual: ut - α*uxx
        """
        ut = derivatives.get('dt', np.zeros_like(u))
        uxx = derivatives.get('dxx', np.zeros_like(u))
        
        return ut - self.alpha * uxx
    
    def exact_solution(self, x: np.ndarray) -> Optional[np.ndarray]:
        """
        Analytical solution for specific initial condition.
        u(x,t) = exp(-π²αt) * sin(πx)
        """
        if x.shape[1] != 2:
            return None
            
        x_coord = x[:, 0]
        t_coord = x[:, 1]
        
        return np.exp(-np.pi**2 * self.alpha * t_coord) * np.sin(np.pi * x_coord)


class HeatEquation2D(PDE):
    """
    2D Heat equation: ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)
    
    Parameters
    ----------
    alpha : float
        Thermal diffusivity coefficient
    """
    
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.dimension = 3  # (x, y, t)
        self.order = 2
        self.ops = DifferentialOperator()
        
    def residual(
        self,
        u: np.ndarray,
        x: np.ndarray,
        derivatives: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Compute 2D heat equation residual.
        
        Parameters
        ----------
        u : np.ndarray
            Temperature values
        x : np.ndarray
            Points (x, y, t)
        derivatives : dict
            Contains 'dt', 'dxx', 'dyy'
            
        Returns
        -------
        np.ndarray
            Residual: ut - α*∇²u
        """
        ut = derivatives.get('dt', np.zeros_like(u))
        laplacian = self.ops.laplacian(derivatives)
        
        return ut - self.alpha * laplacian
    
    def exact_solution(self, x: np.ndarray) -> Optional[np.ndarray]:
        """
        Analytical solution for specific initial condition.
        u(x,y,t) = exp(-2π²αt) * sin(πx) * sin(πy)
        """
        if x.shape[1] != 3:
            return None
            
        x_coord = x[:, 0]
        y_coord = x[:, 1]
        t_coord = x[:, 2]
        
        return (
            np.exp(-2 * np.pi**2 * self.alpha * t_coord) *
            np.sin(np.pi * x_coord) *
            np.sin(np.pi * y_coord)
        )