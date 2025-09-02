"""Poisson equation implementation."""

import numpy as np
from typing import Dict, Optional
from ..pde_base import PDE
from ..operators import DifferentialOperator


class PoissonEquation2D(PDE):
    """
    2D Poisson equation: -∇²u = f
    
    Parameters
    ----------
    source : callable or float
        Source term f(x,y)
    """
    
    def __init__(self, source=None):
        super().__init__()
        self.source = source
        self.dimension = 2  # (x, y)
        self.order = 2
        self.ops = DifferentialOperator()
        
    def residual(
        self,
        u: np.ndarray,
        x: np.ndarray,
        derivatives: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Compute Poisson equation residual.
        
        Parameters
        ----------
        u : np.ndarray
            Solution values
        x : np.ndarray
            Points (x, y)
        derivatives : dict
            Contains 'dxx', 'dyy'
            
        Returns
        -------
        np.ndarray
            Residual: -∇²u - f
        """
        laplacian = self.ops.laplacian(derivatives)
        source_term = self.source_term(x)
        
        return -laplacian - source_term
    
    def source_term(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate source term.
        
        Parameters
        ----------
        x : np.ndarray
            Evaluation points
            
        Returns
        -------
        np.ndarray
            Source values
        """
        if self.source is None:
            return np.zeros(x.shape[0])
        elif callable(self.source):
            return self.source(x)
        else:
            return np.full(x.shape[0], self.source)
    
    def fundamental_solution(self, x: np.ndarray, x0: np.ndarray) -> np.ndarray:
        """
        Fundamental solution (Green's function) for 2D Poisson.
        
        G(x,x0) = -1/(2π) * ln|x - x0|
        
        Parameters
        ----------
        x : np.ndarray
            Evaluation points
        x0 : np.ndarray
            Source point
            
        Returns
        -------
        np.ndarray
            Green's function values
        """
        r = np.sqrt((x[:, 0] - x0[0])**2 + (x[:, 1] - x0[1])**2)
        # Avoid log(0)
        r = np.maximum(r, 1e-10)
        return -np.log(r) / (2 * np.pi)
    
    def manufactured_solution(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Manufactured solution for testing.
        
        u(x,y) = sin(2πx) * cos(2πy)
        f(x,y) = 8π² * sin(2πx) * cos(2πy)
        
        Parameters
        ----------
        x : np.ndarray
            Points (x, y)
            
        Returns
        -------
        dict
            Contains 'u' and 'f'
        """
        x_coord = x[:, 0]
        y_coord = x[:, 1]
        
        u = np.sin(2 * np.pi * x_coord) * np.cos(2 * np.pi * y_coord)
        f = 8 * np.pi**2 * u
        
        return {'u': u, 'f': f}