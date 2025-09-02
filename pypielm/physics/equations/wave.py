"""Wave equation implementations."""

import numpy as np
from typing import Dict, Optional
from ..pde_base import PDE
from ..operators import DifferentialOperator


class WaveEquation1D(PDE):
    """
    1D Wave equation: ∂²u/∂t² = c² ∂²u/∂x²
    
    Parameters
    ----------
    c : float
        Wave speed
    """
    
    def __init__(self, c: float = 1.0):
        super().__init__()
        self.c = c
        self.dimension = 2  # (x, t)
        self.order = 2
        
    def residual(
        self,
        u: np.ndarray,
        x: np.ndarray,
        derivatives: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Compute wave equation residual.
        
        Parameters
        ----------
        u : np.ndarray
            Displacement values
        x : np.ndarray
            Points (x, t)
        derivatives : dict
            Contains 'dtt' and 'dxx'
            
        Returns
        -------
        np.ndarray
            Residual: utt - c²*uxx
        """
        utt = derivatives.get('dtt', np.zeros_like(u))
        uxx = derivatives.get('dxx', np.zeros_like(u))
        
        return utt - self.c**2 * uxx
    
    def d_alembert_solution(
        self,
        x: np.ndarray,
        t: np.ndarray,
        f: callable,
        g: callable
    ) -> np.ndarray:
        """
        D'Alembert's solution for wave equation.
        
        u(x,t) = [f(x-ct) + f(x+ct)]/2 + (1/2c)∫[x-ct to x+ct] g(s)ds
        
        Parameters
        ----------
        x : np.ndarray
            Spatial points
        t : np.ndarray
            Time points
        f : callable
            Initial displacement u(x,0) = f(x)
        g : callable
            Initial velocity ∂u/∂t(x,0) = g(x)
            
        Returns
        -------
        np.ndarray
            Solution values
        """
        x_minus = x - self.c * t
        x_plus = x + self.c * t
        
        # Traveling waves
        u = (f(x_minus) + f(x_plus)) / 2
        
        # Integral term for initial velocity
        if g is not None:
            from scipy.integrate import quad
            integral = np.zeros_like(x)
            for i in range(len(x)):
                integral[i], _ = quad(g, x_minus[i], x_plus[i])
            u += integral / (2 * self.c)
            
        return u


class WaveEquation2D(PDE):
    """
    2D Wave equation: ∂²u/∂t² = c²(∂²u/∂x² + ∂²u/∂y²)
    
    Parameters
    ----------
    c : float
        Wave speed
    """
    
    def __init__(self, c: float = 1.0):
        super().__init__()
        self.c = c
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
        Compute 2D wave equation residual.
        
        Parameters
        ----------
        u : np.ndarray
            Displacement values
        x : np.ndarray
            Points (x, y, t)
        derivatives : dict
            Contains 'dtt', 'dxx', 'dyy'
            
        Returns
        -------
        np.ndarray
            Residual: utt - c²*∇²u
        """
        utt = derivatives.get('dtt', np.zeros_like(u))
        laplacian = self.ops.laplacian(derivatives)
        
        return utt - self.c**2 * laplacian
    
    def standing_wave_solution(
        self,
        x: np.ndarray,
        n: int = 1,
        m: int = 1
    ) -> np.ndarray:
        """
        Standing wave solution on rectangular domain.
        
        u(x,y,t) = sin(nπx)sin(mπy)cos(ωt)
        where ω = cπ√(n² + m²)
        
        Parameters
        ----------
        x : np.ndarray
            Points (x, y, t)
        n : int
            Mode number in x-direction
        m : int
            Mode number in y-direction
            
        Returns
        -------
        np.ndarray
            Standing wave solution
        """
        if x.shape[1] != 3:
            raise ValueError("Expected 3D points (x, y, t)")
            
        x_coord = x[:, 0]
        y_coord = x[:, 1]
        t_coord = x[:, 2]
        
        omega = self.c * np.pi * np.sqrt(n**2 + m**2)
        
        return (
            np.sin(n * np.pi * x_coord) *
            np.sin(m * np.pi * y_coord) *
            np.cos(omega * t_coord)
        )