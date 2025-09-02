"""Navier-Stokes equations implementation."""

import numpy as np
from typing import Dict, Optional, Tuple
from ..pde_base import PDE
from ..operators import DifferentialOperator


class NavierStokes2D(PDE):
    """
    2D incompressible Navier-Stokes equations:
    ∂u/∂t + u·∇u = -∇p + ν∇²u
    ∇·u = 0
    
    Parameters
    ----------
    nu : float
        Kinematic viscosity
    rho : float
        Fluid density (default 1 for normalized equations)
    """
    
    def __init__(self, nu: float = 0.01, rho: float = 1.0):
        super().__init__()
        self.nu = nu
        self.rho = rho
        self.dimension = 3  # (x, y, t) for 2D flow
        self.order = 2
        self.ops = DifferentialOperator()
        
    def residual(
        self,
        solution: np.ndarray,
        x: np.ndarray,
        derivatives: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Compute Navier-Stokes residual.
        
        Parameters
        ----------
        solution : np.ndarray
            Solution [u, v, p] where u,v are velocity components and p is pressure
        x : np.ndarray
            Points (x, y, t)
        derivatives : dict
            Contains all necessary derivatives
            
        Returns
        -------
        np.ndarray
            Residuals for momentum and continuity equations
        """
        # Extract solution components
        if solution.shape[1] != 3:
            raise ValueError("Solution must have 3 components: [u, v, p]")
            
        u = solution[:, 0]
        v = solution[:, 1]
        p = solution[:, 2]
        
        # Time derivatives
        u_t = derivatives.get('u_t', np.zeros_like(u))
        v_t = derivatives.get('v_t', np.zeros_like(v))
        
        # Spatial derivatives
        u_x = derivatives.get('u_x', np.zeros_like(u))
        u_y = derivatives.get('u_y', np.zeros_like(u))
        u_xx = derivatives.get('u_xx', np.zeros_like(u))
        u_yy = derivatives.get('u_yy', np.zeros_like(u))
        
        v_x = derivatives.get('v_x', np.zeros_like(v))
        v_y = derivatives.get('v_y', np.zeros_like(v))
        v_xx = derivatives.get('v_xx', np.zeros_like(v))
        v_yy = derivatives.get('v_yy', np.zeros_like(v))
        
        p_x = derivatives.get('p_x', np.zeros_like(p))
        p_y = derivatives.get('p_y', np.zeros_like(p))
        
        # Momentum equations
        momentum_x = u_t + u * u_x + v * u_y + p_x / self.rho - self.nu * (u_xx + u_yy)
        momentum_y = v_t + u * v_x + v * v_y + p_y / self.rho - self.nu * (v_xx + v_yy)
        
        # Continuity equation (incompressibility)
        continuity = u_x + v_y
        
        # Stack residuals
        residuals = np.column_stack([momentum_x, momentum_y, continuity])
        
        return residuals
    
    def taylor_green_vortex(
        self,
        x: np.ndarray,
        t: float,
        L: float = 2 * np.pi
    ) -> Dict[str, np.ndarray]:
        """
        Taylor-Green vortex exact solution.
        
        Parameters
        ----------
        x : np.ndarray
            Points (x, y)
        t : float
            Time
        L : float
            Domain size
            
        Returns
        -------
        dict
            Contains 'u', 'v', 'p' components
        """
        x_coord = x[:, 0]
        y_coord = x[:, 1]
        
        # Decay factor
        F = np.exp(-2 * self.nu * t * (2 * np.pi / L)**2)
        
        # Velocity components
        u = -np.cos(2 * np.pi * x_coord / L) * np.sin(2 * np.pi * y_coord / L) * F
        v = np.sin(2 * np.pi * x_coord / L) * np.cos(2 * np.pi * y_coord / L) * F
        
        # Pressure
        p = -self.rho / 4 * (
            np.cos(4 * np.pi * x_coord / L) + np.cos(4 * np.pi * y_coord / L)
        ) * F**2
        
        return {'u': u, 'v': v, 'p': p}
    
    def lid_driven_cavity_bc(
        self,
        x: np.ndarray,
        U_lid: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """
        Boundary conditions for lid-driven cavity problem.
        
        Parameters
        ----------
        x : np.ndarray
            Boundary points
        U_lid : float
            Lid velocity
            
        Returns
        -------
        dict
            Boundary values for u, v
        """
        u_bc = np.zeros(x.shape[0])
        v_bc = np.zeros(x.shape[0])
        
        # Top boundary (y = 1): u = U_lid, v = 0
        top_mask = np.abs(x[:, 1] - 1.0) < 1e-10
        u_bc[top_mask] = U_lid
        
        # All boundaries: no-slip (v = 0 everywhere, u = 0 except top)
        
        return {'u': u_bc, 'v': v_bc}