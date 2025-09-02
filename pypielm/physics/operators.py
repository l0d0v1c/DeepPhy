"""Differential operators for PDEs."""

import numpy as np
from typing import Dict, Optional


class DifferentialOperator:
    """
    Collection of differential operators for PDE formulation.
    """
    
    @staticmethod
    def laplacian(derivatives: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute Laplacian operator (∇²u).
        
        Parameters
        ----------
        derivatives : dict
            Dictionary containing second derivatives
            
        Returns
        -------
        np.ndarray
            Laplacian values
        """
        laplacian = np.zeros_like(derivatives.get('dxx', np.array([])))
        
        if 'dxx' in derivatives:
            laplacian = derivatives['dxx']
        if 'dyy' in derivatives:
            laplacian = laplacian + derivatives['dyy']
        if 'dzz' in derivatives:
            laplacian = laplacian + derivatives['dzz']
            
        return laplacian
    
    @staticmethod
    def gradient(derivatives: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Extract gradient components (∇u).
        
        Parameters
        ----------
        derivatives : dict
            Dictionary containing first derivatives
            
        Returns
        -------
        dict
            Gradient components
        """
        gradient = {}
        
        if 'dx' in derivatives:
            gradient['x'] = derivatives['dx']
        if 'dy' in derivatives:
            gradient['y'] = derivatives['dy']
        if 'dz' in derivatives:
            gradient['z'] = derivatives['dz']
            
        return gradient
    
    @staticmethod
    def divergence(
        u_derivatives: Dict[str, np.ndarray],
        v_derivatives: Optional[Dict[str, np.ndarray]] = None,
        w_derivatives: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """
        Compute divergence (∇·F) for vector field F = (u, v, w).
        
        Parameters
        ----------
        u_derivatives : dict
            Derivatives of x-component
        v_derivatives : dict, optional
            Derivatives of y-component
        w_derivatives : dict, optional
            Derivatives of z-component
            
        Returns
        -------
        np.ndarray
            Divergence values
        """
        div = np.zeros_like(u_derivatives.get('dx', np.array([])))
        
        if 'dx' in u_derivatives:
            div = u_derivatives['dx']
        if v_derivatives and 'dy' in v_derivatives:
            div = div + v_derivatives['dy']
        if w_derivatives and 'dz' in w_derivatives:
            div = div + w_derivatives['dz']
            
        return div
    
    @staticmethod
    def advection(
        u: np.ndarray,
        derivatives: Dict[str, np.ndarray],
        velocity: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """
        Compute advection term (v·∇u).
        
        Parameters
        ----------
        u : np.ndarray
            Field values
        derivatives : dict
            Field derivatives
        velocity : dict, optional
            Velocity components. If None, uses u as velocity
            
        Returns
        -------
        np.ndarray
            Advection term values
        """
        advection = np.zeros_like(u)
        
        if velocity is None:
            # Self-advection (e.g., Burgers equation)
            if 'dx' in derivatives:
                advection = u * derivatives['dx']
            if 'dy' in derivatives:
                advection = advection + u * derivatives['dy']
        else:
            # Advection with given velocity field
            if 'x' in velocity and 'dx' in derivatives:
                advection = velocity['x'] * derivatives['dx']
            if 'y' in velocity and 'dy' in derivatives:
                advection = advection + velocity['y'] * derivatives['dy']
            if 'z' in velocity and 'dz' in derivatives:
                advection = advection + velocity['z'] * derivatives['dz']
                
        return advection
    
    @staticmethod
    def biharmonic(derivatives: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute biharmonic operator (∇⁴u).
        
        Parameters
        ----------
        derivatives : dict
            Dictionary containing fourth derivatives
            
        Returns
        -------
        np.ndarray
            Biharmonic values
        """
        biharmonic = np.zeros_like(derivatives.get('dxxxx', np.array([])))
        
        if 'dxxxx' in derivatives:
            biharmonic = derivatives['dxxxx']
        if 'dyyyy' in derivatives:
            biharmonic = biharmonic + derivatives['dyyyy']
        if 'dxxyy' in derivatives:
            biharmonic = biharmonic + 2 * derivatives['dxxyy']
            
        return biharmonic