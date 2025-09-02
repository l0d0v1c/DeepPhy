"""Schrödinger equation implementation."""

import numpy as np
from typing import Dict, Optional, Union
from ..pde_base import PDE
from ..operators import DifferentialOperator


class SchrodingerEquation(PDE):
    """
    Time-dependent Schrödinger equation: iħ∂ψ/∂t = -ħ²/(2m)∇²ψ + Vψ
    
    Parameters
    ----------
    hbar : float
        Reduced Planck constant (set to 1 for normalized units)
    mass : float
        Particle mass (set to 1 for normalized units)
    potential : callable or float
        Potential energy V(x)
    """
    
    def __init__(
        self,
        hbar: float = 1.0,
        mass: float = 1.0,
        potential: Union[float, callable] = None
    ):
        super().__init__()
        self.hbar = hbar
        self.mass = mass
        self.potential = potential
        self.dimension = None  # Depends on problem
        self.order = 2
        self.ops = DifferentialOperator()
        
    def residual(
        self,
        psi: np.ndarray,
        x: np.ndarray,
        derivatives: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Compute Schrödinger equation residual.
        
        Note: psi should be complex-valued in general.
        
        Parameters
        ----------
        psi : np.ndarray
            Wave function (can be complex)
        x : np.ndarray
            Points (x, t) or (x, y, t)
        derivatives : dict
            Contains time and spatial derivatives
            
        Returns
        -------
        np.ndarray
            Residual: iħ∂ψ/∂t + ħ²/(2m)∇²ψ - Vψ
        """
        # Time derivative
        psi_t = derivatives.get('dt', np.zeros_like(psi))
        
        # Laplacian (kinetic energy term)
        laplacian = self.ops.laplacian(derivatives)
        
        # Potential energy
        V = self.potential_energy(x)
        
        # Schrödinger equation residual
        residual = (
            1j * self.hbar * psi_t +
            (self.hbar**2 / (2 * self.mass)) * laplacian -
            V * psi
        )
        
        return residual
    
    def potential_energy(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate potential energy.
        
        Parameters
        ----------
        x : np.ndarray
            Spatial points
            
        Returns
        -------
        np.ndarray
            Potential values
        """
        if self.potential is None:
            return np.zeros(x.shape[0])
        elif callable(self.potential):
            return self.potential(x)
        else:
            return np.full(x.shape[0], self.potential)
    
    def gaussian_wave_packet(
        self,
        x: np.ndarray,
        x0: float,
        k0: float,
        sigma: float
    ) -> np.ndarray:
        """
        Gaussian wave packet initial condition.
        
        ψ(x,0) = (2πσ²)^(-1/4) * exp(ik₀x) * exp(-(x-x₀)²/(4σ²))
        
        Parameters
        ----------
        x : np.ndarray
            Spatial points
        x0 : float
            Center position
        k0 : float
            Initial momentum
        sigma : float
            Width parameter
            
        Returns
        -------
        np.ndarray
            Complex wave function
        """
        norm = (2 * np.pi * sigma**2) ** (-0.25)
        gaussian = np.exp(-(x - x0)**2 / (4 * sigma**2))
        plane_wave = np.exp(1j * k0 * x)
        
        return norm * gaussian * plane_wave
    
    def harmonic_oscillator_eigenstate(
        self,
        x: np.ndarray,
        n: int,
        omega: float = 1.0
    ) -> np.ndarray:
        """
        Eigenstate of quantum harmonic oscillator.
        
        Parameters
        ----------
        x : np.ndarray
            Spatial points
        n : int
            Quantum number (n = 0, 1, 2, ...)
        omega : float
            Oscillator frequency
            
        Returns
        -------
        np.ndarray
            Eigenstate wave function
        """
        from scipy.special import hermite
        from math import factorial
        
        # Length scale
        alpha = np.sqrt(self.mass * omega / self.hbar)
        
        # Normalization
        norm = np.sqrt(
            alpha / (np.sqrt(np.pi) * 2**n * factorial(n))
        )
        
        # Hermite polynomial
        Hn = hermite(n)
        
        # Wave function
        psi = norm * np.exp(-alpha**2 * x**2 / 2) * Hn(alpha * x)
        
        return psi