"""Boundary condition implementations."""

import numpy as np
from typing import Callable, Optional, Union
from abc import ABC, abstractmethod


class BoundaryCondition(ABC):
    """Abstract base class for boundary conditions."""
    
    def __init__(self, value: Union[float, Callable]):
        """
        Initialize boundary condition.
        
        Parameters
        ----------
        value : float or callable
            Boundary value or function
        """
        self.value = value
        
    @abstractmethod
    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply boundary condition at points x."""
        pass
    
    @abstractmethod
    def get_type(self) -> str:
        """Return boundary condition type."""
        pass


class DirichletBC(BoundaryCondition):
    """
    Dirichlet boundary condition: u = g on boundary.
    """
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        Apply Dirichlet condition.
        
        Parameters
        ----------
        x : np.ndarray
            Boundary points
            
        Returns
        -------
        np.ndarray
            Boundary values
        """
        if callable(self.value):
            return self.value(x)
        else:
            return np.full(x.shape[0], self.value)
    
    def get_type(self) -> str:
        return 'dirichlet'


class NeumannBC(BoundaryCondition):
    """
    Neumann boundary condition: ∂u/∂n = g on boundary.
    """
    
    def __init__(self, value: Union[float, Callable], normal: Optional[np.ndarray] = None):
        """
        Initialize Neumann condition.
        
        Parameters
        ----------
        value : float or callable
            Normal derivative value
        normal : np.ndarray, optional
            Normal vectors at boundary
        """
        super().__init__(value)
        self.normal = normal
        
    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        Apply Neumann condition.
        
        Parameters
        ----------
        x : np.ndarray
            Boundary points
            
        Returns
        -------
        np.ndarray
            Normal derivative values
        """
        if callable(self.value):
            return self.value(x)
        else:
            return np.full(x.shape[0], self.value)
    
    def get_type(self) -> str:
        return 'neumann'


class RobinBC(BoundaryCondition):
    """
    Robin boundary condition: au + b∂u/∂n = g on boundary.
    """
    
    def __init__(
        self,
        a: Union[float, Callable],
        b: Union[float, Callable],
        g: Union[float, Callable]
    ):
        """
        Initialize Robin condition.
        
        Parameters
        ----------
        a : float or callable
            Coefficient for u
        b : float or callable
            Coefficient for ∂u/∂n
        g : float or callable
            Right-hand side value
        """
        super().__init__(g)
        self.a = a
        self.b = b
        
    def apply(self, x: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply Robin condition.
        
        Parameters
        ----------
        x : np.ndarray
            Boundary points
        u : np.ndarray, optional
            Solution values at boundary
            
        Returns
        -------
        np.ndarray
            Robin condition values
        """
        # This requires both u and ∂u/∂n, typically handled in the solver
        if callable(self.value):
            return self.value(x)
        else:
            return np.full(x.shape[0], self.value)
    
    def get_type(self) -> str:
        return 'robin'


class PeriodicBC(BoundaryCondition):
    """
    Periodic boundary condition: u(x=0) = u(x=L).
    """
    
    def __init__(self, period: float):
        """
        Initialize periodic condition.
        
        Parameters
        ----------
        period : float
            Period of the domain
        """
        super().__init__(0.0)
        self.period = period
        
    def apply(self, x: np.ndarray) -> np.ndarray:
        """
        Apply periodic condition.
        
        Parameters
        ----------
        x : np.ndarray
            Boundary points
            
        Returns
        -------
        np.ndarray
            Periodic mapping
        """
        # Map points to their periodic equivalents
        return x % self.period
    
    def get_type(self) -> str:
        return 'periodic'