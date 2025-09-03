"""Base class for partial differential equations."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Tuple, Optional, List


class PDE(ABC):
    """
    Abstract base class for partial differential equations.
    
    All PDE implementations should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self):
        self.dimension = None
        self.order = None
        self.boundary_conditions = []
        self.initial_conditions = []
        
    @abstractmethod
    def residual(
        self,
        u: np.ndarray,
        x: np.ndarray,
        derivatives: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Compute the PDE residual.
        
        Parameters
        ----------
        u : np.ndarray
            Solution values at points x
        x : np.ndarray
            Spatial/temporal coordinates
        derivatives : dict
            Dictionary containing derivatives (e.g., 'dx', 'dxx', 'dt')
            
        Returns
        -------
        np.ndarray
            Residual values
        """
        pass
    
    def boundary_conditions(self, x: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Define boundary conditions.
        
        Parameters
        ----------
        x : np.ndarray
            Boundary points
            
        Returns
        -------
        tuple
            (values, type) where type is 'dirichlet', 'neumann', or 'mixed'
        """
        return np.zeros(x.shape[0]), 'dirichlet'
    
    def initial_conditions(self, x: np.ndarray) -> np.ndarray:
        """
        Define initial conditions.
        
        Parameters
        ----------
        x : np.ndarray
            Initial points
            
        Returns
        -------
        np.ndarray
            Initial values
        """
        return np.zeros(x.shape[0])
    
    def source_term(self, x: np.ndarray) -> np.ndarray:
        """
        Source/forcing term in the PDE.
        
        Parameters
        ----------
        x : np.ndarray
            Evaluation points
            
        Returns
        -------
        np.ndarray
            Source term values
        """
        return np.zeros(x.shape[0])
    
    def exact_solution(self, x: np.ndarray) -> Optional[np.ndarray]:
        """
        Exact/analytical solution if available.
        
        Parameters
        ----------
        x : np.ndarray
            Evaluation points
            
        Returns
        -------
        np.ndarray or None
            Exact solution values if available
        """
        return None
    
    def validate_dimensions(self, x: np.ndarray):
        """Validate input dimensions."""
        if self.dimension is not None:
            if x.shape[1] != self.dimension:
                raise ValueError(
                    f"Expected {self.dimension} dimensions, got {x.shape[1]}"
                )