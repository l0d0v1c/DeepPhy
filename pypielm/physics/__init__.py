"""Physics module for PyPIELM."""

from .pde_base import PDE
from .operators import DifferentialOperator
from .boundary import BoundaryCondition, DirichletBC, NeumannBC

__all__ = [
    "PDE",
    "DifferentialOperator",
    "BoundaryCondition",
    "DirichletBC",
    "NeumannBC",
]