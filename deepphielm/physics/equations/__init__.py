"""Common PDE implementations."""

from .heat import HeatEquation1D, HeatEquation2D
from .wave import WaveEquation1D, WaveEquation2D
from .burgers import BurgersEquation
from .poisson import PoissonEquation2D
from .schrodinger import SchrodingerEquation
from .navier_stokes import NavierStokes2D

__all__ = [
    "HeatEquation1D",
    "HeatEquation2D",
    "WaveEquation1D",
    "WaveEquation2D",
    "BurgersEquation",
    "PoissonEquation2D",
    "SchrodingerEquation",
    "NavierStokes2D",
]