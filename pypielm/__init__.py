"""
PyPIELM - Physics-Informed Extreme Learning Machine
====================================================

A fast and accurate framework for solving partial differential equations
using physics-informed extreme learning machines.
"""

__version__ = "0.1.0"
__author__ = "PyPIELM Team"

from pypielm.core.pielm import PIELM
from pypielm.core.elm_base import ELMBase
from pypielm.physics.pde_base import PDE

__all__ = [
    "PIELM",
    "ELMBase", 
    "PDE",
]