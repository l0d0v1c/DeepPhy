"""
DeepPhiELM - Physics-Informed Extreme Learning Machine
======================================================

A fast and accurate framework for solving partial differential equations
using physics-informed extreme learning machines with numerical differentiation.
"""

__version__ = "0.1.0"
__author__ = "DeepPhiELM Team"

from deepphielm.core.pielm import PIELM
from deepphielm.core.elm_base import ELMBase
from deepphielm.physics.pde_base import PDE

__all__ = [
    "PIELM",
    "ELMBase", 
    "PDE",
]