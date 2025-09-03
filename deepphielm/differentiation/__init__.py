"""Numerical differentiation module for DeepPhiELM."""

from .numerical_diff import NumericalDifferentiator
from .finite_difference import FiniteDifference

__all__ = ["NumericalDifferentiator", "FiniteDifference"]