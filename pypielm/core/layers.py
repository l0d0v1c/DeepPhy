"""Activation layers and functions for PyPIELM."""

import numpy as np
import torch
from typing import Callable, Optional, Union


class ActivationLayer:
    """
    Activation layer with support for derivatives.
    
    Parameters
    ----------
    activation : str or callable
        Activation function name or callable
    """
    
    def __init__(self, activation: Union[str, Callable] = 'tanh'):
        self.activation_name = activation if isinstance(activation, str) else 'custom'
        self.activation = self._get_activation(activation)
        self.activation_deriv = self._get_activation_derivative(activation)
        self.activation_deriv2 = self._get_activation_second_derivative(activation)
        
    def _get_activation(self, activation: Union[str, Callable]) -> Callable:
        """Get activation function."""
        if callable(activation):
            return activation
            
        activations = {
            'tanh': np.tanh,
            'sigmoid': lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500))),
            'relu': lambda x: np.maximum(0, x),
            'swish': lambda x: x / (1 + np.exp(-np.clip(x, -500, 500))),
            'sin': np.sin,
            'cos': np.cos,
            'gaussian': lambda x: np.exp(-x**2),
        }
        
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
            
        return activations[activation]
    
    def _get_activation_derivative(self, activation: Union[str, Callable]) -> Callable:
        """Get first derivative of activation function."""
        if callable(activation) and not isinstance(activation, str):
            # For custom functions, use numerical differentiation
            def numerical_deriv(x):
                h = 1e-7
                return (activation(x + h) - activation(x - h)) / (2 * h)
            return numerical_deriv
            
        derivatives = {
            'tanh': lambda x: 1 - np.tanh(x)**2,
            'sigmoid': lambda x: (s := 1 / (1 + np.exp(-np.clip(x, -500, 500)))) * (1 - s),
            'relu': lambda x: (x > 0).astype(float),
            'swish': lambda x: (s := 1 / (1 + np.exp(-np.clip(x, -500, 500)))) * (1 + x * s * (1 - s)),
            'sin': np.cos,
            'cos': lambda x: -np.sin(x),
            'gaussian': lambda x: -2 * x * np.exp(-x**2),
        }
        
        if activation not in derivatives:
            raise ValueError(f"Unknown activation derivative: {activation}")
            
        return derivatives[activation]
    
    def _get_activation_second_derivative(self, activation: Union[str, Callable]) -> Callable:
        """Get second derivative of activation function."""
        if callable(activation) and not isinstance(activation, str):
            # For custom functions, use numerical differentiation
            def numerical_deriv2(x):
                h = 1e-5
                return (activation(x + h) - 2 * activation(x) + activation(x - h)) / h**2
            return numerical_deriv2
            
        second_derivatives = {
            'tanh': lambda x: -2 * np.tanh(x) * (1 - np.tanh(x)**2),
            'sigmoid': lambda x: (s := 1 / (1 + np.exp(-np.clip(x, -500, 500)))) * (1 - s) * (1 - 2*s),
            'relu': lambda x: np.zeros_like(x),
            'swish': lambda x: self._swish_second_derivative(x),
            'sin': lambda x: -np.sin(x),
            'cos': lambda x: -np.cos(x),
            'gaussian': lambda x: (4 * x**2 - 2) * np.exp(-x**2),
        }
        
        if activation not in second_derivatives:
            raise ValueError(f"Unknown activation second derivative: {activation}")
            
        return second_derivatives[activation]
    
    def _swish_second_derivative(self, x):
        """Compute second derivative of swish function."""
        s = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        s_prime = s * (1 - s)
        s_double_prime = s_prime * (1 - 2*s)
        return s_double_prime * (1 + x) + 2 * s_prime * (1 - s)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through activation."""
        return self.activation(x)
    
    def derivative(self, x: np.ndarray, order: int = 1) -> np.ndarray:
        """
        Compute derivative of activation.
        
        Parameters
        ----------
        x : np.ndarray
            Input values
        order : int
            Order of derivative (1 or 2)
            
        Returns
        -------
        np.ndarray
            Derivative values
        """
        if order == 1:
            return self.activation_deriv(x)
        elif order == 2:
            return self.activation_deriv2(x)
        else:
            raise ValueError(f"Derivative order {order} not supported")
    
    def torch_activation(self, name: str) -> Callable:
        """Get PyTorch version of activation for autodiff."""
        torch_activations = {
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid,
            'relu': torch.relu,
            'swish': lambda x: x * torch.sigmoid(x),
            'sin': torch.sin,
            'cos': torch.cos,
            'gaussian': lambda x: torch.exp(-x**2),
        }
        
        if name not in torch_activations:
            raise ValueError(f"Unknown torch activation: {name}")
            
        return torch_activations[name]