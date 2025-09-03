"""Base Extreme Learning Machine implementation."""

import numpy as np
from typing import Optional, Union, Callable
from sklearn.preprocessing import StandardScaler
from scipy.linalg import pinv


class ELMBase:
    """
    Base Extreme Learning Machine class.
    
    Parameters
    ----------
    n_hidden : int
        Number of hidden neurons
    activation : str or callable
        Activation function name or callable
    random_state : int, optional
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        n_hidden: int = 100,
        activation: Union[str, Callable] = 'tanh',
        random_state: Optional[int] = None
    ):
        self.n_hidden = n_hidden
        self.activation_name = activation if isinstance(activation, str) else 'custom'
        self.activation = self._get_activation(activation)
        self.random_state = random_state
        
        self.W = None  # Input weights
        self.b = None  # Biases
        self.beta = None  # Output weights
        self.input_scaler = None
        self.output_scaler = None
        self.n_features = None
        self.n_outputs = None
        
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
        }
        
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
            
        return activations[activation]
    
    def _initialize_weights(self, n_features: int):
        """Initialize random weights using Xavier/Glorot initialization."""
        rng = np.random.RandomState(self.random_state)
        
        # Xavier initialization for better convergence
        limit = np.sqrt(6.0 / (n_features + self.n_hidden))
        self.W = rng.uniform(-limit, limit, (n_features, self.n_hidden))
        self.b = rng.uniform(-limit, limit, self.n_hidden)
        
    def _compute_hidden_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute hidden layer activation matrix."""
        G = np.dot(X, self.W) + self.b
        H = self.activation(G)
        return H
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regularization: str = 'l2',
        reg_param: float = 1e-6,
        use_pinv: bool = True
    ):
        """
        Fit the ELM model.
        
        Parameters
        ----------
        X : np.ndarray
            Training input data of shape (n_samples, n_features)
        y : np.ndarray
            Training target data of shape (n_samples, n_outputs)
        regularization : str
            Type of regularization ('l2', 'l1', or None)
        reg_param : float
            Regularization parameter
        use_pinv : bool
            Whether to use Moore-Penrose pseudoinverse
        """
        # Ensure y is 2D
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            
        # Store dimensions
        self.n_features = X.shape[1]
        self.n_outputs = y.shape[1]
        
        # Normalize data
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
        X_scaled = self.input_scaler.fit_transform(X)
        y_scaled = self.output_scaler.fit_transform(y)
        
        # Initialize weights
        self._initialize_weights(self.n_features)
        
        # Compute hidden matrix
        H = self._compute_hidden_matrix(X_scaled)
        
        # Solve for output weights
        if use_pinv:
            # Use Moore-Penrose pseudoinverse
            if regularization == 'l2':
                # Regularized pseudoinverse: β = (H^T H + λI)^(-1) H^T y
                HTH = H.T @ H
                HTH_reg = HTH + reg_param * np.eye(self.n_hidden)
                self.beta = pinv(HTH_reg) @ H.T @ y_scaled
            else:
                # Direct pseudoinverse: β = H^+ y
                self.beta = pinv(H) @ y_scaled
                
        else:
            # Traditional approach
            if regularization == 'l2':
                # Moore-Penrose pseudoinverse with L2 regularization
                HTH = H.T @ H
                HTH_reg = HTH + reg_param * np.eye(self.n_hidden)
                self.beta = np.linalg.solve(HTH_reg, H.T @ y_scaled)
            elif regularization == 'l1':
                # L1 regularization (requires iterative solver)
                from sklearn.linear_model import Lasso
                lasso = Lasso(alpha=reg_param, fit_intercept=False, max_iter=1000)
                lasso.fit(H, y_scaled.ravel() if y_scaled.shape[1] == 1 else y_scaled)
                self.beta = lasso.coef_.reshape(-1, self.n_outputs)
            else:
                # No regularization - direct pseudoinverse
                self.beta = pinv(H) @ y_scaled
            
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features)
            
        Returns
        -------
        np.ndarray
            Predictions of shape (n_samples, n_outputs)
        """
        if self.beta is None:
            raise ValueError("Model must be fitted before prediction")
            
        X_scaled = self.input_scaler.transform(X)
        H = self._compute_hidden_matrix(X_scaled)
        y_scaled = H @ self.beta
        y_pred = self.output_scaler.inverse_transform(y_scaled)
        
        # Return 1D array if original target was 1D
        if self.n_outputs == 1:
            return y_pred.ravel()
        return y_pred
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R² score.
        
        Parameters
        ----------
        X : np.ndarray
            Input data
        y : np.ndarray
            True target values
            
        Returns
        -------
        float
            R² score
        """
        y_pred = self.predict(X)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            y_pred = y_pred.reshape(-1, 1)
            
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2)
        
        return 1 - (ss_res / ss_tot)