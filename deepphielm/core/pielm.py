"""Physics-Informed Extreme Learning Machine implementation."""

import numpy as np
from typing import Optional, Dict, Tuple, Union, Callable
from sklearn.preprocessing import StandardScaler
from scipy.linalg import pinv
import warnings

from .elm_base import ELMBase
from .layers import ActivationLayer
from ..physics.pde_base import PDE
from ..solvers.linear_solver import LinearSolver
from ..utils.sampling import SamplingStrategy
from ..differentiation.numerical_diff import NumericalDifferentiator


class PIELM(ELMBase):
    """
    Physics-Informed Extreme Learning Machine.
    
    Combines ELM architecture with physics constraints for solving PDEs
    using numerical differentiation instead of automatic differentiation.
    
    Parameters
    ----------
    n_hidden : int
        Number of hidden neurons
    activation : str or callable
        Activation function
    pde : PDE, optional
        PDE object defining the physics constraints
    lambda_data : float
        Weight for data loss term
    lambda_physics : float
        Weight for physics residual term
    lambda_bc : float
        Weight for boundary condition term
    lambda_ic : float
        Weight for initial condition term
    regularization : str
        Type of regularization ('l2', 'l1', None)
    reg_param : float
        Regularization parameter
    random_state : int, optional
        Random seed for reproducibility
    diff_step : float
        Step size for numerical differentiation
    diff_method : str
        Finite difference method ('central', 'forward', 'backward')
    """
    
    def __init__(
        self,
        n_hidden: int = 100,
        activation: Union[str, Callable] = 'tanh',
        pde: Optional[PDE] = None,
        lambda_data: float = 1.0,
        lambda_physics: float = 1.0,
        lambda_bc: float = 1.0,
        lambda_ic: float = 1.0,
        regularization: str = 'l2',
        reg_param: float = 1e-6,
        random_state: Optional[int] = None,
        diff_step: float = 1e-6,
        diff_method: str = 'central'
    ):
        super().__init__(n_hidden, activation, random_state)
        
        self.pde = pde
        self.lambda_data = lambda_data
        self.lambda_physics = lambda_physics
        self.lambda_bc = lambda_bc
        self.lambda_ic = lambda_ic
        self.regularization = regularization
        self.reg_param = reg_param
        
        # Numerical differentiation
        self.differentiator = NumericalDifferentiator(h=diff_step, method=diff_method)
        
        # Additional PIELM-specific attributes
        self.activation_layer = ActivationLayer(activation)
        self.solver = LinearSolver()
        self.sampling_strategy = SamplingStrategy()
        
        # Training history
        self.loss_history = {
            'data': [],
            'physics': [],
            'bc': [],
            'ic': [],
            'total': []
        }
        
    def _compute_derivatives_matrix(
        self,
        X: np.ndarray,
        order: int = 2
    ) -> Dict[str, np.ndarray]:
        """
        Compute derivatives of network output using numerical differentiation.
        
        Parameters
        ----------
        X : np.ndarray
            Input points
        order : int
            Maximum order of derivatives to compute
            
        Returns
        -------
        dict
            Dictionary containing derivatives
        """
        # Create a temporary model for differentiation
        class TempModel:
            def __init__(self, pielm_model):
                self.pielm_model = pielm_model
            
            def predict(self, X_input):
                # Use internal prediction without scaling
                X_scaled = self.pielm_model.input_scaler.transform(X_input)
                H = self.pielm_model._compute_hidden_matrix(X_scaled)
                y_scaled = H @ self.pielm_model.beta
                return self.pielm_model.output_scaler.inverse_transform(y_scaled).flatten()
        
        temp_model = TempModel(self)
        
        # Compute derivatives using numerical differentiation
        derivatives = self.differentiator.compute_derivatives(temp_model, X, order)
        
        return derivatives
    
    def _build_physics_matrix(
        self,
        H: np.ndarray,
        derivatives: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Build matrix for physics residual computation.
        
        For numerical differentiation, we don't build a linear matrix directly.
        Instead, we'll compute the physics residual using the current network.
        This returns the hidden matrix H for now, but physics constraints
        are enforced through the residual computation.
        """
        return H
    
    def _construct_augmented_system(
        self,
        X_data: np.ndarray,
        y_data: np.ndarray,
        X_collocation: np.ndarray,
        X_bc: Optional[np.ndarray] = None,
        y_bc: Optional[np.ndarray] = None,
        X_ic: Optional[np.ndarray] = None,
        y_ic: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct linear system with data, boundary, and initial conditions.
        Physics constraints are handled separately in iterative refinement.
        """
        # Compute hidden matrices
        H_data = self._compute_hidden_matrix(X_data)
        
        # Initialize system matrices with data term
        n_output = y_data.shape[1] if len(y_data.shape) > 1 else 1
        A = self.lambda_data * H_data.T @ H_data
        b = self.lambda_data * H_data.T @ y_data
        
        # Add boundary conditions
        if X_bc is not None and y_bc is not None:
            H_bc = self._compute_hidden_matrix(X_bc)
            if len(y_bc.shape) == 1:
                y_bc = y_bc.reshape(-1, 1)
            A += self.lambda_bc * H_bc.T @ H_bc
            b += self.lambda_bc * H_bc.T @ y_bc
        
        # Add initial conditions
        if X_ic is not None and y_ic is not None:
            H_ic = self._compute_hidden_matrix(X_ic)
            if len(y_ic.shape) == 1:
                y_ic = y_ic.reshape(-1, 1)
            A += self.lambda_ic * H_ic.T @ H_ic
            b += self.lambda_ic * H_ic.T @ y_ic
        
        # Add regularization
        if self.regularization == 'l2':
            A += self.reg_param * np.eye(A.shape[0])
        
        return A, b
    
    def _physics_penalty_update(
        self,
        X_collocation: np.ndarray,
        current_residual: np.ndarray,
        learning_rate: float = 0.1
    ) -> np.ndarray:
        """
        Compute penalty update for physics residual using gradient-based approach.
        """
        if self.pde is None:
            return np.zeros_like(self.beta)
        
        # Compute hidden matrix for collocation points
        H_coll = self._compute_hidden_matrix(X_collocation)
        
        # Ensure residual is the right shape
        if len(current_residual.shape) == 1:
            current_residual = current_residual.reshape(-1, 1)
        
        # Simple gradient-like penalty: penalize high residuals
        residual_flat = current_residual.flatten()
        residual_weights = np.abs(residual_flat)
        residual_weights = residual_weights / (np.max(residual_weights) + 1e-8)
        
        # Weighted penalty term - make sure dimensions match
        weighted_residual = (residual_weights * residual_flat).reshape(-1, 1)
        penalty = learning_rate * self.lambda_physics * H_coll.T @ weighted_residual
        
        # Ensure penalty has same shape as beta
        if penalty.shape != self.beta.shape:
            if len(self.beta.shape) == 2 and self.beta.shape[1] == 1:
                penalty = penalty.reshape(-1, 1)
            else:
                penalty = penalty.flatten()
        
        return penalty
    
    def fit(
        self,
        X_data: np.ndarray,
        y_data: np.ndarray,
        X_collocation: Optional[np.ndarray] = None,
        X_bc: Optional[np.ndarray] = None,
        y_bc: Optional[np.ndarray] = None,
        X_ic: Optional[np.ndarray] = None,
        y_ic: Optional[np.ndarray] = None,
        n_collocation: int = 1000,
        collocation_strategy: str = 'latin_hypercube',
        max_physics_iterations: int = 3
    ):
        """
        Fit the PIELM model using numerical differentiation.
        
        Parameters
        ----------
        max_physics_iterations : int
            Maximum iterations for physics constraint refinement
        """
        # Ensure y is 2D
        if len(y_data.shape) == 1:
            y_data = y_data.reshape(-1, 1)
        
        # Store dimensions
        self.n_features = X_data.shape[1]
        self.n_outputs = y_data.shape[1]
        
        # Normalize data
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
        X_data_scaled = self.input_scaler.fit_transform(X_data)
        y_data_scaled = self.output_scaler.fit_transform(y_data)
        
        # Initialize weights
        self._initialize_weights(self.n_features)
        
        # Generate collocation points if not provided
        if X_collocation is None:
            bounds = self._infer_bounds(X_data)
            X_collocation = self.sampling_strategy.generate_points(
                n_collocation, bounds, strategy=collocation_strategy
            )
        X_collocation_scaled = self.input_scaler.transform(X_collocation)
        
        # Scale boundary and initial conditions if provided
        X_bc_scaled = None
        y_bc_scaled = None
        if X_bc is not None and y_bc is not None:
            X_bc_scaled = self.input_scaler.transform(X_bc)
            if len(y_bc.shape) == 1:
                y_bc = y_bc.reshape(-1, 1)
            y_bc_scaled = self.output_scaler.transform(y_bc)
        
        X_ic_scaled = None
        y_ic_scaled = None
        if X_ic is not None and y_ic is not None:
            X_ic_scaled = self.input_scaler.transform(X_ic)
            if len(y_ic.shape) == 1:
                y_ic = y_ic.reshape(-1, 1)
            y_ic_scaled = self.output_scaler.transform(y_ic)
        
        # Step 1: Solve base system (data + BC + IC)
        A, b = self._construct_augmented_system(
            X_data_scaled, y_data_scaled,
            X_collocation_scaled,
            X_bc_scaled, y_bc_scaled,
            X_ic_scaled, y_ic_scaled
        )
        
        # Initial solution
        if self.regularization == 'l2':
            A_reg = A + self.reg_param * np.eye(A.shape[0])
            self.beta = np.linalg.solve(A_reg, b) if np.linalg.cond(A_reg) < 1e12 else pinv(A_reg) @ b
        else:
            self.beta = np.linalg.solve(A, b) if np.linalg.cond(A) < 1e12 else pinv(A) @ b
        
        # Step 2: Physics-informed refinement using augmented system
        if self.pde is not None and self.lambda_physics > 0:
            prev_residual = float('inf')
            
            for iteration in range(max_physics_iterations):
                # Compute current physics residual
                residual = self.compute_physics_residual(X_collocation)
                avg_residual = np.mean(np.abs(residual))
                
                if avg_residual < 1e-3:  # Looser convergence criterion
                    break
                
                # Re-solve augmented system with physics penalty
                H_coll = self._compute_hidden_matrix(X_collocation_scaled)
                
                # Add physics penalty to system - use residual as target
                physics_penalty_weight = self.lambda_physics / (1.0 + avg_residual)  # Adaptive weight
                A_phys = A + physics_penalty_weight * H_coll.T @ H_coll
                
                # Target is to minimize residual, so we add a small penalty toward zero residual
                physics_target = -0.1 * residual.reshape(-1, 1)  # Small correction toward zero residual
                b_phys = b + physics_penalty_weight * H_coll.T @ physics_target
                
                # Solve updated system
                if self.regularization == 'l2':
                    A_phys += self.reg_param * np.eye(A_phys.shape[0])
                
                try:
                    self.beta = np.linalg.solve(A_phys, b_phys) if np.linalg.cond(A_phys) < 1e12 else pinv(A_phys) @ b_phys
                except np.linalg.LinAlgError:
                    # Fallback to pseudoinverse if solve fails
                    self.beta = pinv(A_phys) @ b_phys
                
                # Check for improvement
                if iteration > 0 and avg_residual > 1.1 * prev_residual:
                    break  # Stop if getting significantly worse
                
                prev_residual = avg_residual
        
        # Validate solution
        self._validate_solution(X_data_scaled, y_data_scaled)
        
        return self
    
    def _infer_bounds(self, X: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """Infer domain bounds from data."""
        bounds = {}
        dim_names = ['x', 'y', 'z', 't'] if X.shape[1] <= 4 else [f'x{i}' for i in range(X.shape[1])]
        
        for i in range(X.shape[1]):
            bounds[dim_names[i]] = (X[:, i].min(), X[:, i].max())
        return bounds
    
    def _validate_solution(self, X: np.ndarray, y: np.ndarray):
        """Validate the fitted solution."""
        # Compute training error
        H = self._compute_hidden_matrix(X)
        y_pred = H @ self.beta
        mse = np.mean((y_pred - y) ** 2)
        
        if mse > 1e3:
            warnings.warn(f"High training MSE: {mse:.2e}. Consider adjusting hyperparameters.")
    
    def compute_physics_residual(self, X: np.ndarray) -> np.ndarray:
        """
        Compute physics residual at given points using numerical differentiation.
        
        Parameters
        ----------
        X : np.ndarray
            Input points
            
        Returns
        -------
        np.ndarray
            Physics residual values
        """
        if self.pde is None:
            raise ValueError("No PDE defined")
        
        # Get model predictions
        u = self.predict(X)
        if len(u.shape) == 1:
            u = u.reshape(-1, 1)
        
        # Compute derivatives using numerical differentiation
        derivatives = self._compute_derivatives_matrix(X, order=2)
        
        # Compute PDE residual
        residual = self.pde.residual(u, X, derivatives)
        
        return residual
    
    def compute_loss(
        self,
        X_data: np.ndarray,
        y_data: np.ndarray,
        X_collocation: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute all loss components.
        
        Parameters
        ----------
        X_data : np.ndarray
            Data points
        y_data : np.ndarray
            Target values
        X_collocation : np.ndarray, optional
            Collocation points
            
        Returns
        -------
        dict
            Dictionary of loss values
        """
        losses = {}
        
        # Data loss
        y_pred = self.predict(X_data)
        if len(y_data.shape) == 1:
            y_data = y_data.reshape(-1, 1)
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape(-1, 1)
        losses['data'] = np.mean((y_pred - y_data) ** 2)
        
        # Physics loss
        if X_collocation is not None and self.pde is not None:
            residual = self.compute_physics_residual(X_collocation)
            losses['physics'] = np.mean(residual ** 2)
        else:
            losses['physics'] = 0.0
        
        # Total weighted loss
        losses['total'] = (
            self.lambda_data * losses['data'] +
            self.lambda_physics * losses['physics']
        )
        
        return losses
    
    def set_differentiation_step(self, h: float):
        """Set the step size for numerical differentiation."""
        self.differentiator.h = h
    
    def get_differentiation_info(self) -> Dict[str, Union[float, str]]:
        """Get information about numerical differentiation settings."""
        return {
            'step_size': self.differentiator.h,
            'method': self.differentiator.method,
            'type': 'numerical'
        }