"""Physics-Informed Extreme Learning Machine implementation."""

import numpy as np
import torch
from typing import Optional, Dict, Tuple, Union, Callable
from sklearn.preprocessing import StandardScaler
import warnings

from .elm_base import ELMBase
from .layers import ActivationLayer
from ..physics.pde_base import PDE
from ..solvers.linear_solver import LinearSolver
from ..utils.sampling import SamplingStrategy


class PIELM(ELMBase):
    """
    Physics-Informed Extreme Learning Machine.
    
    Combines ELM architecture with physics constraints for solving PDEs.
    
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
        random_state: Optional[int] = None
    ):
        super().__init__(n_hidden, activation, random_state)
        
        self.pde = pde
        self.lambda_data = lambda_data
        self.lambda_physics = lambda_physics
        self.lambda_bc = lambda_bc
        self.lambda_ic = lambda_ic
        self.regularization = regularization
        self.reg_param = reg_param
        
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
        Compute derivatives of hidden layer using automatic differentiation.
        
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
        X_torch = torch.tensor(X, requires_grad=True, dtype=torch.float32)
        W_torch = torch.tensor(self.W, dtype=torch.float32)
        b_torch = torch.tensor(self.b, dtype=torch.float32)
        
        # Forward pass
        G = torch.matmul(X_torch, W_torch) + b_torch
        
        # Get torch version of activation
        torch_activation = self.activation_layer.torch_activation(self.activation_name)
        H = torch_activation(G)
        
        derivatives = {'H': H.detach().numpy()}
        
        # Compute derivatives for each output dimension
        n_outputs = H.shape[1]
        
        if order >= 1:
            # First-order derivatives
            dx_list = []
            dy_list = []
            dt_list = []
            
            for i in range(n_outputs):
                H_i = H[:, i].sum()
                grad = torch.autograd.grad(
                    H_i, X_torch, create_graph=True, retain_graph=True
                )[0]
                
                # Assume X has dimensions [x, y, t] or similar
                if X.shape[1] >= 1:
                    dx_list.append(grad[:, 0].detach().numpy())
                if X.shape[1] >= 2:
                    dy_list.append(grad[:, 1].detach().numpy())
                if X.shape[1] >= 3:
                    dt_list.append(grad[:, 2].detach().numpy())
            
            if dx_list:
                derivatives['dx'] = np.column_stack(dx_list)
            if dy_list:
                derivatives['dy'] = np.column_stack(dy_list)
            if dt_list:
                derivatives['dt'] = np.column_stack(dt_list)
        
        if order >= 2:
            # Second-order derivatives
            dxx_list = []
            dyy_list = []
            dtt_list = []
            
            for i in range(n_outputs):
                H_i = H[:, i].sum()
                grad_1 = torch.autograd.grad(
                    H_i, X_torch, create_graph=True, retain_graph=True
                )[0]
                
                # Second derivatives
                if X.shape[1] >= 1:
                    grad_1_x = grad_1[:, 0].sum()
                    grad_2_x = torch.autograd.grad(
                        grad_1_x, X_torch, retain_graph=True
                    )[0]
                    dxx_list.append(grad_2_x[:, 0].detach().numpy())
                    
                if X.shape[1] >= 2:
                    grad_1_y = grad_1[:, 1].sum()
                    grad_2_y = torch.autograd.grad(
                        grad_1_y, X_torch, retain_graph=True
                    )[0]
                    dyy_list.append(grad_2_y[:, 1].detach().numpy())
                    
                if X.shape[1] >= 3:
                    grad_1_t = grad_1[:, 2].sum()
                    grad_2_t = torch.autograd.grad(
                        grad_1_t, X_torch, retain_graph=True
                    )[0]
                    dtt_list.append(grad_2_t[:, 2].detach().numpy())
            
            if dxx_list:
                derivatives['dxx'] = np.column_stack(dxx_list)
            if dyy_list:
                derivatives['dyy'] = np.column_stack(dyy_list)
            if dtt_list:
                derivatives['dtt'] = np.column_stack(dtt_list)
        
        return derivatives
    
    def _build_physics_matrix(
        self,
        H: np.ndarray,
        derivatives: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Build matrix for physics residual computation.
        
        Parameters
        ----------
        H : np.ndarray
            Hidden layer activation matrix
        derivatives : dict
            Dictionary of derivative matrices
            
        Returns
        -------
        np.ndarray
            Physics constraint matrix
        """
        if self.pde is None:
            return H
        
        # The physics matrix depends on the specific PDE
        # This is a placeholder that should be customized per PDE
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
        Construct augmented linear system with all constraints.
        
        Parameters
        ----------
        X_data : np.ndarray
            Training input data points
        y_data : np.ndarray
            Training target values
        X_collocation : np.ndarray
            Collocation points for physics constraints
        X_bc : np.ndarray, optional
            Boundary condition points
        y_bc : np.ndarray, optional
            Boundary condition values
        X_ic : np.ndarray, optional
            Initial condition points
        y_ic : np.ndarray, optional
            Initial condition values
            
        Returns
        -------
        tuple
            (A, b) matrices for linear system Ax = b
        """
        # Compute hidden matrices
        H_data = self._compute_hidden_matrix(X_data)
        H_coll = self._compute_hidden_matrix(X_collocation)
        
        # Get derivatives for collocation points
        derivatives = self._compute_derivatives_matrix(X_collocation)
        
        # Initialize system matrices
        n_output = y_data.shape[1] if len(y_data.shape) > 1 else 1
        A = self.lambda_data * H_data.T @ H_data
        b = self.lambda_data * H_data.T @ y_data
        
        # Add physics constraints
        if self.pde is not None:
            H_physics = self._build_physics_matrix(H_coll, derivatives)
            f_physics = np.zeros((H_physics.shape[0], n_output))
            
            # Add source term if available
            if hasattr(self.pde, 'source_term'):
                f_physics = self.pde.source_term(X_collocation)
                if len(f_physics.shape) == 1:
                    f_physics = f_physics.reshape(-1, 1)
            
            A += self.lambda_physics * H_physics.T @ H_physics
            b += self.lambda_physics * H_physics.T @ f_physics
        
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
        collocation_strategy: str = 'latin_hypercube'
    ):
        """
        Fit the PIELM model.
        
        Parameters
        ----------
        X_data : np.ndarray
            Training input data
        y_data : np.ndarray
            Training target data
        X_collocation : np.ndarray, optional
            Collocation points for physics constraints
        X_bc : np.ndarray, optional
            Boundary condition points
        y_bc : np.ndarray, optional
            Boundary condition values
        X_ic : np.ndarray, optional
            Initial condition points
        y_ic : np.ndarray, optional
            Initial condition values
        n_collocation : int
            Number of collocation points to generate
        collocation_strategy : str
            Strategy for generating collocation points
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
        
        # Construct and solve augmented system
        A, b = self._construct_augmented_system(
            X_data_scaled, y_data_scaled,
            X_collocation_scaled,
            X_bc_scaled, y_bc_scaled,
            X_ic_scaled, y_ic_scaled
        )
        
        # Solve linear system
        self.beta = self.solver.solve(A, b, method='cholesky')
        
        # Validate solution
        self._validate_solution(X_data_scaled, y_data_scaled)
        
        return self
    
    def _infer_bounds(self, X: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """Infer domain bounds from data."""
        bounds = {}
        for i in range(X.shape[1]):
            bounds[f'x{i}'] = (X[:, i].min(), X[:, i].max())
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
        Compute physics residual at given points.
        
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
        
        X_scaled = self.input_scaler.transform(X)
        H = self._compute_hidden_matrix(X_scaled)
        u_scaled = H @ self.beta
        u = self.output_scaler.inverse_transform(u_scaled)
        
        # Compute derivatives
        derivatives = self._compute_derivatives_matrix(X_scaled)
        
        # Transform derivatives to original scale
        derivatives_original = {}
        for key, deriv in derivatives.items():
            if key.startswith('d'):
                # Scale derivatives appropriately
                derivatives_original[key] = deriv * self.output_scaler.scale_
        
        return self.pde.residual(u, X, derivatives_original)
    
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