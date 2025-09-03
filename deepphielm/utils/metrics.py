"""Performance metrics for PIELM models."""

import numpy as np
from typing import Dict, Optional, List, Tuple
from scipy import integrate


class Metrics:
    """
    Collection of metrics for evaluating PIELM performance.
    """
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error."""
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return np.sqrt(Metrics.mse(y_true, y_pred))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R² coefficient of determination."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    @staticmethod
    def relative_l2_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Relative L2 error (normalized)."""
        return np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)
    
    @staticmethod
    def relative_l_inf_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Relative L∞ error (maximum error)."""
        return np.max(np.abs(y_true - y_pred)) / np.max(np.abs(y_true))
    
    @staticmethod
    def physics_informed_loss(
        model,
        X_data: np.ndarray,
        y_data: np.ndarray,
        X_collocation: np.ndarray,
        X_bc: Optional[np.ndarray] = None,
        y_bc: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute all components of physics-informed loss.
        
        Parameters
        ----------
        model : PIELM
            Trained model
        X_data : np.ndarray
            Data points
        y_data : np.ndarray
            Data values
        X_collocation : np.ndarray
            Collocation points
        X_bc : np.ndarray, optional
            Boundary points
        y_bc : np.ndarray, optional
            Boundary values
            
        Returns
        -------
        dict
            Dictionary of loss components
        """
        losses = {}
        
        # Data loss
        y_pred = model.predict(X_data)
        losses['data'] = Metrics.mse(y_data, y_pred)
        
        # Physics residual
        if model.pde is not None:
            residual = model.compute_physics_residual(X_collocation)
            losses['physics'] = np.mean(residual ** 2)
        
        # Boundary conditions
        if X_bc is not None and y_bc is not None:
            y_bc_pred = model.predict(X_bc)
            losses['boundary'] = Metrics.mse(y_bc, y_bc_pred)
        
        # Total weighted loss
        total = model.lambda_data * losses.get('data', 0)
        total += model.lambda_physics * losses.get('physics', 0)
        total += model.lambda_bc * losses.get('boundary', 0)
        losses['total'] = total
        
        return losses
    
    @staticmethod
    def convergence_rate(
        errors: List[float],
        grid_sizes: List[int]
    ) -> Tuple[float, float]:
        """
        Estimate convergence rate from error vs grid size.
        
        Parameters
        ----------
        errors : list
            Error values
        grid_sizes : list
            Corresponding grid sizes
            
        Returns
        -------
        tuple
            (rate, constant) where error ≈ constant * h^rate
        """
        log_h = np.log(1 / np.array(grid_sizes))
        log_error = np.log(errors)
        
        # Linear regression in log-log space
        coeffs = np.polyfit(log_h, log_error, 1)
        rate = coeffs[0]
        constant = np.exp(coeffs[1])
        
        return rate, constant
    
    @staticmethod
    def conservation_error(
        model,
        X: np.ndarray,
        conservation_law: callable
    ) -> float:
        """
        Check conservation law violation.
        
        Parameters
        ----------
        model : PIELM
            Trained model
        X : np.ndarray
            Evaluation points
        conservation_law : callable
            Function that computes conserved quantity
            
        Returns
        -------
        float
            Conservation error
        """
        u = model.predict(X)
        conserved = conservation_law(u, X)
        
        # Should be constant for true conservation
        return np.std(conserved) / (np.mean(np.abs(conserved)) + 1e-10)
    
    @staticmethod
    def stability_metric(
        model,
        X: np.ndarray,
        perturbation_scale: float = 1e-6
    ) -> float:
        """
        Measure model stability to input perturbations.
        
        Parameters
        ----------
        model : PIELM
            Trained model
        X : np.ndarray
            Evaluation points
        perturbation_scale : float
            Scale of perturbations
            
        Returns
        -------
        float
            Stability metric (lower is more stable)
        """
        y_original = model.predict(X)
        
        # Add small perturbations
        X_perturbed = X + np.random.randn(*X.shape) * perturbation_scale
        y_perturbed = model.predict(X_perturbed)
        
        # Compute sensitivity
        sensitivity = np.linalg.norm(y_perturbed - y_original) / (
            np.linalg.norm(X_perturbed - X) + 1e-10
        )
        
        return sensitivity
    
    @staticmethod
    def energy_spectrum(
        u: np.ndarray,
        L: float = 2 * np.pi
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute energy spectrum for turbulence problems.
        
        Parameters
        ----------
        u : np.ndarray
            Velocity field (2D or 3D)
        L : float
            Domain size
            
        Returns
        -------
        tuple
            (wavenumbers, energy_spectrum)
        """
        # Fourier transform
        u_hat = np.fft.fftn(u)
        
        # Power spectrum
        power = np.abs(u_hat) ** 2
        
        # Radial averaging
        n = u.shape[0]
        kx = np.fft.fftfreq(n, d=L/n) * 2 * np.pi
        
        if len(u.shape) == 2:
            ky = kx
            kx, ky = np.meshgrid(kx, ky)
            k = np.sqrt(kx**2 + ky**2)
        else:
            ky = kx
            kz = kx
            kx, ky, kz = np.meshgrid(kx, ky, kz)
            k = np.sqrt(kx**2 + ky**2 + kz**2)
        
        # Bin by wavenumber magnitude
        k_bins = np.arange(0, k.max(), 1)
        E_k = np.zeros(len(k_bins) - 1)
        
        for i in range(len(k_bins) - 1):
            mask = (k >= k_bins[i]) & (k < k_bins[i+1])
            E_k[i] = np.sum(power[mask])
        
        k_centers = (k_bins[:-1] + k_bins[1:]) / 2
        
        return k_centers, E_k