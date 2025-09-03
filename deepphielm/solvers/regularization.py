"""Regularization techniques for ill-conditioned systems."""

import numpy as np
from typing import Literal, Optional, Tuple
from scipy import linalg


class Regularizer:
    """
    Regularization methods for solving ill-conditioned linear systems.
    """
    
    def __init__(self):
        self.alpha_history = []
        self.error_history = []
        
    def tikhonov(
        self,
        A: np.ndarray,
        b: np.ndarray,
        alpha: float,
        L: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Tikhonov regularization (Ridge regression).
        
        Solves: min ||Ax - b||² + α²||Lx||²
        
        Parameters
        ----------
        A : np.ndarray
            Coefficient matrix
        b : np.ndarray
            Right-hand side
        alpha : float
            Regularization parameter
        L : np.ndarray, optional
            Regularization matrix (identity if None)
            
        Returns
        -------
        np.ndarray
            Regularized solution
        """
        if L is None:
            L = np.eye(A.shape[1])
        
        # Form normal equations with regularization
        ATA = A.T @ A
        ATb = A.T @ b
        LTL = L.T @ L
        
        # Solve (A^T A + α² L^T L)x = A^T b
        x = linalg.solve(ATA + alpha**2 * LTL, ATb)
        
        return x
    
    def truncated_svd(
        self,
        A: np.ndarray,
        b: np.ndarray,
        k: Optional[int] = None,
        tol: Optional[float] = None
    ) -> np.ndarray:
        """
        Truncated SVD regularization.
        
        Parameters
        ----------
        A : np.ndarray
            Coefficient matrix
        b : np.ndarray
            Right-hand side
        k : int, optional
            Number of singular values to keep
        tol : float, optional
            Tolerance for truncation (relative to largest SV)
            
        Returns
        -------
        np.ndarray
            Regularized solution
        """
        U, s, Vt = linalg.svd(A, full_matrices=False)
        
        # Determine truncation
        if k is None:
            if tol is None:
                tol = 1e-10
            k = np.sum(s > tol * s[0])
        
        # Truncate
        U_k = U[:, :k]
        s_k = s[:k]
        Vt_k = Vt[:k, :]
        
        # Solve
        x = Vt_k.T @ ((U_k.T @ b) / s_k)
        
        return x
    
    def l_curve(
        self,
        A: np.ndarray,
        b: np.ndarray,
        alpha_range: Optional[np.ndarray] = None,
        n_alphas: int = 100
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        L-curve method for choosing regularization parameter.
        
        Parameters
        ----------
        A : np.ndarray
            Coefficient matrix
        b : np.ndarray
            Right-hand side
        alpha_range : np.ndarray, optional
            Range of alpha values to test
        n_alphas : int
            Number of alpha values to test
            
        Returns
        -------
        tuple
            (optimal_alpha, residual_norms, solution_norms)
        """
        if alpha_range is None:
            # Estimate range based on singular values
            U, s, Vt = linalg.svd(A, full_matrices=False)
            alpha_range = np.logspace(
                np.log10(s[-1] * 1e-3),
                np.log10(s[0]),
                n_alphas
            )
        
        residual_norms = []
        solution_norms = []
        
        for alpha in alpha_range:
            x = self.tikhonov(A, b, alpha)
            residual_norms.append(np.linalg.norm(A @ x - b))
            solution_norms.append(np.linalg.norm(x))
        
        residual_norms = np.array(residual_norms)
        solution_norms = np.array(solution_norms)
        
        # Find corner of L-curve (maximum curvature)
        optimal_alpha = self._find_corner(
            np.log(residual_norms),
            np.log(solution_norms),
            alpha_range
        )
        
        self.alpha_history = alpha_range
        self.error_history = residual_norms
        
        return optimal_alpha, residual_norms, solution_norms
    
    def _find_corner(
        self,
        x: np.ndarray,
        y: np.ndarray,
        alpha_values: np.ndarray
    ) -> float:
        """
        Find corner of L-curve using maximum curvature.
        
        Parameters
        ----------
        x : np.ndarray
            Log residual norms
        y : np.ndarray
            Log solution norms
        alpha_values : np.ndarray
            Corresponding alpha values
            
        Returns
        -------
        float
            Optimal alpha value
        """
        # Compute curvature
        dx = np.gradient(x)
        dy = np.gradient(y)
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        
        curvature = np.abs(dx * d2y - dy * d2x) / (dx**2 + dy**2)**1.5
        
        # Find maximum curvature
        idx = np.argmax(curvature)
        
        return alpha_values[idx]
    
    def gcv(
        self,
        A: np.ndarray,
        b: np.ndarray,
        alpha_range: Optional[np.ndarray] = None,
        n_alphas: int = 100
    ) -> float:
        """
        Generalized Cross-Validation for choosing regularization parameter.
        
        Parameters
        ----------
        A : np.ndarray
            Coefficient matrix
        b : np.ndarray
            Right-hand side
        alpha_range : np.ndarray, optional
            Range of alpha values
        n_alphas : int
            Number of alpha values to test
            
        Returns
        -------
        float
            Optimal alpha value
        """
        if alpha_range is None:
            U, s, Vt = linalg.svd(A, full_matrices=False)
            alpha_range = np.logspace(
                np.log10(s[-1] * 1e-3),
                np.log10(s[0]),
                n_alphas
            )
        
        gcv_values = []
        
        for alpha in alpha_range:
            # Compute influence matrix
            ATA = A.T @ A
            H = A @ linalg.inv(ATA + alpha**2 * np.eye(A.shape[1])) @ A.T
            
            # GCV function
            x = self.tikhonov(A, b, alpha)
            residual = A @ x - b
            trace_I_minus_H = A.shape[0] - np.trace(H)
            
            gcv = np.linalg.norm(residual)**2 / trace_I_minus_H**2
            gcv_values.append(gcv)
        
        gcv_values = np.array(gcv_values)
        optimal_idx = np.argmin(gcv_values)
        
        return alpha_range[optimal_idx]