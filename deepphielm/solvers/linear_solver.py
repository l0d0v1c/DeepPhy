"""Optimized linear solvers for PIELM."""

import numpy as np
import warnings
from typing import Optional, Literal
from scipy import linalg
from scipy.sparse import issparse, csr_matrix
from scipy.sparse.linalg import spsolve, cg, gmres, bicgstab


class LinearSolver:
    """
    Collection of optimized linear solvers for augmented systems.
    
    Provides various methods for solving Ax = b with different
    trade-offs between speed and numerical stability.
    """
    
    def __init__(self):
        self.condition_number = None
        self.last_method_used = None
        
    def solve(
        self,
        A: np.ndarray,
        b: np.ndarray,
        method: Literal['auto', 'cholesky', 'svd', 'qr', 'lu', 'cg', 'gmres'] = 'auto',
        tol: float = 1e-10,
        maxiter: Optional[int] = None
    ) -> np.ndarray:
        """
        Solve linear system Ax = b.
        
        Parameters
        ----------
        A : np.ndarray or sparse matrix
            Coefficient matrix
        b : np.ndarray
            Right-hand side
        method : str
            Solving method:
            - 'auto': Automatically choose best method
            - 'cholesky': Cholesky decomposition (for SPD matrices)
            - 'svd': Singular value decomposition (most stable)
            - 'qr': QR decomposition
            - 'lu': LU decomposition
            - 'cg': Conjugate gradient (iterative, for SPD)
            - 'gmres': GMRES (iterative, general)
        tol : float
            Tolerance for iterative solvers
        maxiter : int, optional
            Maximum iterations for iterative solvers
            
        Returns
        -------
        np.ndarray
            Solution vector x
        """
        self.last_method_used = method
        
        if method == 'auto':
            method = self._choose_method(A)
            self.last_method_used = method
        
        if method == 'cholesky':
            return self._solve_cholesky(A, b)
        elif method == 'svd':
            return self._solve_svd(A, b, tol)
        elif method == 'qr':
            return self._solve_qr(A, b)
        elif method == 'lu':
            return self._solve_lu(A, b)
        elif method == 'cg':
            return self._solve_cg(A, b, tol, maxiter)
        elif method == 'gmres':
            return self._solve_gmres(A, b, tol, maxiter)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _choose_method(self, A: np.ndarray) -> str:
        """Automatically choose best solving method based on matrix properties."""
        n = A.shape[0]
        
        # For small matrices, use direct methods
        if n < 1000:
            # Check if symmetric positive definite
            if self._is_spd(A):
                return 'cholesky'
            else:
                return 'svd'
        
        # For large matrices, use iterative methods
        if issparse(A) or n > 5000:
            if self._is_symmetric(A):
                return 'cg'
            else:
                return 'gmres'
        
        # Medium-sized dense matrices
        return 'qr'
    
    def _is_spd(self, A: np.ndarray, tol: float = 1e-10) -> bool:
        """Check if matrix is symmetric positive definite."""
        if not self._is_symmetric(A, tol):
            return False
        
        try:
            # Try Cholesky decomposition
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    
    def _is_symmetric(self, A: np.ndarray, tol: float = 1e-10) -> bool:
        """Check if matrix is symmetric."""
        if issparse(A):
            return (A - A.T).nnz == 0
        return np.allclose(A, A.T, rtol=tol)
    
    def _solve_cholesky(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve using Cholesky decomposition.
        Most efficient for symmetric positive definite matrices.
        """
        try:
            L = linalg.cholesky(A, lower=True)
            y = linalg.solve_triangular(L, b, lower=True)
            x = linalg.solve_triangular(L.T, y, lower=False)
            return x
        except linalg.LinAlgError:
            warnings.warn("Cholesky failed, falling back to SVD")
            return self._solve_svd(A, b)
    
    def _solve_svd(self, A: np.ndarray, b: np.ndarray, tol: float = 1e-10) -> np.ndarray:
        """
        Solve using singular value decomposition.
        Most stable but slower method.
        """
        U, s, Vt = linalg.svd(A, full_matrices=False)
        
        # Compute condition number
        self.condition_number = s[0] / s[-1] if s[-1] > 0 else np.inf
        
        # Threshold small singular values
        s_inv = np.where(s > tol * s[0], 1/s, 0)
        
        # Solve
        x = Vt.T @ (s_inv * (U.T @ b))
        
        return x
    
    def _solve_qr(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solve using QR decomposition."""
        Q, R = linalg.qr(A, mode='economic')
        y = Q.T @ b
        x = linalg.solve_triangular(R, y)
        return x
    
    def _solve_lu(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solve using LU decomposition."""
        lu, piv = linalg.lu_factor(A)
        x = linalg.lu_solve((lu, piv), b)
        return x
    
    def _solve_cg(
        self,
        A: np.ndarray,
        b: np.ndarray,
        tol: float = 1e-6,
        maxiter: Optional[int] = None
    ) -> np.ndarray:
        """
        Solve using conjugate gradient (iterative).
        Best for large sparse SPD matrices.
        """
        if not issparse(A):
            A = csr_matrix(A)
        
        if maxiter is None:
            maxiter = A.shape[0]
        
        x, info = cg(A, b, tol=tol, maxiter=maxiter)
        
        if info > 0:
            warnings.warn(f"CG did not converge after {info} iterations")
        elif info < 0:
            raise ValueError(f"CG illegal input or breakdown (info={info})")
        
        return x
    
    def _solve_gmres(
        self,
        A: np.ndarray,
        b: np.ndarray,
        tol: float = 1e-6,
        maxiter: Optional[int] = None
    ) -> np.ndarray:
        """
        Solve using GMRES (iterative).
        General purpose iterative solver.
        """
        if not issparse(A):
            A = csr_matrix(A)
        
        if maxiter is None:
            maxiter = min(A.shape[0], 1000)
        
        x, info = gmres(A, b, tol=tol, maxiter=maxiter)
        
        if info > 0:
            warnings.warn(f"GMRES did not converge after {info} iterations")
        elif info < 0:
            raise ValueError(f"GMRES illegal input or breakdown (info={info})")
        
        return x
    
    def solve_least_squares(
        self,
        A: np.ndarray,
        b: np.ndarray,
        method: Literal['svd', 'qr'] = 'svd'
    ) -> np.ndarray:
        """
        Solve least squares problem min ||Ax - b||².
        
        Parameters
        ----------
        A : np.ndarray
            Coefficient matrix (can be over/underdetermined)
        b : np.ndarray
            Right-hand side
        method : str
            'svd' or 'qr'
            
        Returns
        -------
        np.ndarray
            Least squares solution
        """
        if method == 'svd':
            x, residuals, rank, s = linalg.lstsq(A, b)
            self.condition_number = s[0] / s[-1] if len(s) > 0 and s[-1] > 0 else np.inf
        else:  # QR
            Q, R = linalg.qr(A, mode='economic')
            x = linalg.solve_triangular(R, Q.T @ b)
        
        return x