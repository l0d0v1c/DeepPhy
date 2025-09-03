"""Iterative solvers for large-scale systems."""

import numpy as np
from typing import Optional, Callable, Tuple
import warnings


class IterativeSolver:
    """
    Custom iterative solvers optimized for PIELM systems.
    """
    
    def __init__(self):
        self.iteration_count = 0
        self.residual_history = []
        
    def conjugate_gradient(
        self,
        A: np.ndarray,
        b: np.ndarray,
        x0: Optional[np.ndarray] = None,
        tol: float = 1e-6,
        maxiter: Optional[int] = None,
        preconditioner: Optional[Callable] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Preconditioned Conjugate Gradient method.
        
        Parameters
        ----------
        A : np.ndarray or callable
            Symmetric positive definite matrix or function computing Ax
        b : np.ndarray
            Right-hand side
        x0 : np.ndarray, optional
            Initial guess
        tol : float
            Convergence tolerance
        maxiter : int, optional
            Maximum iterations
        preconditioner : callable, optional
            Preconditioner M^{-1}
            
        Returns
        -------
        tuple
            (solution, iterations)
        """
        n = len(b)
        if x0 is None:
            x = np.zeros_like(b)
        else:
            x = x0.copy()
        
        if maxiter is None:
            maxiter = n
        
        # Compute initial residual
        if callable(A):
            r = b - A(x)
        else:
            r = b - A @ x
        
        # Apply preconditioner
        if preconditioner is not None:
            z = preconditioner(r)
        else:
            z = r.copy()
        
        p = z.copy()
        rz_old = np.dot(r, z)
        
        self.residual_history = [np.linalg.norm(r)]
        
        for k in range(maxiter):
            # Matrix-vector product
            if callable(A):
                Ap = A(p)
            else:
                Ap = A @ p
            
            # Step size
            alpha = rz_old / np.dot(p, Ap)
            
            # Update solution and residual
            x += alpha * p
            r -= alpha * Ap
            
            # Check convergence
            residual_norm = np.linalg.norm(r)
            self.residual_history.append(residual_norm)
            
            if residual_norm < tol:
                self.iteration_count = k + 1
                return x, k + 1
            
            # Apply preconditioner
            if preconditioner is not None:
                z = preconditioner(r)
            else:
                z = r.copy()
            
            # Update search direction
            rz_new = np.dot(r, z)
            beta = rz_new / rz_old
            p = z + beta * p
            
            rz_old = rz_new
        
        warnings.warn(f"CG did not converge in {maxiter} iterations")
        self.iteration_count = maxiter
        return x, maxiter
    
    def bicgstab(
        self,
        A: np.ndarray,
        b: np.ndarray,
        x0: Optional[np.ndarray] = None,
        tol: float = 1e-6,
        maxiter: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Biconjugate Gradient Stabilized method for non-symmetric systems.
        
        Parameters
        ----------
        A : np.ndarray or callable
            Matrix or function computing Ax
        b : np.ndarray
            Right-hand side
        x0 : np.ndarray, optional
            Initial guess
        tol : float
            Convergence tolerance
        maxiter : int, optional
            Maximum iterations
            
        Returns
        -------
        tuple
            (solution, iterations)
        """
        n = len(b)
        if x0 is None:
            x = np.zeros_like(b)
        else:
            x = x0.copy()
        
        if maxiter is None:
            maxiter = n
        
        # Initial residual
        if callable(A):
            r = b - A(x)
        else:
            r = b - A @ x
        
        r_hat = r.copy()  # Arbitrary choice
        
        rho_old = alpha = omega = 1.0
        v = np.zeros_like(b)
        p = np.zeros_like(b)
        
        self.residual_history = [np.linalg.norm(r)]
        
        for k in range(maxiter):
            rho = np.dot(r_hat, r)
            
            if np.abs(rho) < 1e-15:
                warnings.warn("BiCGSTAB breakdown: rho = 0")
                break
            
            beta = (rho / rho_old) * (alpha / omega)
            p = r + beta * (p - omega * v)
            
            # Matrix-vector product
            if callable(A):
                v = A(p)
            else:
                v = A @ p
            
            alpha = rho / np.dot(r_hat, v)
            s = r - alpha * v
            
            # Check convergence
            if np.linalg.norm(s) < tol:
                x += alpha * p
                self.iteration_count = k + 1
                return x, k + 1
            
            # Matrix-vector product
            if callable(A):
                t = A(s)
            else:
                t = A @ s
            
            omega = np.dot(t, s) / np.dot(t, t)
            
            # Update solution
            x += alpha * p + omega * s
            r = s - omega * t
            
            # Check convergence
            residual_norm = np.linalg.norm(r)
            self.residual_history.append(residual_norm)
            
            if residual_norm < tol:
                self.iteration_count = k + 1
                return x, k + 1
            
            rho_old = rho
        
        warnings.warn(f"BiCGSTAB did not converge in {maxiter} iterations")
        self.iteration_count = maxiter
        return x, maxiter
    
    def gmres_custom(
        self,
        A: np.ndarray,
        b: np.ndarray,
        x0: Optional[np.ndarray] = None,
        tol: float = 1e-6,
        restart: int = 30,
        maxiter: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Restarted GMRES implementation.
        
        Parameters
        ----------
        A : np.ndarray or callable
            Matrix or function computing Ax
        b : np.ndarray
            Right-hand side
        x0 : np.ndarray, optional
            Initial guess
        tol : float
            Convergence tolerance
        restart : int
            Restart parameter (Krylov subspace size)
        maxiter : int, optional
            Maximum iterations
            
        Returns
        -------
        tuple
            (solution, iterations)
        """
        n = len(b)
        if x0 is None:
            x = np.zeros_like(b)
        else:
            x = x0.copy()
        
        if maxiter is None:
            maxiter = n
        
        self.residual_history = []
        total_iters = 0
        
        for outer in range(maxiter // restart + 1):
            # Compute initial residual
            if callable(A):
                r = b - A(x)
            else:
                r = b - A @ x
            
            r_norm = np.linalg.norm(r)
            self.residual_history.append(r_norm)
            
            if r_norm < tol:
                self.iteration_count = total_iters
                return x, total_iters
            
            # Initialize
            V = np.zeros((n, restart + 1))
            H = np.zeros((restart + 1, restart))
            V[:, 0] = r / r_norm
            
            # Arnoldi process
            for j in range(restart):
                # Matrix-vector product
                if callable(A):
                    w = A(V[:, j])
                else:
                    w = A @ V[:, j]
                
                # Gram-Schmidt orthogonalization
                for i in range(j + 1):
                    H[i, j] = np.dot(w, V[:, i])
                    w -= H[i, j] * V[:, i]
                
                H[j + 1, j] = np.linalg.norm(w)
                
                if H[j + 1, j] < 1e-10:
                    # Lucky breakdown
                    m = j + 1
                    break
                
                V[:, j + 1] = w / H[j + 1, j]
            else:
                m = restart
            
            # Solve least squares problem
            e1 = np.zeros(m + 1)
            e1[0] = r_norm
            
            # QR decomposition of H
            Q, R = np.linalg.qr(H[:m+1, :m])
            y = np.linalg.solve(R, Q.T @ e1)
            
            # Update solution
            x += V[:, :m] @ y
            total_iters += m
            
            if total_iters >= maxiter:
                break
        
        warnings.warn(f"GMRES did not converge in {maxiter} iterations")
        self.iteration_count = maxiter
        return x, maxiter