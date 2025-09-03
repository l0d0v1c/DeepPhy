"""Visualization tools for PIELM results."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Dict, Tuple, List
import warnings


class Visualizer:
    """
    Visualization utilities for PIELM models and results.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        self.figsize = figsize
        self.cmap = 'viridis'
        
    def plot_solution_1d(
        self,
        model,
        x_range: Tuple[float, float],
        t: float = 0,
        n_points: int = 200,
        exact_solution: Optional[callable] = None,
        title: Optional[str] = None
    ):
        """
        Plot 1D solution at given time.
        
        Parameters
        ----------
        model : PIELM
            Trained model
        x_range : tuple
            Spatial domain (x_min, x_max)
        t : float
            Time value
        n_points : int
            Number of evaluation points
        exact_solution : callable, optional
            Exact solution for comparison
        title : str, optional
            Plot title
        """
        x = np.linspace(x_range[0], x_range[1], n_points)
        X = np.column_stack([x, np.full(n_points, t)])
        
        y_pred = model.predict(X)
        
        fig, axes = plt.subplots(1, 2 if exact_solution else 1, figsize=self.figsize)
        
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        # Predicted solution
        axes[0].plot(x, y_pred, 'b-', linewidth=2, label='PIELM')
        
        if exact_solution is not None:
            y_exact = exact_solution(X)
            axes[0].plot(x, y_exact, 'r--', linewidth=2, label='Exact')
            axes[0].legend()
            
            # Error plot
            error = np.abs(y_pred - y_exact)
            axes[1].plot(x, error, 'g-', linewidth=2)
            axes[1].set_xlabel('x')
            axes[1].set_ylabel('Absolute Error')
            axes[1].set_title('Prediction Error')
            axes[1].grid(True, alpha=0.3)
        
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('u(x,t)')
        axes[0].set_title(title or f'Solution at t={t:.2f}')
        axes[0].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_solution_2d(
        self,
        model,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        t: float = 0,
        resolution: int = 50,
        plot_type: str = 'contour',
        title: Optional[str] = None
    ):
        """
        Plot 2D solution.
        
        Parameters
        ----------
        model : PIELM
            Trained model
        x_range : tuple
            X domain
        y_range : tuple
            Y domain
        t : float
            Time value (if applicable)
        resolution : int
            Grid resolution
        plot_type : str
            'contour', 'surface', or 'both'
        title : str, optional
            Plot title
        """
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        
        # Prepare input
        if model.n_features == 3:  # Space-time problem
            XY = np.column_stack([
                X.flatten(),
                Y.flatten(),
                np.full(resolution**2, t)
            ])
        else:  # Steady-state problem
            XY = np.column_stack([X.flatten(), Y.flatten()])
        
        Z = model.predict(XY).reshape(resolution, resolution)
        
        if plot_type == 'both':
            fig = plt.figure(figsize=(self.figsize[0]*2, self.figsize[1]))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122, projection='3d')
            
            # Contour plot
            cs = ax1.contourf(X, Y, Z, levels=20, cmap=self.cmap)
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_title(title or 'Solution (Contour)')
            plt.colorbar(cs, ax=ax1)
            
            # Surface plot
            surf = ax2.plot_surface(X, Y, Z, cmap=self.cmap, alpha=0.9)
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            ax2.set_zlabel('u')
            ax2.set_title(title or 'Solution (Surface)')
            plt.colorbar(surf, ax=ax2, shrink=0.5)
            
        elif plot_type == 'contour':
            fig, ax = plt.subplots(figsize=self.figsize)
            cs = ax.contourf(X, Y, Z, levels=20, cmap=self.cmap)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title(title or f'Solution at t={t:.2f}')
            plt.colorbar(cs)
            
        elif plot_type == 'surface':
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X, Y, Z, cmap=self.cmap, alpha=0.9)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('u')
            ax.set_title(title or f'Solution at t={t:.2f}')
            plt.colorbar(surf, shrink=0.5)
        
        plt.tight_layout()
        plt.show()
    
    def plot_physics_residual(
        self,
        model,
        X: np.ndarray,
        log_scale: bool = True,
        title: Optional[str] = None
    ):
        """
        Plot physics residual distribution.
        
        Parameters
        ----------
        model : PIELM
            Trained model
        X : np.ndarray
            Evaluation points
        log_scale : bool
            Use log scale for residuals
        title : str, optional
            Plot title
        """
        residual = np.abs(model.compute_physics_residual(X))
        
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # Histogram
        if log_scale:
            residual_plot = np.log10(residual + 1e-16)
            axes[0].hist(residual_plot, bins=50, edgecolor='black', alpha=0.7)
            axes[0].set_xlabel('log₁₀(|Residual|)')
        else:
            axes[0].hist(residual, bins=50, edgecolor='black', alpha=0.7)
            axes[0].set_xlabel('|Residual|')
        
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Residual Distribution')
        axes[0].grid(True, alpha=0.3)
        
        # Spatial distribution (if 2D)
        if X.shape[1] >= 2:
            scatter = axes[1].scatter(
                X[:, 0], X[:, 1],
                c=np.log10(residual + 1e-16) if log_scale else residual,
                cmap='hot', s=10
            )
            axes[1].set_xlabel('x')
            axes[1].set_ylabel('y' if X.shape[1] > 1 else 't')
            axes[1].set_title('Residual Spatial Distribution')
            plt.colorbar(scatter, ax=axes[1])
        
        plt.suptitle(title or 'Physics Residual Analysis')
        plt.tight_layout()
        plt.show()
    
    def plot_convergence(
        self,
        loss_history: Dict[str, List[float]],
        log_scale: bool = True,
        title: Optional[str] = None
    ):
        """
        Plot convergence history.
        
        Parameters
        ----------
        loss_history : dict
            Dictionary of loss components over iterations
        log_scale : bool
            Use log scale for y-axis
        title : str, optional
            Plot title
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for key, values in loss_history.items():
            if len(values) > 0:
                if log_scale:
                    ax.semilogy(values, label=key, linewidth=2)
                else:
                    ax.plot(values, label=key, linewidth=2)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss' + (' (log scale)' if log_scale else ''))
        ax.set_title(title or 'Training Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_eigenvalues(
        self,
        model,
        n_eigenvalues: int = 50,
        title: Optional[str] = None
    ):
        """
        Plot eigenvalue spectrum of the hidden layer matrix.
        
        Parameters
        ----------
        model : PIELM
            Trained model
        n_eigenvalues : int
            Number of eigenvalues to plot
        title : str, optional
            Plot title
        """
        # Compute hidden layer matrix eigenvalues
        if model.W is None:
            warnings.warn("Model not trained yet")
            return
        
        # Compute correlation matrix of hidden weights
        H_corr = model.W.T @ model.W
        eigenvalues = np.linalg.eigvalsh(H_corr)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # Eigenvalue magnitude
        axes[0].semilogy(
            range(min(n_eigenvalues, len(eigenvalues))),
            eigenvalues[:n_eigenvalues],
            'bo-', linewidth=2
        )
        axes[0].set_xlabel('Index')
        axes[0].set_ylabel('Eigenvalue (log scale)')
        axes[0].set_title('Eigenvalue Spectrum')
        axes[0].grid(True, alpha=0.3)
        
        # Cumulative explained variance
        cumsum = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        axes[1].plot(cumsum[:n_eigenvalues], 'ro-', linewidth=2)
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('Cumulative Variance Explained')
        axes[1].set_title('Variance Explained')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(title or 'Hidden Layer Analysis')
        plt.tight_layout()
        plt.show()
    
    def animate_solution(
        self,
        model,
        x_range: Tuple[float, float],
        t_range: Tuple[float, float],
        n_frames: int = 50,
        resolution: int = 100,
        save_path: Optional[str] = None
    ):
        """
        Create animation of time-dependent solution.
        
        Parameters
        ----------
        model : PIELM
            Trained model
        x_range : tuple
            Spatial domain
        t_range : tuple
            Time range
        n_frames : int
            Number of animation frames
        resolution : int
            Spatial resolution
        save_path : str, optional
            Path to save animation
        """
        from matplotlib.animation import FuncAnimation
        
        x = np.linspace(x_range[0], x_range[1], resolution)
        t_values = np.linspace(t_range[0], t_range[1], n_frames)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        line, = ax.plot([], [], 'b-', linewidth=2)
        ax.set_xlim(x_range)
        ax.set_xlabel('x')
        ax.set_ylabel('u(x,t)')
        ax.grid(True, alpha=0.3)
        
        def init():
            line.set_data([], [])
            return line,
        
        def animate(frame):
            t = t_values[frame]
            X = np.column_stack([x, np.full(resolution, t)])
            y = model.predict(X)
            
            line.set_data(x, y)
            ax.set_title(f'Solution at t={t:.3f}')
            
            # Update y-limits if needed
            ax.set_ylim(y.min() * 1.1, y.max() * 1.1)
            
            return line,
        
        anim = FuncAnimation(
            fig, animate, init_func=init,
            frames=n_frames, interval=50, blit=True
        )
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=10)
        
        plt.show()
        return anim