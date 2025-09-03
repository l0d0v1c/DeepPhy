"""Sampling strategies for collocation and data points."""

import numpy as np
from typing import Dict, Tuple, Literal, Optional, List
from scipy.stats import qmc


class SamplingStrategy:
    """
    Collection of sampling strategies for generating collocation points.
    """
    
    def generate_points(
        self,
        n_points: int,
        bounds: Dict[str, Tuple[float, float]],
        strategy: Literal['uniform', 'latin_hypercube', 'sobol', 'halton', 'grid'] = 'latin_hypercube'
    ) -> np.ndarray:
        """
        Generate sampling points using specified strategy.
        
        Parameters
        ----------
        n_points : int
            Number of points to generate
        bounds : dict
            Dictionary of dimension names to (min, max) bounds
        strategy : str
            Sampling strategy
            
        Returns
        -------
        np.ndarray
            Generated points of shape (n_points, n_dims)
        """
        if strategy == 'uniform':
            return self._uniform_random(n_points, bounds)
        elif strategy == 'latin_hypercube':
            return self._latin_hypercube(n_points, bounds)
        elif strategy == 'sobol':
            return self._sobol(n_points, bounds)
        elif strategy == 'halton':
            return self._halton(n_points, bounds)
        elif strategy == 'grid':
            return self._grid(n_points, bounds)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _uniform_random(
        self,
        n_points: int,
        bounds: Dict[str, Tuple[float, float]]
    ) -> np.ndarray:
        """Uniform random sampling."""
        dims = len(bounds)
        points = np.random.rand(n_points, dims)
        
        # Scale to bounds using vectorized operations
        bounds_array = np.array(list(bounds.values()))
        low_bounds = bounds_array[:, 0]
        high_bounds = bounds_array[:, 1]
        points = low_bounds + points * (high_bounds - low_bounds)
        
        return points
    
    def _latin_hypercube(
        self,
        n_points: int,
        bounds: Dict[str, Tuple[float, float]]
    ) -> np.ndarray:
        """Latin Hypercube Sampling for better space coverage."""
        dims = len(bounds)
        sampler = qmc.LatinHypercube(d=dims, seed=42)
        points = sampler.random(n_points)
        
        # Scale to bounds using vectorized operations
        bounds_array = np.array(list(bounds.values()))
        low_bounds = bounds_array[:, 0]
        high_bounds = bounds_array[:, 1]
        points = qmc.scale(points, low_bounds, high_bounds)
        
        return points
    
    def _sobol(
        self,
        n_points: int,
        bounds: Dict[str, Tuple[float, float]]
    ) -> np.ndarray:
        """Sobol sequence sampling (quasi-random)."""
        dims = len(bounds)
        sampler = qmc.Sobol(d=dims, seed=42)
        points = sampler.random(n_points)
        
        # Scale to bounds using vectorized operations
        bounds_array = np.array(list(bounds.values()))
        low_bounds = bounds_array[:, 0]
        high_bounds = bounds_array[:, 1]
        points = qmc.scale(points, low_bounds, high_bounds)
        
        return points
    
    def _halton(
        self,
        n_points: int,
        bounds: Dict[str, Tuple[float, float]]
    ) -> np.ndarray:
        """Halton sequence sampling (quasi-random)."""
        dims = len(bounds)
        sampler = qmc.Halton(d=dims, seed=42)
        points = sampler.random(n_points)
        
        # Scale to bounds using vectorized operations
        bounds_array = np.array(list(bounds.values()))
        low_bounds = bounds_array[:, 0]
        high_bounds = bounds_array[:, 1]
        points = qmc.scale(points, low_bounds, high_bounds)
        
        return points
    
    def _grid(
        self,
        n_points: int,
        bounds: Dict[str, Tuple[float, float]]
    ) -> np.ndarray:
        """Regular grid sampling."""
        dims = len(bounds)
        points_per_dim = int(np.ceil(n_points ** (1/dims)))
        
        # Create grid
        axes = []
        for key, (low, high) in bounds.items():
            axes.append(np.linspace(low, high, points_per_dim))
        
        mesh = np.meshgrid(*axes)
        points = np.column_stack([m.flatten() for m in mesh])
        
        # Randomly select n_points if we have too many
        if len(points) > n_points:
            idx = np.random.choice(len(points), n_points, replace=False)
            points = points[idx]
        
        return points
    
    def boundary_points(
        self,
        n_points: int,
        bounds: Dict[str, Tuple[float, float]],
        include_corners: bool = True
    ) -> np.ndarray:
        """
        Generate points on domain boundaries.
        
        Parameters
        ----------
        n_points : int
            Number of boundary points
        bounds : dict
            Domain bounds
        include_corners : bool
            Whether to include corner points
            
        Returns
        -------
        np.ndarray
            Boundary points
        """
        dims = len(bounds)
        boundary_points = []
        
        if dims == 1:
            # 1D: just endpoints
            key, (low, high) = list(bounds.items())[0]
            return np.array([[low], [high]])
        
        elif dims == 2:
            # 2D: points along rectangle edges
            keys = list(bounds.keys())
            x_range = bounds[keys[0]]
            y_range = bounds[keys[1]]
            
            points_per_edge = n_points // 4
            
            # Bottom edge
            x_bottom = np.linspace(x_range[0], x_range[1], points_per_edge)
            y_bottom = np.full(points_per_edge, y_range[0])
            
            # Top edge
            x_top = np.linspace(x_range[0], x_range[1], points_per_edge)
            y_top = np.full(points_per_edge, y_range[1])
            
            # Left edge
            x_left = np.full(points_per_edge, x_range[0])
            y_left = np.linspace(y_range[0], y_range[1], points_per_edge)
            
            # Right edge
            x_right = np.full(points_per_edge, x_range[1])
            y_right = np.linspace(y_range[0], y_range[1], points_per_edge)
            
            boundary_points = np.vstack([
                np.column_stack([x_bottom, y_bottom]),
                np.column_stack([x_top, y_top]),
                np.column_stack([x_left, y_left]),
                np.column_stack([x_right, y_right])
            ])
        
        else:
            # Higher dimensions: sample randomly on faces
            n_per_face = n_points // (2 * dims)
            
            for dim in range(dims):
                # Low boundary
                low_points = self._uniform_random(n_per_face, bounds)
                low_points[:, dim] = list(bounds.values())[dim][0]
                boundary_points.append(low_points)
                
                # High boundary
                high_points = self._uniform_random(n_per_face, bounds)
                high_points[:, dim] = list(bounds.values())[dim][1]
                boundary_points.append(high_points)
            
            boundary_points = np.vstack(boundary_points)
        
        # Remove duplicates
        boundary_points = np.unique(boundary_points, axis=0)
        
        # Randomly select if we have too many
        if len(boundary_points) > n_points:
            idx = np.random.choice(len(boundary_points), n_points, replace=False)
            boundary_points = boundary_points[idx]
        
        return boundary_points


class AdaptiveSampler:
    """
    Adaptive sampling based on solution properties and residuals.
    """
    
    def __init__(self, model=None, bounds: Optional[Dict] = None):
        self.model = model
        self.bounds = bounds
        self.refinement_history = []
        
    def sample(
        self,
        n_points: int,
        refinement_ratio: float = 0.3,
        residual_threshold_percentile: float = 90
    ) -> np.ndarray:
        """
        Generate points with adaptive refinement.
        
        Parameters
        ----------
        n_points : int
            Total number of points
        refinement_ratio : float
            Fraction of points in high-residual regions
        residual_threshold_percentile : float
            Percentile for identifying high-residual regions
            
        Returns
        -------
        np.ndarray
            Adaptively sampled points
        """
        if self.model is None or not hasattr(self.model, 'beta') or self.model.beta is None:
            # Model not trained yet, return uniform sampling
            sampler = SamplingStrategy()
            return sampler.generate_points(n_points, self.bounds, 'latin_hypercube')
        
        n_uniform = int(n_points * (1 - refinement_ratio))
        n_refined = n_points - n_uniform
        
        # Generate uniform base points
        sampler = SamplingStrategy()
        uniform_points = sampler.generate_points(n_uniform, self.bounds, 'latin_hypercube')
        
        # Evaluate residuals on fine grid
        grid_points = self._create_evaluation_grid(resolution=50)
        residuals = np.abs(self.model.compute_physics_residual(grid_points))
        
        # Identify high-residual regions
        threshold = np.percentile(residuals, residual_threshold_percentile)
        high_residual_mask = residuals.flatten() > threshold
        high_residual_points = grid_points[high_residual_mask]
        
        if len(high_residual_points) > 0:
            # Sample from high-residual regions
            if len(high_residual_points) > n_refined:
                idx = np.random.choice(len(high_residual_points), n_refined, replace=False)
                refined_points = high_residual_points[idx]
            else:
                refined_points = high_residual_points
                # Add more points if needed
                if len(refined_points) < n_refined:
                    extra = sampler.generate_points(
                        n_refined - len(refined_points),
                        self.bounds,
                        'uniform'
                    )
                    refined_points = np.vstack([refined_points, extra])
        else:
            # No high-residual regions, use uniform
            refined_points = sampler.generate_points(n_refined, self.bounds, 'uniform')
        
        # Combine points
        all_points = np.vstack([uniform_points, refined_points])
        
        # Store refinement info
        self.refinement_history.append({
            'n_uniform': n_uniform,
            'n_refined': n_refined,
            'threshold': threshold,
            'max_residual': np.max(residuals)
        })
        
        return all_points
    
    def _create_evaluation_grid(self, resolution: int = 50) -> np.ndarray:
        """Create regular grid for residual evaluation."""
        if self.bounds is None:
            raise ValueError("Bounds must be specified for grid creation")
        
        axes = []
        for key, (low, high) in self.bounds.items():
            axes.append(np.linspace(low, high, resolution))
        
        mesh = np.meshgrid(*axes)
        return np.column_stack([m.flatten() for m in mesh])
    
    def density_based_sampling(
        self,
        n_points: int,
        density_function: callable
    ) -> np.ndarray:
        """
        Sample points according to a density function.
        
        Parameters
        ----------
        n_points : int
            Number of points to sample
        density_function : callable
            Function that returns sampling density at given points
            
        Returns
        -------
        np.ndarray
            Sampled points
        """
        # Use rejection sampling
        sampler = SamplingStrategy()
        points = []
        max_density = 1.0  # Will be updated
        
        while len(points) < n_points:
            # Generate candidate points
            candidates = sampler.generate_points(n_points * 10, self.bounds, 'uniform')
            densities = density_function(candidates)
            
            # Update max density
            max_density = max(max_density, np.max(densities))
            
            # Accept/reject
            accept_prob = densities / max_density
            accept_mask = np.random.rand(len(candidates)) < accept_prob
            accepted = candidates[accept_mask]
            
            points.append(accepted)
            
            if len(np.vstack(points)) >= n_points:
                break
        
        points = np.vstack(points)[:n_points]
        return points