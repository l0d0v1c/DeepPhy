"""Adaptive training strategies for PIELM."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import warnings


class AdaptiveTrainer:
    """
    Adaptive training strategies for PIELM models.
    
    Implements various adaptive techniques:
    - Adaptive collocation point sampling
    - Progressive training
    - Multi-scale training
    """
    
    def __init__(self, base_model, initial_params: Optional[Dict] = None):
        """
        Initialize adaptive trainer.
        
        Parameters
        ----------
        base_model : class
            Base PIELM model class
        initial_params : dict, optional
            Initial model parameters
        """
        self.base_model = base_model
        self.initial_params = initial_params or {}
        self.training_history = []
        self.current_model = None
        
    def adaptive_refinement_training(
        self,
        X_data: np.ndarray,
        y_data: np.ndarray,
        bounds: Dict[str, Tuple[float, float]],
        max_iterations: int = 10,
        initial_collocation: int = 1000,
        refinement_factor: float = 1.5,
        tolerance: float = 1e-6
    ):
        """
        Train with adaptive refinement of collocation points.
        
        Parameters
        ----------
        X_data : np.ndarray
            Training data points
        y_data : np.ndarray
            Training target values
        bounds : dict
            Domain bounds for collocation point generation
        max_iterations : int
            Maximum refinement iterations
        initial_collocation : int
            Initial number of collocation points
        refinement_factor : float
            Factor to increase collocation points each iteration
        tolerance : float
            Convergence tolerance for physics residual
            
        Returns
        -------
        PIELM
            Trained model
        """
        from ..utils.sampling import AdaptiveSampler
        
        best_model = None
        best_residual = np.inf
        
        n_collocation = initial_collocation
        
        for iteration in range(max_iterations):
            print(f"Refinement iteration {iteration + 1}/{max_iterations}")
            print(f"Using {n_collocation} collocation points")
            
            # Create model
            model = self.base_model(**self.initial_params)
            
            # Generate collocation points
            if iteration == 0:
                # Initial uniform sampling
                from ..utils.sampling import SamplingStrategy
                sampler = SamplingStrategy()
                X_collocation = sampler.generate_points(
                    n_collocation, bounds, 'latin_hypercube'
                )
            else:
                # Adaptive sampling based on previous model
                adaptive_sampler = AdaptiveSampler(best_model, bounds)
                X_collocation = adaptive_sampler.sample(n_collocation)
            
            # Train model
            try:
                model.fit(X_data, y_data, X_collocation)
                
                # Evaluate physics residual
                residual = model.compute_physics_residual(X_collocation)
                avg_residual = np.mean(np.abs(residual))
                max_residual = np.max(np.abs(residual))
                
                print(f"Average residual: {avg_residual:.2e}")
                print(f"Maximum residual: {max_residual:.2e}")
                
                # Store training info
                self.training_history.append({
                    'iteration': iteration,
                    'n_collocation': n_collocation,
                    'avg_residual': avg_residual,
                    'max_residual': max_residual
                })
                
                # Check if this is the best model
                if avg_residual < best_residual:
                    best_residual = avg_residual
                    best_model = model
                
                # Check convergence
                if avg_residual < tolerance:
                    print(f"Converged at iteration {iteration + 1}")
                    break
                
                # Increase collocation points for next iteration
                n_collocation = int(n_collocation * refinement_factor)
                
            except Exception as e:
                warnings.warn(f"Training failed at iteration {iteration}: {e}")
                continue
        
        self.current_model = best_model
        return best_model
    
    def progressive_training(
        self,
        X_data: np.ndarray,
        y_data: np.ndarray,
        X_collocation: np.ndarray,
        lambda_schedule: List[Dict[str, float]],
        n_epochs_per_stage: int = 5
    ):
        """
        Progressive training with changing loss weights.
        
        Parameters
        ----------
        X_data : np.ndarray
            Training data
        y_data : np.ndarray
            Training targets
        X_collocation : np.ndarray
            Collocation points
        lambda_schedule : list
            List of lambda dictionaries for each stage
        n_epochs_per_stage : int
            Training iterations per stage
            
        Returns
        -------
        PIELM
            Final trained model
        """
        model = self.base_model(**self.initial_params)
        
        for stage, lambdas in enumerate(lambda_schedule):
            print(f"Training stage {stage + 1}/{len(lambda_schedule)}")
            print(f"Lambdas: {lambdas}")
            
            # Update model parameters
            for key, value in lambdas.items():
                if hasattr(model, key):
                    setattr(model, key, value)
            
            # Train for this stage
            for epoch in range(n_epochs_per_stage):
                try:
                    model.fit(X_data, y_data, X_collocation)
                    
                    # Evaluate losses
                    losses = model.compute_loss(X_data, y_data, X_collocation)
                    
                    self.training_history.append({
                        'stage': stage,
                        'epoch': epoch,
                        'losses': losses
                    })
                    
                    if epoch == 0 or (epoch + 1) % 2 == 0:
                        print(f"  Epoch {epoch + 1}: Total loss = {losses['total']:.2e}")
                
                except Exception as e:
                    warnings.warn(f"Training failed at stage {stage}, epoch {epoch}: {e}")
                    break
        
        self.current_model = model
        return model
    
    def multi_scale_training(
        self,
        X_data: np.ndarray,
        y_data: np.ndarray,
        bounds: Dict[str, Tuple[float, float]],
        scales: List[int],
        transfer_weights: bool = True
    ):
        """
        Multi-scale training from coarse to fine resolution.
        
        Parameters
        ----------
        X_data : np.ndarray
            Training data
        y_data : np.ndarray
            Training targets
        bounds : dict
            Domain bounds
        scales : list
            List of resolutions (number of points per dimension)
        transfer_weights : bool
            Whether to transfer weights between scales
            
        Returns
        -------
        PIELM
            Final fine-scale model
        """
        from ..utils.sampling import SamplingStrategy
        
        previous_model = None
        sampler = SamplingStrategy()
        
        for scale_idx, scale in enumerate(scales):
            print(f"Training at scale {scale} ({scale_idx + 1}/{len(scales)})")
            
            # Generate collocation points for this scale
            n_points = scale ** len(bounds)
            X_collocation = sampler.generate_points(n_points, bounds, 'grid')
            
            # Create model
            model = self.base_model(**self.initial_params)
            
            # Transfer weights from previous scale if requested
            if transfer_weights and previous_model is not None:
                try:
                    self._transfer_weights(previous_model, model)
                except Exception as e:
                    warnings.warn(f"Weight transfer failed: {e}")
            
            # Train at this scale
            try:
                model.fit(X_data, y_data, X_collocation)
                
                # Evaluate
                losses = model.compute_loss(X_data, y_data, X_collocation)
                
                self.training_history.append({
                    'scale': scale,
                    'n_collocation': n_points,
                    'losses': losses
                })
                
                print(f"  Scale {scale}: Total loss = {losses['total']:.2e}")
                
                previous_model = model
                
            except Exception as e:
                warnings.warn(f"Training failed at scale {scale}: {e}")
                continue
        
        self.current_model = previous_model
        return previous_model
    
    def _transfer_weights(self, source_model, target_model):
        """Transfer weights between models of different scales."""
        if (source_model.W is not None and 
            source_model.beta is not None and
            target_model.n_hidden >= source_model.n_hidden):
            
            # Initialize target weights
            target_model._initialize_weights(source_model.n_features)
            
            # Copy source weights
            n_copy = min(source_model.n_hidden, target_model.n_hidden)
            target_model.W[:, :n_copy] = source_model.W[:, :n_copy]
            target_model.b[:n_copy] = source_model.b[:n_copy]
    
    def curriculum_training(
        self,
        X_data: np.ndarray,
        y_data: np.ndarray,
        bounds: Dict[str, Tuple[float, float]],
        difficulty_function: Callable[[np.ndarray], np.ndarray],
        n_stages: int = 5,
        points_per_stage: int = 500
    ):
        """
        Curriculum training with gradually increasing difficulty.
        
        Parameters
        ----------
        X_data : np.ndarray
            Training data
        y_data : np.ndarray
            Training targets
        bounds : dict
            Domain bounds
        difficulty_function : callable
            Function that assigns difficulty scores to points
        n_stages : int
            Number of curriculum stages
        points_per_stage : int
            Points to add each stage
            
        Returns
        -------
        PIELM
            Trained model
        """
        from ..utils.sampling import SamplingStrategy
        
        # Generate full collocation set
        sampler = SamplingStrategy()
        X_full = sampler.generate_points(
            n_stages * points_per_stage, bounds, 'latin_hypercube'
        )
        
        # Compute difficulty scores
        difficulties = difficulty_function(X_full)
        sorted_indices = np.argsort(difficulties)
        
        model = self.base_model(**self.initial_params)
        current_collocation = np.empty((0, X_full.shape[1]))
        
        for stage in range(n_stages):
            print(f"Curriculum stage {stage + 1}/{n_stages}")
            
            # Add points for this stage (easiest first)
            start_idx = stage * points_per_stage
            end_idx = (stage + 1) * points_per_stage
            stage_indices = sorted_indices[start_idx:end_idx]
            stage_points = X_full[stage_indices]
            
            current_collocation = np.vstack([current_collocation, stage_points])
            
            print(f"  Training with {len(current_collocation)} points")
            
            # Train with current point set
            try:
                model.fit(X_data, y_data, current_collocation)
                
                losses = model.compute_loss(X_data, y_data, current_collocation)
                
                self.training_history.append({
                    'stage': stage,
                    'n_points': len(current_collocation),
                    'difficulty_range': (
                        difficulties[sorted_indices[0]],
                        difficulties[sorted_indices[end_idx - 1]]
                    ),
                    'losses': losses
                })
                
                print(f"  Total loss: {losses['total']:.2e}")
                
            except Exception as e:
                warnings.warn(f"Training failed at stage {stage}: {e}")
                continue
        
        self.current_model = model
        return model
    
    def plot_training_history(self):
        """Plot training history."""
        if not self.training_history:
            print("No training history available")
            return
        
        import matplotlib.pyplot as plt
        
        # Determine what type of training was performed
        if 'iteration' in self.training_history[0]:
            # Adaptive refinement
            iterations = [h['iteration'] for h in self.training_history]
            residuals = [h['avg_residual'] for h in self.training_history]
            
            plt.figure(figsize=(10, 6))
            plt.semilogy(iterations, residuals, 'bo-', linewidth=2)
            plt.xlabel('Refinement Iteration')
            plt.ylabel('Average Physics Residual')
            plt.title('Adaptive Refinement Training')
            plt.grid(True, alpha=0.3)
            
        elif 'stage' in self.training_history[0]:
            # Progressive or curriculum training
            stages = [h['stage'] for h in self.training_history]
            losses = [h['losses']['total'] for h in self.training_history]
            
            plt.figure(figsize=(10, 6))
            plt.semilogy(stages, losses, 'ro-', linewidth=2)
            plt.xlabel('Training Stage')
            plt.ylabel('Total Loss')
            plt.title('Progressive Training')
            plt.grid(True, alpha=0.3)
        
        plt.show()