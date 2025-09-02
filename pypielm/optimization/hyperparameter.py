"""Hyperparameter optimization for PIELM models."""

import numpy as np
from typing import Dict, Tuple, Optional, List, Any
from sklearn.model_selection import KFold
import warnings


class HyperparameterOptimizer:
    """
    Automated hyperparameter optimization for PIELM models.
    """
    
    def __init__(
        self,
        model_class,
        param_bounds: Dict[str, Tuple[float, float]],
        fixed_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize hyperparameter optimizer.
        
        Parameters
        ----------
        model_class : class
            PIELM model class
        param_bounds : dict
            Parameter bounds {param_name: (min, max)}
        fixed_params : dict, optional
            Fixed parameters not to optimize
        """
        self.model_class = model_class
        self.param_bounds = param_bounds
        self.fixed_params = fixed_params or {}
        self.optimization_history = []
        self.best_params = None
        self.best_score = np.inf
        
    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_collocation: np.ndarray,
        method: str = 'bayesian',
        n_trials: int = 50,
        cv_folds: int = 3,
        scoring: str = 'combined'
    ) -> Dict[str, float]:
        """
        Optimize hyperparameters.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training data points
        y_train : np.ndarray
            Training target values
        X_collocation : np.ndarray
            Collocation points for physics constraints
        method : str
            Optimization method ('bayesian', 'grid', 'random')
        n_trials : int
            Number of optimization trials
        cv_folds : int
            Number of cross-validation folds
        scoring : str
            Scoring method ('mse', 'physics', 'combined')
            
        Returns
        -------
        dict
            Optimal hyperparameters
        """
        if method == 'bayesian':
            return self._bayesian_optimization(
                X_train, y_train, X_collocation, n_trials, cv_folds, scoring
            )
        elif method == 'grid':
            return self._grid_search(
                X_train, y_train, X_collocation, cv_folds, scoring
            )
        elif method == 'random':
            return self._random_search(
                X_train, y_train, X_collocation, n_trials, cv_folds, scoring
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _bayesian_optimization(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_collocation: np.ndarray,
        n_trials: int,
        cv_folds: int,
        scoring: str
    ) -> Dict[str, float]:
        """Bayesian optimization using Gaussian Process."""
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer
        except ImportError:
            warnings.warn("scikit-optimize not available, falling back to random search")
            return self._random_search(X_train, y_train, X_collocation, n_trials, cv_folds, scoring)
        
        # Define search space
        space = []
        param_names = []
        
        for name, (low, high) in self.param_bounds.items():
            if 'n_' in name or 'hidden' in name:
                space.append(Integer(int(low), int(high), name=name))
            else:
                space.append(Real(low, high, prior='log-uniform', name=name))
            param_names.append(name)
        
        def objective(params):
            """Objective function to minimize."""
            param_dict = dict(zip(param_names, params))
            param_dict.update(self.fixed_params)
            
            score = self._cross_validate(
                X_train, y_train, X_collocation, param_dict, cv_folds, scoring
            )
            
            self.optimization_history.append({
                'params': param_dict.copy(),
                'score': score
            })
            
            if score < self.best_score:
                self.best_score = score
                self.best_params = param_dict.copy()
            
            return score
        
        # Run optimization
        result = gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=n_trials,
            random_state=42,
            acq_func='EI'  # Expected Improvement
        )
        
        optimal_params = dict(zip(param_names, result.x))
        optimal_params.update(self.fixed_params)
        
        return optimal_params
    
    def _grid_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_collocation: np.ndarray,
        cv_folds: int,
        scoring: str
    ) -> Dict[str, float]:
        """Grid search over parameter space."""
        # Create parameter grid
        param_grids = {}
        for name, (low, high) in self.param_bounds.items():
            if 'n_' in name or 'hidden' in name:
                param_grids[name] = np.logspace(
                    np.log10(low), np.log10(high), 5
                ).astype(int)
            else:
                param_grids[name] = np.logspace(np.log10(low), np.log10(high), 5)
        
        # Generate all combinations
        from itertools import product
        param_names = list(param_grids.keys())
        param_combinations = list(product(*param_grids.values()))
        
        best_score = np.inf
        best_params = None
        
        for param_values in param_combinations:
            param_dict = dict(zip(param_names, param_values))
            param_dict.update(self.fixed_params)
            
            try:
                score = self._cross_validate(
                    X_train, y_train, X_collocation, param_dict, cv_folds, scoring
                )
                
                self.optimization_history.append({
                    'params': param_dict.copy(),
                    'score': score
                })
                
                if score < best_score:
                    best_score = score
                    best_params = param_dict.copy()
                    
            except Exception as e:
                warnings.warn(f"Failed with params {param_dict}: {e}")
                continue
        
        self.best_score = best_score
        self.best_params = best_params
        
        return best_params
    
    def _random_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_collocation: np.ndarray,
        n_trials: int,
        cv_folds: int,
        scoring: str
    ) -> Dict[str, float]:
        """Random search over parameter space."""
        best_score = np.inf
        best_params = None
        
        for trial in range(n_trials):
            # Sample random parameters
            param_dict = {}
            for name, (low, high) in self.param_bounds.items():
                if 'n_' in name or 'hidden' in name:
                    param_dict[name] = int(np.random.uniform(low, high))
                else:
                    param_dict[name] = np.exp(
                        np.random.uniform(np.log(low), np.log(high))
                    )
            
            param_dict.update(self.fixed_params)
            
            try:
                score = self._cross_validate(
                    X_train, y_train, X_collocation, param_dict, cv_folds, scoring
                )
                
                self.optimization_history.append({
                    'params': param_dict.copy(),
                    'score': score
                })
                
                if score < best_score:
                    best_score = score
                    best_params = param_dict.copy()
                    
            except Exception as e:
                warnings.warn(f"Trial {trial} failed: {e}")
                continue
        
        self.best_score = best_score
        self.best_params = best_params
        
        return best_params
    
    def _cross_validate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_collocation: np.ndarray,
        params: Dict[str, Any],
        cv_folds: int,
        scoring: str
    ) -> float:
        """Cross-validate model with given parameters."""
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in kf.split(X_train):
            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train[val_idx]
            y_fold_val = y_train[val_idx]
            
            # Create and train model
            model = self.model_class(**params)
            
            try:
                model.fit(X_fold_train, y_fold_train, X_collocation)
                
                # Compute score
                if scoring == 'mse':
                    y_pred = model.predict(X_fold_val)
                    score = np.mean((y_pred - y_fold_val) ** 2)
                    
                elif scoring == 'physics':
                    if hasattr(model, 'compute_physics_residual'):
                        residual = model.compute_physics_residual(X_collocation)
                        score = np.mean(residual ** 2)
                    else:
                        score = 0.0
                        
                elif scoring == 'combined':
                    y_pred = model.predict(X_fold_val)
                    data_loss = np.mean((y_pred - y_fold_val) ** 2)
                    
                    if hasattr(model, 'compute_physics_residual'):
                        residual = model.compute_physics_residual(X_collocation)
                        physics_loss = np.mean(residual ** 2)
                    else:
                        physics_loss = 0.0
                    
                    score = data_loss + physics_loss
                    
                else:
                    raise ValueError(f"Unknown scoring method: {scoring}")
                
                scores.append(score)
                
            except Exception as e:
                # Penalize failed models
                scores.append(1e6)
                warnings.warn(f"Model training failed: {e}")
        
        return np.mean(scores)
    
    def plot_optimization_history(self):
        """Plot optimization history."""
        if not self.optimization_history:
            print("No optimization history available")
            return
        
        import matplotlib.pyplot as plt
        
        scores = [entry['score'] for entry in self.optimization_history]
        best_scores = np.minimum.accumulate(scores)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Score evolution
        axes[0].plot(scores, 'b-', alpha=0.7, label='Trial scores')
        axes[0].plot(best_scores, 'r-', linewidth=2, label='Best score')
        axes[0].set_xlabel('Trial')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Optimization Progress')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Parameter correlation (example with first parameter)
        if len(self.param_bounds) > 0:
            param_name = list(self.param_bounds.keys())[0]
            param_values = [entry['params'][param_name] for entry in self.optimization_history]
            
            axes[1].scatter(param_values, scores, alpha=0.6)
            axes[1].set_xlabel(param_name)
            axes[1].set_ylabel('Score')
            axes[1].set_title(f'Score vs {param_name}')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def suggest_bounds(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_name: str
    ) -> Tuple[float, float]:
        """
        Suggest parameter bounds based on data characteristics.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training data
        y_train : np.ndarray
            Training targets
        param_name : str
            Parameter name
            
        Returns
        -------
        tuple
            Suggested (min, max) bounds
        """
        n_samples, n_features = X_train.shape
        
        if param_name == 'n_hidden':
            # Rule of thumb: between sqrt(n_features) and 10*n_features
            return (int(np.sqrt(n_features)), min(10 * n_features, 500))
            
        elif param_name in ['lambda_data', 'lambda_physics', 'lambda_bc']:
            # Regularization parameters: wide range
            return (1e-6, 1e2)
            
        elif param_name == 'reg_param':
            # Small regularization
            return (1e-10, 1e-3)
            
        else:
            # Default wide range
            return (1e-3, 1e1)