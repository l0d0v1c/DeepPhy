"""Cross-validation and model validation for PIELM."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import KFold, TimeSeriesSplit
import warnings


class CrossValidator:
    """
    Cross-validation strategies for PIELM models.
    
    Includes specialized validation for physics-informed problems.
    """
    
    def __init__(self):
        self.validation_history = []
        
    def physics_aware_cv(
        self,
        model_class,
        model_params: Dict[str, Any],
        X_data: np.ndarray,
        y_data: np.ndarray,
        X_collocation: np.ndarray,
        n_folds: int = 5,
        validation_metrics: List[str] = ['mse', 'physics_residual']
    ) -> Dict[str, float]:
        """
        Physics-aware cross-validation.
        
        Validates both data fitting and physics compliance.
        
        Parameters
        ----------
        model_class : class
            PIELM model class
        model_params : dict
            Model parameters
        X_data : np.ndarray
            Training data points
        y_data : np.ndarray
            Training target values
        X_collocation : np.ndarray
            Collocation points for physics
        n_folds : int
            Number of CV folds
        validation_metrics : list
            Metrics to compute
            
        Returns
        -------
        dict
            Cross-validation scores
        """
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_data)):
            print(f"Fold {fold + 1}/{n_folds}")
            
            # Split data
            X_train_fold = X_data[train_idx]
            y_train_fold = y_data[train_idx]
            X_val_fold = X_data[val_idx]
            y_val_fold = y_data[val_idx]
            
            # Train model
            model = model_class(**model_params)
            
            try:
                model.fit(X_train_fold, y_train_fold, X_collocation)
                
                # Compute metrics
                fold_metrics = {}
                
                if 'mse' in validation_metrics:
                    y_pred = model.predict(X_val_fold)
                    fold_metrics['mse'] = np.mean((y_val_fold - y_pred) ** 2)
                
                if 'mae' in validation_metrics:
                    y_pred = model.predict(X_val_fold)
                    fold_metrics['mae'] = np.mean(np.abs(y_val_fold - y_pred))
                
                if 'physics_residual' in validation_metrics:
                    if hasattr(model, 'compute_physics_residual'):
                        residual = model.compute_physics_residual(X_collocation)
                        fold_metrics['physics_residual'] = np.mean(np.abs(residual))
                    else:
                        fold_metrics['physics_residual'] = 0.0
                
                if 'r2' in validation_metrics:
                    y_pred = model.predict(X_val_fold)
                    ss_res = np.sum((y_val_fold - y_pred) ** 2)
                    ss_tot = np.sum((y_val_fold - np.mean(y_val_fold)) ** 2)
                    fold_metrics['r2'] = 1 - (ss_res / ss_tot)
                
                fold_results.append(fold_metrics)
                
            except Exception as e:
                warnings.warn(f"Fold {fold} failed: {e}")
                # Add worst-case scores
                fold_metrics = {metric: 1e6 if metric != 'r2' else -1e6 
                               for metric in validation_metrics}
                fold_results.append(fold_metrics)
        
        # Aggregate results
        cv_results = {}
        for metric in validation_metrics:
            scores = [result[metric] for result in fold_results if metric in result]
            cv_results[f'{metric}_mean'] = np.mean(scores)
            cv_results[f'{metric}_std'] = np.std(scores)
        
        self.validation_history.append({
            'model_params': model_params,
            'cv_results': cv_results,
            'fold_results': fold_results
        })
        
        return cv_results
    
    def temporal_validation(
        self,
        model_class,
        model_params: Dict[str, Any],
        X_data: np.ndarray,
        y_data: np.ndarray,
        X_collocation: np.ndarray,
        n_splits: int = 5,
        gap: int = 0
    ) -> Dict[str, float]:
        """
        Temporal cross-validation for time-dependent problems.
        
        Parameters
        ----------
        model_class : class
            Model class
        model_params : dict
            Model parameters
        X_data : np.ndarray
            Training data (assumed to have time as last dimension)
        y_data : np.ndarray
            Training targets
        X_collocation : np.ndarray
            Collocation points
        n_splits : int
            Number of temporal splits
        gap : int
            Gap between train and test sets
            
        Returns
        -------
        dict
            Temporal validation scores
        """
        # Sort by time (assuming last column is time)
        if X_data.shape[1] > 1:
            time_idx = np.argsort(X_data[:, -1])
            X_data_sorted = X_data[time_idx]
            y_data_sorted = y_data[time_idx]
        else:
            X_data_sorted = X_data
            y_data_sorted = y_data
        
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_data_sorted)):
            print(f"Temporal fold {fold + 1}/{n_splits}")
            
            X_train_fold = X_data_sorted[train_idx]
            y_train_fold = y_data_sorted[train_idx]
            X_val_fold = X_data_sorted[val_idx]
            y_val_fold = y_data_sorted[val_idx]
            
            # Train model
            model = model_class(**model_params)
            
            try:
                model.fit(X_train_fold, y_train_fold, X_collocation)
                
                # Evaluate
                y_pred = model.predict(X_val_fold)
                mse = np.mean((y_val_fold - y_pred) ** 2)
                
                # Physics residual
                physics_residual = 0.0
                if hasattr(model, 'compute_physics_residual'):
                    residual = model.compute_physics_residual(X_collocation)
                    physics_residual = np.mean(np.abs(residual))
                
                fold_results.append({
                    'mse': mse,
                    'physics_residual': physics_residual
                })
                
            except Exception as e:
                warnings.warn(f"Temporal fold {fold} failed: {e}")
                fold_results.append({'mse': 1e6, 'physics_residual': 1e6})
        
        # Aggregate
        mse_scores = [r['mse'] for r in fold_results]
        physics_scores = [r['physics_residual'] for r in fold_results]
        
        return {
            'mse_mean': np.mean(mse_scores),
            'mse_std': np.std(mse_scores),
            'physics_residual_mean': np.mean(physics_scores),
            'physics_residual_std': np.std(physics_scores)
        }
    
    def spatial_validation(
        self,
        model_class,
        model_params: Dict[str, Any],
        X_data: np.ndarray,
        y_data: np.ndarray,
        X_collocation: np.ndarray,
        spatial_dims: List[int] = [0, 1],
        n_regions: int = 4
    ) -> Dict[str, float]:
        """
        Spatial cross-validation by domain partitioning.
        
        Parameters
        ----------
        model_class : class
            Model class
        model_params : dict
            Model parameters
        X_data : np.ndarray
            Training data
        y_data : np.ndarray
            Training targets
        X_collocation : np.ndarray
            Collocation points
        spatial_dims : list
            Dimensions to use for spatial partitioning
        n_regions : int
            Number of spatial regions
            
        Returns
        -------
        dict
            Spatial validation scores
        """
        # Partition domain spatially
        fold_results = []
        
        for region in range(n_regions):
            print(f"Spatial region {region + 1}/{n_regions}")
            
            # Define region bounds
            region_bounds = self._get_region_bounds(
                X_data[:, spatial_dims], region, n_regions
            )
            
            # Split data based on spatial location
            train_mask = np.ones(len(X_data), dtype=bool)
            val_mask = np.zeros(len(X_data), dtype=bool)
            
            for i, dim in enumerate(spatial_dims):
                dim_vals = X_data[:, dim]
                in_region = (
                    (dim_vals >= region_bounds[i][0]) & 
                    (dim_vals <= region_bounds[i][1])
                )
                val_mask |= in_region
                train_mask &= ~in_region
            
            if np.sum(val_mask) == 0 or np.sum(train_mask) == 0:
                warnings.warn(f"Empty region {region}, skipping")
                continue
            
            X_train = X_data[train_mask]
            y_train = y_data[train_mask]
            X_val = X_data[val_mask]
            y_val = y_data[val_mask]
            
            # Train model
            model = model_class(**model_params)
            
            try:
                model.fit(X_train, y_train, X_collocation)
                
                # Evaluate
                y_pred = model.predict(X_val)
                mse = np.mean((y_val - y_pred) ** 2)
                
                fold_results.append({'mse': mse})
                
            except Exception as e:
                warnings.warn(f"Spatial region {region} failed: {e}")
                fold_results.append({'mse': 1e6})
        
        # Aggregate
        mse_scores = [r['mse'] for r in fold_results]
        
        return {
            'mse_mean': np.mean(mse_scores),
            'mse_std': np.std(mse_scores)
        }
    
    def _get_region_bounds(
        self,
        X_spatial: np.ndarray,
        region: int,
        n_regions: int
    ) -> List[Tuple[float, float]]:
        """Get bounds for spatial region."""
        bounds = []
        
        # Simple grid-based partitioning
        grid_size = int(np.ceil(n_regions ** (1.0 / X_spatial.shape[1])))
        
        for dim in range(X_spatial.shape[1]):
            dim_min = X_spatial[:, dim].min()
            dim_max = X_spatial[:, dim].max()
            dim_range = dim_max - dim_min
            
            # Determine region index in this dimension
            region_dim = region % grid_size
            
            bound_min = dim_min + region_dim * dim_range / grid_size
            bound_max = dim_min + (region_dim + 1) * dim_range / grid_size
            
            bounds.append((bound_min, bound_max))
        
        return bounds
    
    def bootstrap_validation(
        self,
        model_class,
        model_params: Dict[str, Any],
        X_data: np.ndarray,
        y_data: np.ndarray,
        X_collocation: np.ndarray,
        n_bootstrap: int = 100,
        bootstrap_ratio: float = 0.8
    ) -> Dict[str, float]:
        """
        Bootstrap validation for uncertainty estimation.
        
        Parameters
        ----------
        model_class : class
            Model class
        model_params : dict
            Model parameters
        X_data : np.ndarray
            Training data
        y_data : np.ndarray
            Training targets
        X_collocation : np.ndarray
            Collocation points
        n_bootstrap : int
            Number of bootstrap samples
        bootstrap_ratio : float
            Fraction of data to use in each bootstrap
            
        Returns
        -------
        dict
            Bootstrap validation statistics
        """
        n_samples = len(X_data)
        n_bootstrap_samples = int(bootstrap_ratio * n_samples)
        
        bootstrap_scores = []
        
        for i in range(n_bootstrap):
            if (i + 1) % 10 == 0:
                print(f"Bootstrap sample {i + 1}/{n_bootstrap}")
            
            # Random sampling with replacement
            bootstrap_idx = np.random.choice(
                n_samples, n_bootstrap_samples, replace=True
            )
            out_of_bag_idx = np.setdiff1d(np.arange(n_samples), bootstrap_idx)
            
            if len(out_of_bag_idx) == 0:
                continue
            
            X_bootstrap = X_data[bootstrap_idx]
            y_bootstrap = y_data[bootstrap_idx]
            X_oob = X_data[out_of_bag_idx]
            y_oob = y_data[out_of_bag_idx]
            
            # Train on bootstrap sample
            model = model_class(**model_params)
            
            try:
                model.fit(X_bootstrap, y_bootstrap, X_collocation)
                
                # Evaluate on out-of-bag
                y_pred_oob = model.predict(X_oob)
                mse_oob = np.mean((y_oob - y_pred_oob) ** 2)
                
                bootstrap_scores.append(mse_oob)
                
            except Exception as e:
                warnings.warn(f"Bootstrap sample {i} failed: {e}")
                continue
        
        if not bootstrap_scores:
            return {'mse_mean': np.nan, 'mse_std': np.nan, 'confidence_interval': (np.nan, np.nan)}
        
        bootstrap_scores = np.array(bootstrap_scores)
        
        return {
            'mse_mean': np.mean(bootstrap_scores),
            'mse_std': np.std(bootstrap_scores),
            'confidence_interval': (
                np.percentile(bootstrap_scores, 2.5),
                np.percentile(bootstrap_scores, 97.5)
            )
        }