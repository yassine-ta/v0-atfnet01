#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hyperparameter optimization for ATFNet models.
Implements Bayesian optimization, grid search, and random search.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
import time
import joblib
from sklearn.model_selection import KFold, TimeSeriesSplit
import matplotlib.pyplot as plt

logger = logging.getLogger('atfnet.models.hyperparameter')


class BaseOptimizer:
    """Base class for hyperparameter optimizers."""
    
    def __init__(
        self,
        model_func: Callable,
        param_space: Dict[str, Any],
        scoring_func: Callable,
        cv: int = 5,
        n_jobs: int = -1,
        verbose: int = 1
    ):
        """
        Initialize the base optimizer.
        
        Args:
            model_func: Function that creates and returns a model
            param_space: Parameter space to search
            scoring_func: Function to score model performance
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
        """
        self.model_func = model_func
        self.param_space = param_space
        self.scoring_func = scoring_func
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Results
        self.results = []
        self.best_params = None
        self.best_score = None
        self.best_model = None
    
    def optimize(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Optimize hyperparameters.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Optimization results
        """
        raise NotImplementedError("Subclasses must implement optimize method")
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: Dict[str, Any],
        time_series: bool = False
    ) -> Tuple[float, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Feature matrix
            y: Target values
            params: Model parameters
            time_series: Whether to use time series split
            
        Returns:
            Tuple of (mean score, std score)
        """
        # Create cross-validation splitter
        if time_series:
            cv_splitter = TimeSeriesSplit(n_splits=self.cv)
        else:
            cv_splitter = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        
        # Perform cross-validation
        scores = []
        
        for train_idx, test_idx in cv_splitter.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Create and train model
            model = self.model_func(**params)
            model.fit(X_train, y_train)
            
            # Score model
            score = self.scoring_func(model, X_test, y_test)
            scores.append(score)
        
        # Calculate mean and std of scores
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        return mean_score, std_score
    
    def plot_results(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot optimization results.
        
        Args:
            figsize: Figure size
        """
        if not self.results:
            raise ValueError("No optimization results available")
        
        # Extract scores
        iteration = [i for i in range(len(self.results))]
        scores = [r.get('score', 0) for r in self.results]
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot scores
        plt.plot(iteration, scores, 'o-')
        
        # Add best score
        best_idx = np.argmax(scores)
        plt.plot(best_idx, scores[best_idx], 'ro', markersize=10)
        plt.text(best_idx, scores[best_idx], f' Best: {scores[best_idx]:.4f}', verticalalignment='bottom')
        
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.title('Hyperparameter Optimization Results')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class GridSearchOptimizer(BaseOptimizer):
    """Grid search hyperparameter optimizer."""
    
    def optimize(self, X: np.ndarray, y: np.ndarray, time_series: bool = False) -> Dict[str, Any]:
        """
        Optimize hyperparameters using grid search.
        
        Args:
            X: Feature matrix
            y: Target values
            time_series: Whether to use time series split
            
        Returns:
            Optimization results
        """
        # Generate parameter grid
        param_grid = self._generate_param_grid()
        
        if self.verbose > 0:
            logger.info(f"Grid search with {len(param_grid)} parameter combinations")
        
        # Perform grid search
        self.results = []
        self.best_score = float('-inf')
        self.best_params = None
        
        for i, params in enumerate(param_grid):
            start_time = time.time()
            
            # Perform cross-validation
            mean_score, std_score = self.cross_validate(X, y, params, time_series)
            
            # Record results
            result = {
                'params': params,
                'score': mean_score,
                'std': std_score,
                'time': time.time() - start_time
            }
            
            self.results.append(result)
            
            # Update best parameters
            if mean_score > self.best_score:
                self.best_score = mean_score
                self.best_params = params
            
            if self.verbose > 0 and (i + 1) % 10 == 0:
                logger.info(f"Completed {i+1}/{len(param_grid)} parameter combinations")
        
        # Train best model
        if self.best_params is not None:
            self.best_model = self.model_func(**self.best_params)
            self.best_model.fit(X, y)
        
        # Prepare results
        optimization_results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_model': self.best_model,
            'all_results': self.results
        }
        
        if self.verbose > 0:
            logger.info(f"Best parameters: {self.best_params}")
            logger.info(f"Best score: {self.best_score:.4f}")
        
        return optimization_results
    
    def _generate_param_grid(self) -> List[Dict[str, Any]]:
        """
        Generate parameter grid from param_space.
        
        Returns:
            List of parameter dictionaries
        """
        import itertools
        
        # Get parameter names and values
        param_names = list(self.param_space.keys())
        param_values = list(self.param_space.values())
        
        # Generate all combinations
        combinations = list(itertools.product(*param_values))
        
        # Convert to list of dictionaries
        param_grid = []
        for combo in combinations:
            param_dict = {name: value for name, value in zip(param_names, combo)}
            param_grid.append(param_dict)
        
        return param_grid


class RandomSearchOptimizer(BaseOptimizer):
    """Random search hyperparameter optimizer."""
    
    def __init__(
        self,
        model_func: Callable,
        param_space: Dict[str, Any],
        scoring_func: Callable,
        cv: int = 5,
        n_jobs: int = -1,
        verbose: int = 1,
        n_iter: int = 10,
        random_state: Optional[int] = None
    ):
        """
        Initialize the random search optimizer.
        
        Args:
            model_func: Function that creates and returns a model
            param_space: Parameter space to search
            scoring_func: Function to score model performance
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
            n_iter: Number of random iterations
            random_state: Random state for reproducibility
        """
        super().__init__(model_func, param_space, scoring_func, cv, n_jobs, verbose)
        self.n_iter = n_iter
        self.random_state = random_state
        
        # Set random state
        if random_state is not None:
            np.random.seed(random_state)
    
    def optimize(self, X: np.ndarray, y: np.ndarray, time_series: bool = False) -> Dict[str, Any]:
        """
        Optimize hyperparameters using random search.
        
        Args:
            X: Feature matrix
            y: Target values
            time_series: Whether to use time series split
            
        Returns:
            Optimization results
        """
        if self.verbose > 0:
            logger.info(f"Random search with {self.n_iter} iterations")
        
        # Perform random search
        self.results = []
        self.best_score = float('-inf')
        self.best_params = None
        
        for i in range(self.n_iter):
            start_time = time.time()
            
            # Sample random parameters
            params = self._sample_params()
            
            # Perform cross-validation
            mean_score, std_score = self.cross_validate(X, y, params, time_series)
            
            # Record results
            result = {
                'params': params,
                'score': mean_score,
                'std': std_score,
                'time': time.time() - start_time
            }
            
            self.results.append(result)
            
            # Update best parameters
            if mean_score > self.best_score:
                self.best_score = mean_score
                self.best_params = params
            
            if self.verbose > 0:
                logger.info(f"Iteration {i+1}/{self.n_iter}: score={mean_score:.4f}, time={result['time']:.2f}s")
        
        # Train best model
        if self.best_params is not None:
            self.best_model = self.model_func(**self.best_params)
            self.best_model.fit(X, y)
        
        # Prepare results
        optimization_results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_model': self.best_model,
            'all_results': self.results
        }
        
        if self.verbose > 0:
            logger.info(f"Best parameters: {self.best_params}")
            logger.info(f"Best score: {self.best_score:.4f}")
        
        return optimization_results
    
    def _sample_params(self) -> Dict[str, Any]:
        """
        Sample random parameters from param_space.
        
        Returns:
            Dictionary of sampled parameters
        """
        params = {}
        
        for param_name, param_values in self.param_space.items():
            if isinstance(param_values, list):
                # Discrete values
                params[param_name] = np.random.choice(param_values)
            elif isinstance(param_values, tuple) and len(param_values) == 2:
                # Continuous range
                low, high = param_values
                if isinstance(low, int) and isinstance(high, int):
                    # Integer range
                    params[param_name] = np.random.randint(low, high + 1)
                else:
                    # Float range
                    params[param_name] = np.random.uniform(low, high)
            else:
                raise ValueError(f"Invalid parameter space for {param_name}")
        
        return params


class BayesianOptimizer(BaseOptimizer):
    """Bayesian hyperparameter optimizer."""
    
    def __init__(
        self,
        model_func: Callable,
        param_space: Dict[str, Any],
        scoring_func: Callable,
        cv: int = 5,
        n_jobs: int = -1,
        verbose: int = 1,
        n_iter: int = 10,
        n_initial_points: int = 5,
        random_state: Optional[int] = None
    ):
        """
        Initialize the Bayesian optimizer.
        
        Args:
            model_func: Function that creates and returns a model
            param_space: Parameter space to search
            scoring_func: Function to score model performance
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
            n_iter: Number of optimization iterations
            n_initial_points: Number of initial random points
            random_state: Random state for reproducibility
        """
        super().__init__(model_func, param_space, scoring_func, cv, n_jobs, verbose)
        self.n_iter = n_iter
        self.n_initial_points = n_initial_points
        self.random_state = random_state
        
        # Check if skopt is available
        try:
            import skopt
        except ImportError:
            raise ImportError("scikit-optimize is required for Bayesian optimization. "
                             "Install it with: pip install scikit-optimize")
    
    def optimize(self, X: np.ndarray, y: np.ndarray, time_series: bool = False) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Bayesian optimization.
        
        Args:
            X: Feature matrix
            y: Target values
            time_series: Whether to use time series split
            
        Returns:
            Optimization results
        """
        from skopt import gp_minimize
        from skopt.space import Real, Integer, Categorical
        from skopt.utils import use_named_args
        
        if self.verbose > 0:
            logger.info(f"Bayesian optimization with {self.n_iter} iterations")
        
        # Convert param_space to skopt space
        dimensions = []
        dimension_names = []
        
        for param_name, param_values in self.param_space.items():
            if isinstance(param_values, list):
                # Categorical values
                dimensions.append(Categorical(param_values, name=param_name))
            elif isinstance(param_values, tuple) and len(param_values) == 2:
                # Continuous range
                low, high = param_values
                if isinstance(low, int) and isinstance(high, int):
                    # Integer range
                    dimensions.append(Integer(low, high, name=param_name))
                else:
                    # Float range
                    dimensions.append(Real(low, high, name=param_name))
            else:
                raise ValueError(f"Invalid parameter space for {param_name}")
            
            dimension_names.append(param_name)
        
        # Define objective function
        @use_named_args(dimensions)
        def objective(**params):
            # Perform cross-validation
            mean_score, _ = self.cross_validate(X, y, params, time_series)
            
            # Record results
            result = {
                'params': params,
                'score': mean_score
            }
            
            self.results.append(result)
            
            # Return negative score (minimize)
            return -mean_score
        
        # Perform Bayesian optimization
        self.results = []
        
        res = gp_minimize(
            objective,
            dimensions,
            n_calls=self.n_iter,
            n_initial_points=self.n_initial_points,
            random_state=self.random_state,
            verbose=self.verbose
        )
        
        # Extract best parameters
        self.best_params = {dim.name: value for dim, value in zip(dimensions, res.x)}
        self.best_score = -res.fun
        
        # Train best model
        self.best_model = self.model_func(**self.best_params)
        self.best_model.fit(X, y)
        
        # Prepare results
        optimization_results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_model': self.best_model,
            'all_results': self.results,
            'skopt_result': res
        }
        
        if self.verbose > 0:
            logger.info(f"Best parameters: {self.best_params}")
            logger.info(f"Best score: {self.best_score:.4f}")
        
        return optimization_results
    
    def plot_convergence(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot optimization convergence.
        
        Args:
            figsize: Figure size
        """
        try:
            from skopt.plots import plot_convergence
            
            plt.figure(figsize=figsize)
            plot_convergence(self.optimization_results['skopt_result'])
            plt.tight_layout()
            plt.show()
        except (ImportError, KeyError, AttributeError) as e:
            logger.error(f"Error plotting convergence: {e}")
            super().plot_results(figsize)


class HyperparameterOptimizer:
    """
    Hyperparameter optimizer that supports multiple optimization methods.
    """
    
    def __init__(
        self,
        model_func: Callable,
        param_space: Dict[str, Any],
        scoring_func: Callable,
        method: str = 'bayesian',
        cv: int = 5,
        n_jobs: int = -1,
        verbose: int = 1,
        n_iter: int = 10,
        random_state: Optional[int] = None
    ):
        """
        Initialize the hyperparameter optimizer.
        
        Args:
            model_func: Function that creates and returns a model
            param_space: Parameter space to search
            scoring_func: Function to score model performance
            method: Optimization method ('grid', 'random', 'bayesian')
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
            n_iter: Number of iterations (for random and Bayesian)
            random_state: Random state for reproducibility
        """
        self.model_func = model_func
        self.param_space = param_space
        self.scoring_func = scoring_func
        self.method = method
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.n_iter = n_iter
        self.random_state = random_state
        
        # Initialize optimizer
        self._init_optimizer()
        
        # Results
        self.optimization_results = None
    
    def _init_optimizer(self) -> None:
        """Initialize optimizer based on method."""
        if self.method == 'grid':
            self.optimizer = GridSearchOptimizer(
                self.model_func,
                self.param_space,
                self.scoring_func,
                self.cv,
                self.n_jobs,
                self.verbose
            )
        elif self.method == 'random':
            self.optimizer = RandomSearchOptimizer(
                self.model_func,
                self.param_space,
                self.scoring_func,
                self.cv,
                self.n_jobs,
                self.verbose,
                self.n_iter,
                self.random_state
            )
        elif self.method == 'bayesian':
            self.optimizer = BayesianOptimizer(
                self.model_func,
                self.param_space,
                self.scoring_func,
                self.cv,
                self.n_jobs,
                self.verbose,
                self.n_iter,
                n_initial_points=5,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown optimization method: {self.method}")
    
    def optimize(self, X: np.ndarray, y: np.ndarray, time_series: bool = False) -> Dict[str, Any]:
        """
        Optimize hyperparameters.
        
        Args:
            X: Feature matrix
            y: Target values
            time_series: Whether to use time series split
            
        Returns:
            Optimization results
        """
        self.optimization_results = self.optimizer.optimize(X, y, time_series)
        return self.optimization_results
    
    def get_best_params(self) -> Dict[str, Any]:
        """
        Get best parameters.
        
        Returns:
            Best parameters
        """
        if self.optimization_results is None:
            raise ValueError("No optimization results available")
        
        return self.optimization_results['best_params']
    
    def get_best_model(self) -> Any:
        """
        Get best model.
        
        Returns:
            Best model
        """
        if self.optimization_results is None:
            raise ValueError("No optimization results available")
        
        return self.optimization_results['best_model']
    
    def plot_results(self, figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot optimization results.
        
        Args:
            figsize: Figure size
        """
        if self.optimization_results is None:
            raise ValueError("No optimization results available")
        
        if self.method == 'bayesian' and hasattr(self.optimizer, 'plot_convergence'):
            self.optimizer.plot_convergence(figsize)
        else:
            self.optimizer.plot_results(figsize)
    
    def save_results(self, path: str) -> None:
        """
        Save optimization results.
        
        Args:
            path: Save path
        """
        if self.optimization_results is None:
            raise ValueError("No optimization results available")
        
        # Remove model from results (may not be serializable)
        results_copy = self.optimization_results.copy()
        if 'best_model' in results_copy:
            del results_copy['best_model']
        
        # Save results
        joblib.dump(results_copy, path)
        logger.info(f"Saved optimization results to {path}")
    
    @staticmethod
    def load_results(path: str) -> Dict[str, Any]:
        """
        Load optimization results.
        
        Args:
            path: Load path
            
        Returns:
            Optimization results
        """
        results = joblib.load(path)
        logger.info(f"Loaded optimization results from {path}")
        return results
