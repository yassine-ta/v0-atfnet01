#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quantum-Inspired Optimization module for ATFNet.
Implements quantum-inspired algorithms for portfolio optimization and hyperparameter tuning.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from scipy.optimize import minimize
import random
from datetime import datetime
import joblib

logger = logging.getLogger('atfnet.models.quantum_optimizer')


class QuantumInspiredOptimizer:
    """
    Quantum-inspired optimization for ATFNet.
    Implements quantum-inspired algorithms for portfolio optimization and hyperparameter tuning.
    """
    
    def __init__(self, config: Dict[str, Any] = None, output_dir: str = 'quantum_optimization'):
        """
        Initialize the quantum-inspired optimizer.
        
        Args:
            config: Configuration dictionary
            output_dir: Directory to save outputs
        """
        self.config = config or {}
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load default parameters
        self.population_size = self.config.get('population_size', 50)
        self.max_iterations = self.config.get('max_iterations', 100)
        self.convergence_threshold = self.config.get('convergence_threshold', 1e-6)
        self.interference_rate = self.config.get('interference_rate', 0.3)
        self.entanglement_rate = self.config.get('entanglement_rate', 0.2)
        self.mutation_rate = self.config.get('mutation_rate', 0.1)
        
        # Initialize state variables
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.optimization_history = []
        
        logger.info("Quantum-Inspired Optimizer initialized")
    
    def optimize_portfolio(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray,
                          risk_aversion: float = 1.0, constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Optimize portfolio weights using quantum-inspired algorithm.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            risk_aversion: Risk aversion parameter
            constraints: Dictionary of constraints
            
        Returns:
            Dictionary containing optimization results
        """
        n_assets = len(expected_returns)
        
        # Set up constraints
        if constraints is None:
            constraints = {}
        
        # Default constraints
        sum_to_one = constraints.get('sum_to_one', True)
        long_only = constraints.get('long_only', True)
        max_weight = constraints.get('max_weight', 1.0)
        min_weight = constraints.get('min_weight', 0.0)
        
        # Define fitness function (negative utility to maximize)
        def fitness_function(weights):
            # Apply sum to one constraint
            if sum_to_one:
                weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
            
            # Calculate portfolio return and risk
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
            
            # Calculate utility (Markowitz mean-variance utility)
            utility = portfolio_return - 0.5 * risk_aversion * portfolio_risk**2
            
            # Apply penalty for constraint violations
            penalty = 0.0
            
            if long_only and np.any(weights < 0):
                penalty += 100 * np.sum(np.abs(np.minimum(weights, 0)))
            
            if np.any(weights > max_weight):
                penalty += 100 * np.sum(np.maximum(weights - max_weight, 0))
            
            if np.any(weights < min_weight):
                penalty += 100 * np.sum(np.maximum(min_weight - weights, 0))
            
            return utility - penalty
        
        # Run quantum-inspired optimization
        result = self._quantum_inspired_optimization(
            fitness_function=fitness_function,
            n_dimensions=n_assets,
            bounds=[(min_weight, max_weight) for _ in range(n_assets)]
        )
        
        # Apply sum to one constraint to final solution
        if sum_to_one:
            result['solution'] = result['solution'] / np.sum(result['solution'])
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(result['solution'], expected_returns)
        portfolio_risk = np.sqrt(np.dot(result['solution'].T, np.dot(covariance_matrix, result['solution'])))
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        
        # Add metrics to result
        result['portfolio_return'] = portfolio_return
        result['portfolio_risk'] = portfolio_risk
        result['sharpe_ratio'] = sharpe_ratio
        
        logger.info(f"Portfolio optimization completed: Return={portfolio_return:.4f}, "
                   f"Risk={portfolio_risk:.4f}, Sharpe={sharpe_ratio:.4f}")
        
        return result
    
    def optimize_hyperparameters(self, objective_function: Callable, parameter_ranges: Dict[str, Tuple],
                               maximize: bool = True, n_iterations: int = None) -> Dict[str, Any]:
        """
        Optimize hyperparameters using quantum-inspired algorithm.
        
        Args:
            objective_function: Function to optimize (takes parameters dict, returns score)
            parameter_ranges: Dictionary of parameter ranges (name: (min, max))
            maximize: Whether to maximize (True) or minimize (False) the objective
            n_iterations: Number of iterations (overrides default)
            
        Returns:
            Dictionary containing optimization results
        """
        # Extract parameter names and ranges
        param_names = list(parameter_ranges.keys())
        param_mins = [parameter_ranges[name][0] for name in param_names]
        param_maxs = [parameter_ranges[name][1] for name in param_names]
        
        # Define fitness function wrapper
        def fitness_function(x):
            # Convert to parameters dictionary
            params = {name: x[i] for i, name in enumerate(param_names)}
            
            # Call objective function
            score = objective_function(params)
            
            # Return score (negated if minimizing)
            return score if maximize else -score
        
        # Run quantum-inspired optimization
        result = self._quantum_inspired_optimization(
            fitness_function=fitness_function,
            n_dimensions=len(param_names),
            bounds=[(param_mins[i], param_maxs[i]) for i in range(len(param_names))],
            max_iterations=n_iterations or self.max_iterations
        )
        
        # Convert solution to parameters dictionary
        best_params = {name: result['solution'][i] for i, name in enumerate(param_names)}
        
        # Calculate best score
        best_score = objective_function(best_params)
        
        # Add to result
        result['best_params'] = best_params
        result['best_score'] = best_score
        
        logger.info(f"Hyperparameter optimization completed: Best score={best_score:.4f}")
        
        return result
    
    def _quantum_inspired_optimization(self, fitness_function: Callable, n_dimensions: int,
                                     bounds: List[Tuple[float, float]],
                                     max_iterations: int = None) -> Dict[str, Any]:
        """
        Run quantum-inspired optimization algorithm.
        
        Args:
            fitness_function: Function to optimize
            n_dimensions: Number of dimensions
            bounds: List of (min, max) bounds for each dimension
            max_iterations: Maximum number of iterations
            
        Returns:
            Dictionary containing optimization results
        """
        if max_iterations is None:
            max_iterations = self.max_iterations
        
        # Initialize population (quantum particles)
        population = self._initialize_quantum_population(n_dimensions, bounds)
        
        # Evaluate initial population
        fitness_values = np.array([fitness_function(p) for p in population])
        
        # Track best solution
        best_idx = np.argmax(fitness_values)
        best_solution = population[best_idx].copy()
        best_fitness = fitness_values[best_idx]
        
        # Initialize history
        history = []
        
        # Main optimization loop
        for iteration in range(max_iterations):
            # Apply quantum interference
            population = self._apply_quantum_interference(population, fitness_values)
            
            # Apply quantum entanglement
            population = self._apply_quantum_entanglement(population, best_solution)
            
            # Apply quantum mutation
            population = self._apply_quantum_mutation(population, bounds)
            
            # Evaluate population
            fitness_values = np.array([fitness_function(p) for p in population])
            
            # Update best solution
            current_best_idx = np.argmax(fitness_values)
            current_best_fitness = fitness_values[current_best_idx]
            
            if current_best_fitness > best_fitness:
                best_solution = population[current_best_idx].copy()
                best_fitness = current_best_fitness
            
            # Record history
            history.append({
                'iteration': iteration,
                'best_fitness': best_fitness,
                'mean_fitness': np.mean(fitness_values),
                'std_fitness': np.std(fitness_values)
            })
            
            # Check convergence
            if iteration > 10:
                recent_improvement = history[-1]['best_fitness'] - history[-10]['best_fitness']
                if abs(recent_improvement) < self.convergence_threshold:
                    logger.info(f"Converged after {iteration} iterations")
                    break
            
            # Log progress
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: Best fitness = {best_fitness:.6f}")
        
        # Update state
        self.best_solution = best_solution
        self.best_fitness = best_fitness
        self.optimization_history = history
        
        # Prepare result
        result = {
            'solution': best_solution,
            'fitness': best_fitness,
            'iterations': len(history),
            'history': history,
            'converged': len(history) < max_iterations
        }
        
        return result
    
    def _initialize_quantum_population(self, n_dimensions: int, 
                                      bounds: List[Tuple[float, float]]) -> np.ndarray:
        """
        Initialize quantum population.
        
        Args:
            n_dimensions: Number of dimensions
            bounds: List of (min, max) bounds for each dimension
            
        Returns:
            Initialized population
        """
        population = np.zeros((self.population_size, n_dimensions))
        
        for i in range(self.population_size):
            for j in range(n_dimensions):
                min_val, max_val = bounds[j]
                # Initialize with quantum superposition (uniform distribution)
                population[i, j] = min_val + (max_val - min_val) * random.random()
        
        return population
    
    def _apply_quantum_interference(self, population: np.ndarray, 
                                   fitness_values: np.ndarray) -> np.ndarray:
        """
        Apply quantum interference operation.
        
        Args:
            population: Current population
            fitness_values: Fitness values for population
            
        Returns:
            Updated population
        """
        # Normalize fitness values to probabilities
        fitness_min = np.min(fitness_values)
        fitness_range = np.max(fitness_values) - fitness_min
        
        if fitness_range > 0:
            probabilities = (fitness_values - fitness_min) / fitness_range
        else:
            probabilities = np.ones_like(fitness_values) / len(fitness_values)
        
        # Create new population
        new_population = population.copy()
        
        # Apply interference
        for i in range(self.population_size):
            if random.random() < self.interference_rate:
                # Select two particles for interference
                idx1, idx2 = np.random.choice(self.population_size, size=2, p=probabilities/np.sum(probabilities))
                
                # Quantum interference (weighted superposition)
                alpha = random.random()
                new_population[i] = alpha * population[idx1] + (1 - alpha) * population[idx2]
        
        return new_population
    
    def _apply_quantum_entanglement(self, population: np.ndarray, 
                                   best_solution: np.ndarray) -> np.ndarray:
        """
        Apply quantum entanglement operation.
        
        Args:
            population: Current population
            best_solution: Best solution found so far
            
        Returns:
            Updated population
        """
        new_population = population.copy()
        
        # Apply entanglement
        for i in range(self.population_size):
            if random.random() < self.entanglement_rate:
                # Entangle with best solution
                entanglement_strength = random.random()
                new_population[i] = (1 - entanglement_strength) * population[i] + entanglement_strength * best_solution
        
        return new_population
    
    def _apply_quantum_mutation(self, population: np.ndarray, 
                               bounds: List[Tuple[float, float]]) -> np.ndarray:
        """
        Apply quantum mutation operation.
        
        Args:
            population: Current population
            bounds: List of (min, max) bounds for each dimension
            
        Returns:
            Updated population
        """
        new_population = population.copy()
        n_dimensions = population.shape[1]
        
        # Apply mutation
        for i in range(self.population_size):
            for j in range(n_dimensions):
                if random.random() < self.mutation_rate:
                    # Quantum tunneling (random jump)
                    min_val, max_val = bounds[j]
                    new_population[i, j] = min_val + (max_val - min_val) * random.random()
        
        return new_population
    
    def plot_optimization_history(self, save_plot: bool = True,
                                 filename: str = 'optimization_history.png') -> None:
        """
        Plot optimization history.
        
        Args:
            save_plot: Whether to save the plot
            filename: Filename for saved plot
        """
        if not self.optimization_history:
            logger.warning("No optimization history to plot")
            return
        
        # Extract data
        iterations = [h['iteration'] for h in self.optimization_history]
        best_fitness = [h['best_fitness'] for h in self.optimization_history]
        mean_fitness = [h['mean_fitness'] for h in self.optimization_history]
        std_fitness = [h['std_fitness'] for h in self.optimization_history]
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot fitness values
        plt.plot(iterations, best_fitness, 'b-', label='Best Fitness')
        plt.plot(iterations, mean_fitness, 'g-', label='Mean Fitness')
        
        # Plot standard deviation band
        plt.fill_between(iterations, 
                        [m - s for m, s in zip(mean_fitness, std_fitness)],
                        [m + s for m, s in zip(mean_fitness, std_fitness)],
                        color='g', alpha=0.2)
        
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.title('Quantum-Inspired Optimization History')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        if save_plot:
            plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight', dpi=300)
            logger.info(f"Optimization history plot saved to {self.output_dir}/{filename}")
        
        plt.close()
    
    def plot_efficient_frontier(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray,
                               n_portfolios: int = 50, save_plot: bool = True,
                               filename: str = 'efficient_frontier.png') -> None:
        """
        Plot efficient frontier using quantum-inspired optimization.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            n_portfolios: Number of portfolios to generate
            save_plot: Whether to save the plot
            filename: Filename for saved plot
        """
        # Generate range of risk aversion parameters
        risk_aversions = np.logspace(-1, 1, n_portfolios)
        
        # Optimize portfolios
        portfolios = []
        
        for risk_aversion in risk_aversions:
            result = self.optimize_portfolio(
                expected_returns=expected_returns,
                covariance_matrix=covariance_matrix,
                risk_aversion=risk_aversion
            )
            
            portfolios.append({
                'weights': result['solution'],
                'return': result['portfolio_return'],
                'risk': result['portfolio_risk'],
                'sharpe': result['sharpe_ratio']
            })
        
        # Extract data
        returns = [p['return'] for p in portfolios]
        risks = [p['risk'] for p in portfolios]
        sharpes = [p['sharpe'] for p in portfolios]
        
        # Find maximum Sharpe ratio portfolio
        max_sharpe_idx = np.argmax(sharpes)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot efficient frontier
        sc = plt.scatter(risks, returns, c=sharpes, cmap='viridis', s=50, alpha=0.8)
        
        # Highlight maximum Sharpe ratio portfolio
        plt.scatter([risks[max_sharpe_idx]], [returns[max_sharpe_idx]], 
                   c='red', s=100, marker='*', label='Max Sharpe')
        
        plt.colorbar(sc, label='Sharpe Ratio')
        plt.xlabel('Portfolio Risk (Standard Deviation)')
        plt.ylabel('Portfolio Return')
        plt.title('Efficient Frontier (Quantum-Inspired Optimization)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        if save_plot:
            plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight', dpi=300)
            logger.info(f"Efficient frontier plot saved to {self.output_dir}/{filename}")
        
        plt.close()
