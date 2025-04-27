#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reinforcement Learning Execution System for ATFNet.
Optimizes order execution using reinforcement learning to minimize market impact.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple, Union
import time
import pickle
from collections import deque
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Concatenate, Flatten
from tensorflow.keras.optimizers import Adam

logger = logging.getLogger('atfnet.models.rl_execution')


class RLExecutionEnvironment:
    """
    Reinforcement learning environment for order execution.
    Simulates market impact and provides rewards based on execution quality.
    """
    
    def __init__(self, price_data: pd.DataFrame, order_size: float, time_limit: int,
                market_impact_factor: float = 0.1, execution_time_penalty: float = 0.01):
        """
        Initialize the RL execution environment.
        
        Args:
            price_data: Historical price data with OHLCV
            order_size: Total order size to execute
            time_limit: Maximum number of steps for execution
            market_impact_factor: Factor for market impact calculation
            execution_time_penalty: Penalty for execution time
        """
        self.price_data = price_data
        self.total_order_size = order_size
        self.time_limit = time_limit
        self.market_impact_factor = market_impact_factor
        self.execution_time_penalty = execution_time_penalty
        
        # State variables
        self.current_step = 0
        self.current_price = 0.0
        self.remaining_size = order_size
        self.executed_sizes = []
        self.execution_prices = []
        self.market_impacts = []
        self.is_buy_order = True  # Default to buy order
        
        # Market state features
        self.price_history = deque(maxlen=10)  # Store recent prices
        self.volume_history = deque(maxlen=10)  # Store recent volumes
        self.volatility = 0.0
        
        # Performance tracking
        self.vwap = 0.0
        self.implementation_shortfall = 0.0
    
    def reset(self, start_idx: int = None, is_buy_order: bool = True) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Args:
            start_idx: Starting index in price data
            is_buy_order: Whether this is a buy order (True) or sell order (False)
            
        Returns:
            Initial state observation
        """
        # Set random start point if not specified
        if start_idx is None:
            max_start = len(self.price_data) - self.time_limit - 1
            start_idx = random.randint(0, max_start) if max_start > 0 else 0
        
        self.current_step = start_idx
        self.is_buy_order = is_buy_order
        self.remaining_size = self.total_order_size
        self.executed_sizes = []
        self.execution_prices = []
        self.market_impacts = []
        
        # Initialize price and market state
        self.current_price = self.price_data.iloc[self.current_step]['close']
        self.price_history.clear()
        self.volume_history.clear()
        
        # Fill history with initial data
        for i in range(max(0, self.current_step - 9), self.current_step + 1):
            self.price_history.append(self.price_data.iloc[i]['close'])
            if 'volume' in self.price_data.columns:
                self.volume_history.append(self.price_data.iloc[i]['volume'])
            else:
                self.volume_history.append(1.0)
        
        # Calculate initial volatility
        self._update_volatility()
        
        # Return initial state
        return self._get_state()
    
    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment by executing a portion of the order.
        
        Args:
            action: Portion of remaining order to execute (0.0 to 1.0)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Ensure action is in valid range
        action = max(0.0, min(1.0, action))
        
        # Calculate execution size
        execution_size = action * self.remaining_size
        
        # Calculate market impact
        market_impact = self._calculate_market_impact(execution_size)
        
        # Calculate execution price with market impact
        if self.is_buy_order:
            execution_price = self.current_price * (1.0 + market_impact)
        else:
            execution_price = self.current_price * (1.0 - market_impact)
        
        # Update state
        self.executed_sizes.append(execution_size)
        self.execution_prices.append(execution_price)
        self.market_impacts.append(market_impact)
        self.remaining_size -= execution_size
        
        # Move to next time step
        self.current_step += 1
        if self.current_step < len(self.price_data):
            self.current_price = self.price_data.iloc[self.current_step]['close']
            
            # Update price and volume history
            self.price_history.append(self.current_price)
            if 'volume' in self.price_data.columns:
                self.volume_history.append(self.price_data.iloc[self.current_step]['volume'])
            else:
                self.volume_history.append(1.0)
            
            # Update volatility
            self._update_volatility()
        
        # Check if done
        done = (self.current_step >= min(len(self.price_data) - 1, self.current_step + self.time_limit) or 
                self.remaining_size <= 1e-6)
        
        # Calculate reward
        reward = self._calculate_reward(execution_size, execution_price, market_impact)
        
        # Calculate performance metrics if done
        info = {}
        if done:
            info = self._calculate_performance()
        
        # Return next state, reward, done flag, and info
        return self._get_state(), reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state observation.
        
        Returns:
            State observation as numpy array
        """
        # Normalize remaining size
        normalized_remaining = self.remaining_size / self.total_order_size
        
        # Normalize time remaining
        normalized_time = (self.time_limit - (self.current_step - self.current_step)) / self.time_limit
        
        # Normalize price history
        if len(self.price_history) > 1:
            base_price = self.price_history[0]
            normalized_prices = [p / base_price - 1.0 for p in self.price_history]
        else:
            normalized_prices = [0.0] * len(self.price_history)
        
        # Normalize volume history
        if len(self.volume_history) > 1:
            mean_volume = np.mean(self.volume_history)
            if mean_volume > 0:
                normalized_volumes = [v / mean_volume for v in self.volume_history]
            else:
                normalized_volumes = [1.0] * len(self.volume_history)
        else:
            normalized_volumes = [1.0] * len(self.volume_history)
        
        # Combine features
        state = [normalized_remaining, normalized_time, self.volatility]
        state.extend(normalized_prices)
        state.extend(normalized_volumes)
        
        return np.array(state)
    
    def _calculate_market_impact(self, execution_size: float) -> float:
        """
        Calculate market impact of an execution.
        
        Args:
            execution_size: Size of execution
            
        Returns:
            Market impact as a price multiplier
        """
        # Simple square-root model for market impact
        if 'volume' in self.price_data.columns:
            current_volume = self.price_data.iloc[self.current_step]['volume']
            relative_size = execution_size / max(current_volume, 1e-6)
        else:
            # If no volume data, use a simple percentage of total order
            relative_size = execution_size / self.total_order_size
        
        # Square root model with volatility adjustment
        impact = self.market_impact_factor * self.volatility * np.sqrt(relative_size)
        
        return impact
    
    def _calculate_reward(self, execution_size: float, execution_price: float, market_impact: float) -> float:
        """
        Calculate reward for the current action.
        
        Args:
            execution_size: Size executed
            execution_price: Price of execution
            market_impact: Market impact of execution
            
        Returns:
            Reward value
        """
        if execution_size <= 0:
            return -0.1  # Small penalty for doing nothing
        
        # Implementation shortfall component (negative impact is good)
        if self.is_buy_order:
            price_quality = -1.0 * (execution_price / self.price_data.iloc[self.current_step - 1]['close'] - 1.0)
        else:
            price_quality = execution_price / self.price_data.iloc[self.current_step - 1]['close'] - 1.0
        
        # Execution progress component
        progress_reward = execution_size / self.total_order_size
        
        # Time penalty
        time_penalty = -1.0 * self.execution_time_penalty * (self.current_step / self.time_limit)
        
        # Combine components
        reward = price_quality + 0.5 * progress_reward + time_penalty
        
        return reward
    
    def _update_volatility(self) -> None:
        """
        Update volatility estimate based on recent price history.
        """
        if len(self.price_history) > 1:
            # Calculate returns
            returns = np.diff(self.price_history) / np.array(list(self.price_history)[:-1])
            # Calculate volatility (standard deviation of returns)
            self.volatility = np.std(returns)
        else:
            self.volatility = 0.01  # Default value
    
    def _calculate_performance(self) -> Dict[str, Any]:
        """
        Calculate execution performance metrics.
        
        Returns:
            Dict containing performance metrics
        """
        # Calculate VWAP of executions
        if sum(self.executed_sizes) > 0:
            self.vwap = sum(p * s for p, s in zip(self.execution_prices, self.executed_sizes)) / sum(self.executed_sizes)
        else:
            self.vwap = 0.0
        
        # Calculate implementation shortfall
        benchmark_price = self.price_data.iloc[self.current_step - len(self.executed_sizes)]['close']
        
        if self.is_buy_order:
            self.implementation_shortfall = (self.vwap / benchmark_price - 1.0) * 100  # In percentage
        else:
            self.implementation_shortfall = (benchmark_price / self.vwap - 1.0) * 100  # In percentage
        
        # Calculate average market impact
        avg_market_impact = np.mean(self.market_impacts) * 100 if self.market_impacts else 0.0
        
        # Calculate execution time
        execution_time = len(self.executed_sizes)
        
        # Calculate percentage of order completed
        pct_completed = (self.total_order_size - self.remaining_size) / self.total_order_size * 100
        
        return {
            'vwap': self.vwap,
            'implementation_shortfall': self.implementation_shortfall,
            'avg_market_impact': avg_market_impact,
            'execution_time': execution_time,
            'pct_completed': pct_completed
        }


class RLExecutionAgent:
    """
    Reinforcement learning agent for order execution optimization.
    Uses Deep Q-Network (DQN) to learn optimal execution strategy.
    """
    
    def __init__(self, state_size: int, action_size: int = 10, 
                memory_size: int = 10000, batch_size: int = 64,
                gamma: float = 0.95, epsilon: float = 1.0,
                epsilon_min: float = 0.01, epsilon_decay: float = 0.995,
                learning_rate: float = 0.001):
        """
        Initialize the RL execution agent.
        
        Args:
            state_size: Size of state space
            action_size: Number of discrete actions
            memory_size: Size of replay memory
            batch_size: Batch size for training
            gamma: Discount factor
            epsilon: Exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            learning_rate: Learning rate for optimizer
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        
        # Initialize replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Build models
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    
    def _build_model(self) -> Model:
        """
        Build neural network model for DQN.
        
        Returns:
            Keras Model
        """
        # Input layer
        input_layer = Input(shape=(self.state_size,))
        
        # Hidden layers
        x = Dense(64, activation='relu')(input_layer)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        
        # Output layer
        output_layer = Dense(self.action_size, activation='linear')(x)
        
        # Create model
        model = Model(inputs=input_layer, outputs=output_layer)
        
        # Compile model
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        
        return model
    
    def update_target_model(self) -> None:
        """
        Update target model weights with current model weights.
        """
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool) -> None:
        """
        Store experience in replay memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray) -> int:
        """
        Choose action based on current state.
        
        Args:
            state: Current state
            
        Returns:
            Selected action index
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size: int = None) -> float:
        """
        Train model using experience replay.
        
        Args:
            batch_size: Size of batch to train on
            
        Returns:
            Loss value
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        if len(self.memory) < batch_size:
            return 0.0
        
        # Sample batch from memory
        minibatch = random.sample(self.memory, batch_size)
        
        # Prepare batch data
        states = np.zeros((batch_size, self.state_size))
        targets = np.zeros((batch_size, self.action_size))
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            
            # Get target Q values
            target = self.model.predict(state.reshape(1, -1), verbose=0)[0]
            
            if done:
                target[action] = reward
            else:
                t = self.target_model.predict(next_state.reshape(1, -1), verbose=0)[0]
                target[action] = reward + self.gamma * np.amax(t)
            
            targets[i] = target
        
        # Train model
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return history.history['loss'][0] if 'loss' in history.history else 0.0
    
    def load(self, filepath: str) -> None:
        """
        Load model weights from file.
        
        Args:
            filepath: Path to model weights file
        """
        self.model.load_weights(filepath)
        self.update_target_model()
    
    def save(self, filepath: str) -> None:
        """
        Save model weights to file.
        
        Args:
            filepath: Path to save model weights
        """
        self.model.save_weights(filepath)


class RLExecutionSystem:
    """
    Reinforcement Learning Execution System for ATFNet.
    Trains and uses RL agents to optimize order execution.
    """
    
    def __init__(self, config: Dict[str, Any] = None, models_dir: str = 'models/rl_execution'):
        """
        Initialize the RL execution system.
        
        Args:
            config: Configuration dictionary
            models_dir: Directory to save models
        """
        self.config = config or {}
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Default parameters
        self.action_size = self.config.get('action_size', 10)
        self.memory_size = self.config.get('memory_size', 10000)
        self.batch_size = self.config.get('batch_size', 64)
        self.gamma = self.config.get('gamma', 0.95)
        self.epsilon = self.config.get('epsilon', 1.0)
        self.epsilon_min = self.config.get('epsilon_min', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.update_target_every = self.config.get('update_target_every', 5)
        
        # Initialize agent
        self.agent = None
        self.environment = None
        
        # Training metrics
        self.training_history = {
            'episode_rewards': [],
            'losses': [],
            'implementation_shortfalls': [],
            'execution_times': []
        }
    
    def create_environment(self, price_data: pd.DataFrame, order_size: float, 
                          time_limit: int, market_impact_factor: float = 0.1,
                          execution_time_penalty: float = 0.01) -> None:
        """
        Create RL execution environment.
        
        Args:
            price_data: Historical price data
            order_size: Total order size
            time_limit: Maximum execution time
            market_impact_factor: Market impact factor
            execution_time_penalty: Time penalty factor
        """
        self.environment = RLExecutionEnvironment(
            price_data=price_data,
            order_size=order_size,
            time_limit=time_limit,
            market_impact_factor=market_impact_factor,
            execution_time_penalty=execution_time_penalty
        )
        
        # Initialize agent with correct state size
        state = self.environment.reset()
        state_size = len(state)
        
        self.agent = RLExecutionAgent(
            state_size=state_size,
            action_size=self.action_size,
            memory_size=self.memory_size,
            batch_size=self.batch_size,
            gamma=self.gamma,
            epsilon=self.epsilon,
            epsilon_min=self.epsilon_min,
            epsilon_decay=self.epsilon_decay,
            learning_rate=self.learning_rate
        )
        
        logger.info(f"RL execution environment created with state size {state_size}")
    
    def train(self, episodes: int = 1000, max_steps: int = 100, 
             save_best: bool = True, save_every: int = 100,
             render_every: int = 100) -> Dict[str, Any]:
        """
        Train the RL agent.
        
        Args:
            episodes: Number of episodes to train
            max_steps: Maximum steps per episode
            save_best: Whether to save best model
            save_every: Save model every N episodes
            render_every: Render execution every N episodes
            
        Returns:
            Training history
        """
        if self.environment is None or self.agent is None:
            raise ValueError("Environment and agent must be created before training")
        
        best_reward = -float('inf')
        best_shortfall = float('inf')
        
        for episode in range(episodes):
            # Reset environment
            is_buy_order = random.choice([True, False])
            state = self.environment.reset(is_buy_order=is_buy_order)
            
            total_reward = 0
            losses = []
            
            for step in range(max_steps):
                # Choose action
                action_idx = self.agent.act(state)
                action = action_idx / (self.action_size - 1)  # Convert to [0, 1]
                
                # Take action
                next_state, reward, done, info = self.environment.step(action)
                
                # Remember experience
                self.agent.remember(state, action_idx, reward, next_state, done)
                
                # Update state and accumulate reward
                state = next_state
                total_reward += reward
                
                # Train agent
                loss = self.agent.replay()
                if loss > 0:
                    losses.append(loss)
                
                # Update target model periodically
                if episode % self.update_target_every == 0 and step == 0:
                    self.agent.update_target_model()
                
                # Check if done
                if done:
                    break
            
            # Record metrics
            self.training_history['episode_rewards'].append(total_reward)
            self.training_history['losses'].append(np.mean(losses) if losses else 0)
            
            if 'implementation_shortfall' in info:
                self.training_history['implementation_shortfalls'].append(info['implementation_shortfall'])
            
            if 'execution_time' in info:
                self.training_history['execution_times'].append(info['execution_time'])
            
            # Save best model
            if save_best and total_reward > best_reward:
                best_reward = total_reward
                self.save_model('best_reward')
            
            if save_best and 'implementation_shortfall' in info and info['implementation_shortfall'] < best_shortfall:
                best_shortfall = info['implementation_shortfall']
                self.save_model('best_shortfall')
            
            # Save periodically
            if save_every > 0 and episode > 0 and episode % save_every == 0:
                self.save_model(f'episode_{episode}')
            
            # Render execution
            if render_every > 0 and episode > 0 and episode % render_every == 0:
                self._render_execution(episode)
            
            # Log progress
            if episode % 10 == 0:
                logger.info(f"Episode {episode}/{episodes}, Reward: {total_reward:.2f}, "
                           f"Epsilon: {self.agent.epsilon:.4f}, "
                           f"Shortfall: {info.get('implementation_shortfall', 'N/A')}")
        
        # Save final model
        self.save_model('final')
        
        # Plot training history
        self._plot_training_history()
        
        return self.training_history
    
    def execute_order(self, price_data: pd.DataFrame, order_size: float, 
                     is_buy_order: bool = True, start_idx: int = 0,
                     render: bool = True) -> Dict[str, Any]:
        """
        Execute an order using the trained RL agent.
        
        Args:
            price_data: Historical price data
            order_size: Order size to execute
            is_buy_order: Whether this is a buy order
            start_idx: Starting index in price data
            render: Whether to render execution
            
        Returns:
            Execution results
        """
        if self.environment is None or self.agent is None:
            raise ValueError("Environment and agent must be created before execution")
        
        # Create temporary environment with provided data
        temp_env = RLExecutionEnvironment(
            price_data=price_data,
            order_size=order_size,
            time_limit=self.environment.time_limit,
            market_impact_factor=self.environment.market_impact_factor,
            execution_time_penalty=self.environment.execution_time_penalty
        )
        
        # Reset environment
        state = temp_env.reset(start_idx=start_idx, is_buy_order=is_buy_order)
        
        # Execute order
        done = False
        total_reward = 0
        execution_log = []
        
        while not done:
            # Choose action (no exploration)
            action_idx = np.argmax(self.agent.model.predict(state.reshape(1, -1), verbose=0)[0])
            action = action_idx / (self.action_size - 1)  # Convert to [0, 1]
            
            # Take action
            next_state, reward, done, info = temp_env.step(action)
            
            # Log execution
            execution_log.append({
                'step': len(execution_log),
                'action': action,
                'execution_size': temp_env.executed_sizes[-1] if temp_env.executed_sizes else 0,
                'execution_price': temp_env.execution_prices[-1] if temp_env.execution_prices else 0,
                'market_impact': temp_env.market_impacts[-1] if temp_env.market_impacts else 0,
                'remaining_size': temp_env.remaining_size,
                'reward': reward
            })
            
            # Update state and accumulate reward
            state = next_state
            total_reward += reward
        
        # Get final performance
        performance = temp_env._calculate_performance()
        performance['total_reward'] = total_reward
        
        # Render execution if requested
        if render:
            self._render_execution_log(execution_log, performance, is_buy_order)
        
        return {
            'execution_log': execution_log,
            'performance': performance
        }
    
    def save_model(self, name: str = 'model') -> str:
        """
        Save agent model.
        
        Args:
            name: Model name
            
        Returns:
            Path to saved model
        """
        if self.agent is None:
            raise ValueError("Agent must be created before saving")
        
        filepath = os.path.join(self.models_dir, f"{name}.h5")
        self.agent.save(filepath)
        
        logger.info(f"Model saved to {filepath}")
        return filepath
    
    def load_model(self, name: str = 'model') -> None:
        """
        Load agent model.
        
        Args:
            name: Model name
        """
        if self.agent is None:
            raise ValueError("Agent must be created before loading")
        
        filepath = os.path.join(self.models_dir, f"{name}.h5")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found")
        
        self.agent.load(filepath)
        logger.info(f"Model loaded from {filepath}")
    
    def _render_execution(self, episode: int) -> None:
        """
        Render current execution strategy.
        
        Args:
            episode: Current episode number
        """
        # Create a test execution
        state = self.environment.reset()
        done = False
        execution_log = []
        
        while not done:
            # Choose action (no exploration)
            action_idx = np.argmax(self.agent.model.predict(state.reshape(1, -1), verbose=0)[0])
            action = action_idx / (self.action_size - 1)
            
            # Take action
            next_state, reward, done, info = self.environment.step(action)
            
            # Log execution
            execution_log.append({
                'step': len(execution_log),
                'action': action,
                'execution_size': self.environment.executed_sizes[-1] if self.environment.executed_sizes else 0,
                'execution_price': self.environment.execution_prices[-1] if self.environment.execution_prices else 0,
                'market_impact': self.environment.market_impacts[-1] if self.environment.market_impacts else 0,
                'remaining_size': self.environment.remaining_size,
                'reward': reward
            })
            
            # Update state
            state = next_state
        
        # Plot execution
        self._render_execution_log(execution_log, info, self.environment.is_buy_order, 
                                  title=f"Execution Strategy (Episode {episode})")
    
    def _render_execution_log(self, execution_log: List[Dict[str, Any]], 
                             performance: Dict[str, Any], is_buy_order: bool,
                             title: str = "Order Execution") -> None:
        """
        Render execution log.
        
        Args:
            execution_log: Execution log
            performance: Performance metrics
            is_buy_order: Whether this is a buy order
            title: Plot title
        """
        if not execution_log:
            return
        
        # Extract data
        steps = [log['step'] for log in execution_log]
        actions = [log['action'] for log in execution_log]
        execution_sizes = [log['execution_size'] for log in execution_log]
        execution_prices = [log['execution_price'] for log in execution_log]
        remaining_sizes = [log['remaining_size'] for log in execution_log]
        
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Plot execution sizes
        ax1.bar(steps, execution_sizes, alpha=0.7, color='blue')
        ax1.set_ylabel('Execution Size')
        ax1.set_title(title)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot remaining size
        ax2.plot(steps, remaining_sizes, 'o-', color='green')
        ax2.set_ylabel('Remaining Size')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Plot execution prices
        ax3.plot(steps, execution_prices, 'o-', color='red')
        ax3.set_ylabel('Execution Price')
        ax3.set_xlabel('Step')
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # Add performance metrics as text
        performance_text = (
            f"VWAP: {performance.get('vwap', 'N/A'):.4f}\n"
            f"Implementation Shortfall: {performance.get('implementation_shortfall', 'N/A'):.4f}%\n"
            f"Avg Market Impact: {performance.get('avg_market_impact', 'N/A'):.4f}%\n"
            f"Execution Time: {performance.get('execution_time', 'N/A')}\n"
            f"Order Type: {'Buy' if is_buy_order else 'Sell'}\n"
            f"Completion: {performance.get('pct_completed', 'N/A'):.1f}%"
        )
        
        fig.text(0.02, 0.02, performance_text, fontsize=10, 
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        # Save figure
        plt.savefig(os.path.join(self.models_dir, f"{title.replace(' ', '_').lower()}.png"), 
                   bbox_inches='tight', dpi=300)
        plt.close()
    
    def _plot_training_history(self) -> None:
        """
        Plot training history.
        """
        if not self.training_history['episode_rewards']:
            return
        
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Plot episode rewards
        ax1.plot(self.training_history['episode_rewards'], color='blue')
        ax1.set_ylabel('Episode Reward')
        ax1.set_title('Training History')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot losses
        if self.training_history['losses']:
            ax2.plot(self.training_history['losses'], color='red')
            ax2.set_ylabel('Loss')
            ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Plot implementation shortfall
        if self.training_history['implementation_shortfalls']:
            ax3.plot(self.training_history['implementation_shortfalls'], color='green')
            ax3.set_ylabel('Implementation Shortfall (%)')
            ax3.set_xlabel('Episode')
            ax3.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.models_dir, "training_history.png"), 
                   bbox_inches='tight', dpi=300)
        plt.close()
