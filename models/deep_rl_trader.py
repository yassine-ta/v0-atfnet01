#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Deep Reinforcement Learning module for ATFNet.
Implements DRL agents for optimizing trading decisions.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
import random
from collections import deque
import time
from datetime import datetime
import joblib

# Try to import tensorflow, with fallback to PyTorch
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model, load_model
    from tensorflow.keras.layers import Dense, LSTM, Input, Concatenate, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    BACKEND = 'tensorflow'
except ImportError:
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        BACKEND = 'pytorch'
    except ImportError:
        raise ImportError("Either TensorFlow or PyTorch is required for the Deep RL module")

logger = logging.getLogger('atfnet.models.deep_rl')


class TradingEnvironment:
    """
    Trading environment for reinforcement learning.
    Simulates a trading environment for RL agents to interact with.
    """
    
    def __init__(self, data: pd.DataFrame, features: List[str], 
                window_size: int = 50, initial_balance: float = 10000,
                commission: float = 0.001, slippage: float = 0.0005,
                reward_scaling: float = 0.01, max_position: int = 1):
        """
        Initialize the trading environment.
        
        Args:
            data: DataFrame with price data and features
            features: List of feature columns to use
            window_size: Size of observation window
            initial_balance: Initial account balance
            commission: Trading commission (percentage)
            slippage: Slippage (percentage)
            reward_scaling: Scaling factor for rewards
            max_position: Maximum position size
        """
        self.data = data
        self.features = features
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        self.reward_scaling = reward_scaling
        self.max_position = max_position
        
        # Validate data
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        for feature in self.features:
            if feature not in self.data.columns:
                raise ValueError(f"Feature '{feature}' not found in data")
        
        # Initialize state
        self.reset()
        
        logger.info(f"Trading environment initialized with {len(data)} bars and {len(features)} features")
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial observation
        """
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.total_reward = 0
        self.history = []
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (0: hold, 1: buy, 2: sell)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Get current price data
        current_price = self.data.iloc[self.current_step]
        
        # Calculate reward based on action
        reward = 0
        info = {}
        
        # Execute action
        if action == 1:  # Buy
            if self.position <= 0:
                # Close short position if exists
                if self.position < 0:
                    reward += self._close_position(current_price['close'])
                
                # Open long position
                reward += self._open_position(1, current_price['close'])
                info['action'] = 'buy'
            else:
                # Already in long position
                reward -= 0.1  # Small penalty for redundant action
                info['action'] = 'hold'
        
        elif action == 2:  # Sell
            if self.position >= 0:
                # Close long position if exists
                if self.position > 0:
                    reward += self._close_position(current_price['close'])
                
                # Open short position
                reward += self._open_position(-1, current_price['close'])
                info['action'] = 'sell'
            else:
                # Already in short position
                reward -= 0.1  # Small penalty for redundant action
                info['action'] = 'hold'
        
        else:  # Hold
            # Calculate unrealized PnL
            if self.position != 0:
                unrealized_pnl = self.position * (current_price['close'] - self.entry_price)
                unrealized_pnl -= abs(self.position) * self.entry_price * self.commission
                reward += unrealized_pnl * self.reward_scaling
            
            info['action'] = 'hold'
        
        # Update history
        self.history.append({
            'step': self.current_step,
            'timestamp': self.data.index[self.current_step],
            'price': current_price['close'],
            'balance': self.balance,
            'position': self.position,
            'action': info['action'],
            'reward': reward
        })
        
        # Move to next step
        self.current_step += 1
        self.total_reward += reward
        
        # Check if done
        done = self.current_step >= len(self.data) - 1
        
        # Get next observation
        observation = self._get_observation()
        
        # Add info
        info['balance'] = self.balance
        info['position'] = self.position
        info['total_reward'] = self.total_reward
        
        return observation, reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation.
        
        Returns:
            Observation array
        """
        # Get window of data
        start = max(0, self.current_step - self.window_size)
        end = self.current_step
        
        # Extract features
        features = self.data.iloc[start:end][self.features].values
        
        # Add position info
        position_info = np.array([[self.position, self.entry_price if self.position != 0 else 0]])
        
        # Combine features and position info
        observation = np.concatenate([features, np.tile(position_info, (features.shape[0], 1))], axis=1)
        
        return observation
    
    def _open_position(self, direction: int, price: float) -> float:
        """
        Open a new position.
        
        Args:
            direction: Position direction (1 for long, -1 for short)
            price: Entry price
            
        Returns:
            Immediate reward
        """
        # Apply slippage
        if direction > 0:
            adjusted_price = price * (1 + self.slippage)  # Buy at slightly higher price
        else:
            adjusted_price = price * (1 - self.slippage)  # Sell at slightly lower price
        
        # Calculate position size
        position_size = min(self.max_position, int(self.balance / adjusted_price))
        
        if position_size == 0:
            return 0  # Not enough balance
        
        # Update state
        self.position = direction * position_size
        self.entry_price = adjusted_price
        
        # Apply commission
        commission_cost = abs(self.position) * adjusted_price * self.commission
        self.balance -= commission_cost
        
        return -commission_cost * self.reward_scaling  # Immediate cost of opening position
    
    def _close_position(self, price: float) -> float:
        """
        Close current position.
        
        Args:
            price: Exit price
            
        Returns:
            Reward from closing position
        """
        if self.position == 0:
            return 0
        
        # Apply slippage
        if self.position > 0:
            adjusted_price = price * (1 - self.slippage)  # Sell at slightly lower price
        else:
            adjusted_price = price * (1 + self.slippage)  # Buy at slightly higher price
        
        # Calculate PnL
        pnl = self.position * (adjusted_price - self.entry_price)
        
        # Apply commission
        commission_cost = abs(self.position) * adjusted_price * self.commission
        pnl -= commission_cost
        
        # Update balance
        self.balance += pnl + abs(self.position) * adjusted_price
        
        # Reset position
        self.position = 0
        self.entry_price = 0
        
        return pnl * self.reward_scaling
    
    def render(self, mode: str = 'human') -> None:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode
        """
        if mode == 'human':
            # Print current state
            current_price = self.data.iloc[self.current_step]['close']
            print(f"Step: {self.current_step}, Price: {current_price:.2f}, "
                 f"Balance: {self.balance:.2f}, Position: {self.position}, "
                 f"Total Reward: {self.total_reward:.2f}")
    
    def plot_performance(self, save_plot: bool = True,
                        filename: str = 'trading_performance.png') -> None:
        """
        Plot trading performance.
        
        Args:
            save_plot: Whether to save the plot
            filename: Filename for saved plot
        """
        if not self.history:
            logger.warning("No trading history to plot")
            return
        
        try:
            # Extract data
            steps = [h['step'] for h in self.history]
            prices = [h['price'] for h in self.history]
            balances = [h['balance'] for h in self.history]
            positions = [h['position'] for h in self.history]
            actions = [h['action'] for h in self.history]
            
            # Create figure
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            
            # Plot price
            ax1.plot(steps, prices, 'b-', label='Price')
            ax1.set_ylabel('Price')
            ax1.set_title('Trading Performance')
            ax1.legend(loc='upper left')
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Plot balance
            ax2.plot(steps, balances, 'g-', label='Balance')
            ax2.set_ylabel('Balance')
            ax2.legend(loc='upper left')
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # Plot position
            ax3.plot(steps, positions, 'r-', label='Position')
            ax3.set_ylabel('Position')
            ax3.set_xlabel('Step')
            ax3.legend(loc='upper left')
            ax3.grid(True, linestyle='--', alpha=0.7)
            
            # Mark buy/sell actions
            for i, action in enumerate(actions):
                if action == 'buy':
                    ax1.plot(steps[i], prices[i], 'g^', markersize=8)
                elif action == 'sell':
                    ax1.plot(steps[i], prices[i], 'rv', markersize=8)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            if save_plot:
                plt.savefig(filename, bbox_inches='tight', dpi=300)
                logger.info(f"Trading performance plot saved to {filename}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting trading performance: {e}")
            raise


class DQNAgent:
    """
    Deep Q-Network agent for trading.
    Implements DQN algorithm for learning optimal trading policies.
    """
    
    def __init__(self, state_size: Tuple[int, int], action_size: int, 
                config: Dict[str, Any] = None, output_dir: str = 'dqn_agent'):
        """
        Initialize the DQN agent.
        
        Args:
            state_size: Size of state space (window_size, n_features)
            action_size: Size of action space
            config: Configuration dictionary
            output_dir: Directory to save outputs
        """
        self.state_size = state_size
        self.action_size = action_size
        self.config = config or {}
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load default parameters
        self.gamma = self.config.get('gamma', 0.95)  # Discount factor
        self.epsilon = self.config.get('epsilon', 1.0)  # Exploration rate
        self.epsilon_min = self.config.get('epsilon_min', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.batch_size = self.config.get('batch_size', 32)
        self.train_start = self.config.get('train_start', 1000)
        self.update_target_freq = self.config.get('update_target_freq', 100)
        
        # Initialize memory
        self.memory = deque(maxlen=self.config.get('memory_size', 10000))
        
        # Initialize step counter
        self.step_counter = 0
        
        # Build model
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # Initialize training history
        self.train_history = []
        
        logger.info(f"DQN agent initialized with state size {state_size} and action size {action_size}")
    
    def _build_model(self) -> Any:
        """
        Build neural network model.
        
        Returns:
            Model
        """
        if BACKEND == 'tensorflow':
            return self._build_tensorflow_model()
        else:
            return self._build_pytorch_model()
    
    def _build_tensorflow_model(self) -> Any:
        """
        Build TensorFlow model.
        
        Returns:
            TensorFlow model
        """
        # Input layers
        input_features = Input(shape=self.state_size)
        
        # LSTM layers
        x = LSTM(64, return_sequences=True)(input_features)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = LSTM(32)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Dense layers
        x = Dense(16, activation='relu')(x)
        
        # Output layer
        outputs = Dense(self.action_size, activation='linear')(x)
        
        # Create model
        model = Model(inputs=input_features, outputs=outputs)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        
        return model
    
    def _build_pytorch_model(self) -> Any:
        """
        Build PyTorch model.
        
        Returns:
            PyTorch model
        """
        class DQNModel(nn.Module):
            def __init__(self, input_shape, action_size):
                super(DQNModel, self).__init__()
                self.lstm1 = nn.LSTM(input_shape[1], 64, batch_first=True, return_sequences=True)
                self.bn1 = nn.BatchNorm1d(64)
                self.dropout1 = nn.Dropout(0.2)
                
                self.lstm2 = nn.LSTM(64, 32, batch_first=True)
                self.bn2 = nn.BatchNorm1d(32)
                self.dropout2 = nn.Dropout(0.2)
                
                self.fc = nn.Linear(32, 16)
                self.output = nn.Linear(16, action_size)
            
            def forward(self, x):
                # LSTM layers
                x, _ = self.lstm1(x)
                x = x.permute(0, 2, 1)  # Change for batch norm
                x = self.bn1(x)
                x = x.permute(0, 2, 1)  # Change back
                x = self.dropout1(x)
                
                x, _ = self.lstm2(x)
                x = self.bn2(x.squeeze(1))
                x = self.dropout2(x)
                
                # Dense layers
                x = torch.relu(self.fc(x))
                x = self.output(x)
                
                return x
        
        model = DQNModel(self.state_size, self.action_size)
        model.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        model.loss_fn = nn.MSELoss()
        
        return model
    
    def update_target_model(self) -> None:
        """
        Update target model with weights from main model.
        """
        if BACKEND == 'tensorflow':
            self.target_model.set_weights(self.model.get_weights())
        else:
            self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool) -> None:
        """
        Store experience in memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Choose action based on state.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        if BACKEND == 'tensorflow':
            q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0]
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(np.expand_dims(state, axis=0))
                q_values = self.model(state_tensor).cpu().numpy()[0]
        
        return np.argmax(q_values)
    
    def replay(self) -> float:
        """
        Train model on batch of experiences.
        
        Returns:
            Loss value
        """
        if len(self.memory) < self.train_start:
            return 0
        
        # Sample batch from memory
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        
        if BACKEND == 'tensorflow':
            return self._replay_tensorflow(minibatch)
        else:
            return self._replay_pytorch(minibatch)
    
    def _replay_tensorflow(self, minibatch) -> float:
        """
        Train TensorFlow model on batch of experiences.
        
        Args:
            minibatch: Batch of experiences
            
        Returns:
            Loss value
        """
        # Extract batch data
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        # Predict Q-values
        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        # Update targets for actions taken
        for i in range(len(minibatch)):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train model
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        
        # Update target model periodically
        self.step_counter += 1
        if self.step_counter % self.update_target_freq == 0:
            self.update_target_model()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss
    
    def _replay_pytorch(self, minibatch) -> float:
        """
        Train PyTorch model on batch of experiences.
        
        Args:
            minibatch: Batch of experiences
            
        Returns:
            Loss value
        """
        # Extract batch data
        states = torch.FloatTensor(np.array([experience[0] for experience in minibatch]))
        actions = torch.LongTensor(np.array([experience[1] for experience in minibatch]))
        rewards = torch.FloatTensor(np.array([experience[2] for experience in minibatch]))
        next_states = torch.FloatTensor(np.array([experience[3] for experience in minibatch]))
        dones = torch.FloatTensor(np.array([experience[4] for experience in minibatch]))
        
        # Compute current Q values
        self.model.train()
        q_values = self.model(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute next Q values
        self.target_model.eval()
        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.model.loss_fn(q_values, target_q_values)
        
        # Optimize the model
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()
        
        # Update target model periodically
        self.step_counter += 1
        if self.step_counter % self.update_target_freq == 0:
            self.update_target_model()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def train(self, env: TradingEnvironment, episodes: int = 100, 
             render_freq: int = 10, save_freq: int = 10) -> Dict[str, Any]:
        """
        Train agent on environment.
        
        Args:
            env: Trading environment
            episodes: Number of episodes
            render_freq: Frequency of rendering
            save_freq: Frequency of saving model
            
        Returns:
            Training results
        """
        # Initialize results
        results = {
            'episode_rewards': [],
            'episode_profits': [],
            'episode_losses': []
        }
        
        # Training loop
        for episode in range(1, episodes + 1):
            # Reset environment
            state = env.reset()
            total_reward = 0
            losses = []
            
            # Episode loop
            done = False
            while not done:
                # Choose action
                action = self.act(state)
                
                # Take action
                next_state, reward, done, info = env.step(action)
                
                # Store experience
                self.remember(state, action, reward, next_state, done)
                
                # Train model
                loss = self.replay()
                if loss > 0:
                    losses.append(loss)
                
                # Update state and reward
                state = next_state
                total_reward += reward
                
                # Render environment
                if episode % render_freq == 0:
                    env.render()
            
            # Calculate episode profit
            initial_balance = env.initial_balance
            final_balance = env.balance
            profit = final_balance - initial_balance
            profit_pct = (profit / initial_balance) * 100
            
            # Store results
            results['episode_rewards'].append(total_reward)
            results['episode_profits'].append(profit_pct)
            results['episode_losses'].append(np.mean(losses) if losses else 0)
            
            # Log progress
            logger.info(f"Episode {episode}/{episodes}, Reward: {total_reward:.2f}, "
                       f"Profit: {profit_pct:.2f}%, Epsilon: {self.epsilon:.4f}")
            
            # Save model
            if episode % save_freq == 0:
                self.save_model(f"model_episode_{episode}")
            
            # Store training history
            self.train_history.append({
                'episode': episode,
                'reward': total_reward,
                'profit': profit_pct,
                'loss': np.mean(losses) if losses else 0,
                'epsilon': self.epsilon
            })
        
        # Save final model
        self.save_model("model_final")
        
        # Plot training history
        self.plot_training_history()
        
        return results
    
    def test(self, env: TradingEnvironment) -> Dict[str, Any]:
        """
        Test agent on environment.
        
        Args:
            env: Trading environment
            
        Returns:
            Test results
        """
        # Reset environment
        state = env.reset()
        
        # Test loop
        done = False
        while not done:
            # Choose action (no exploration)
            action = self.act(state, training=False)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Update state
            state = next_state
        
        # Calculate test results
        initial_balance = env.initial_balance
        final_balance = env.balance
        profit = final_balance - initial_balance
        profit_pct = (profit / initial_balance) * 100
        
        # Log results
        logger.info(f"Test completed. Initial balance: {initial_balance:.2f}, "
                   f"Final balance: {final_balance:.2f}, Profit: {profit_pct:.2f}%")
        
        # Plot performance
        env.plot_performance(filename=os.path.join(self.output_dir, 'test_performance.png'))
        
        # Prepare results
        results = {
            'initial_balance': initial_balance,
            'final_balance': final_balance,
            'profit': profit,
            'profit_pct': profit_pct,
            'history': env.history
        }
        
        return results
    
    def save_model(self, filename: str = 'model') -> str:
        """
        Save model to file.
        
        Args:
            filename: Base filename
            
        Returns:
            Path to saved model
        """
        filepath = os.path.join(self.output_dir, f"{filename}")
        
        if BACKEND == 'tensorflow':
            self.model.save(f"{filepath}.h5")
            logger.info(f"Model saved to {filepath}.h5")
            return f"{filepath}.h5"
        else:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.model.optimizer.state_dict(),
                'epsilon': self.epsilon
            }, f"{filepath}.pt")
            logger.info(f"Model saved to {filepath}.pt")
            return f"{filepath}.pt"
    
    def load_model(self, filepath: str) -> None:
        """
        Load model from file.
        
        Args:
            filepath: Path to model file
        """
        if not os.path.exists(filepath):
            logger.warning(f"Model file {filepath} not found")
            return
        
        if BACKEND == 'tensorflow':
            self.model = load_model(filepath)
            self.target_model = load_model(filepath)
            logger.info(f"Model loaded from {filepath}")
        else:
            checkpoint = torch.load(filepath)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            
            self.target_model.load_state_dict(self.model.state_dict())
            logger.info(f"Model loaded from {filepath}")
    
    def plot_training_history(self, save_plot: bool = True,
                             filename: str = 'training_history.png') -> None:
        """
        Plot training history.
        
        Args:
            save_plot: Whether to save the plot
            filename: Filename for saved plot
        """
        if not self.train_history:
            logger.warning("No training history to plot")
            return
        
        try:
            # Extract data
            episodes = [h['episode'] for h in self.train_history]
            rewards = [h['reward'] for h in self.train_history]
            profits = [h['profit'] for h in self.train_history]
            losses = [h['loss'] for h in self.train_history]
            epsilons = [h['epsilon'] for h in self.train_history]
            
            # Create figure
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            
            # Plot rewards
            ax1.plot(episodes, rewards, 'b-', label='Reward')
            ax1.set_ylabel('Reward')
            ax1.set_title('Training History')
            ax1.legend(loc='upper left')
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Plot profits
            ax2.plot(episodes, profits, 'g-', label='Profit %')
            ax2.set_ylabel('Profit %')
            ax2.legend(loc='upper left')
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # Plot losses and epsilon
            ax3.plot(episodes, losses, 'r-', label='Loss')
            ax3.set_ylabel('Loss')
            ax3.set_xlabel('Episode')
            ax3.legend(loc='upper left')
            ax3.grid(True, linestyle='--', alpha=0.7)
            
            # Add epsilon on secondary y-axis
            ax4 = ax3.twinx()
            ax4.plot(episodes, epsilons, 'm--', label='Epsilon')
            ax4.set_ylabel('Epsilon')
            ax4.legend(loc='upper right')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            if save_plot:
                plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight', dpi=300)
                logger.info(f"Training history plot saved to {self.output_dir}/{filename}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting training history: {e}")
            raise


class DeepRLTrader:
    """
    Deep Reinforcement Learning Trader for ATFNet.
    Integrates DRL agents with ATFNet for trading.
    """
    
    def __init__(self, config: Dict[str, Any] = None, output_dir: str = 'deep_rl_trader'):
        """
        Initialize the Deep RL Trader.
        
        Args:
            config: Configuration dictionary
            output_dir: Directory to save outputs
        """
        self.config = config or {}
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load default parameters
        self.window_size = self.config.get('window_size', 50)
        self.batch_size = self.config.get('batch_size', 32)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.gamma = self.config.get('gamma', 0.95)
        self.epsilon = self.config.get('epsilon', 1.0)
        self.epsilon_min = self.config.get('epsilon_min', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
        self.memory_size = self.config.get('memory_size', 10000)
        self.train_episodes = self.config.get('train_episodes', 100)
        
        # Initialize agent
        self.agent = None
        self.env = None
        
        logger.info("Deep RL Trader initialized")
    
    def prepare_data(self, data: pd.DataFrame, features: List[str] = None) -> pd.DataFrame:
        """
        Prepare data for training.
        
        Args:
            data: DataFrame with price data
            features: List of feature columns to use
            
        Returns:
            Prepared DataFrame
        """
        # Ensure required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Use default features if not provided
        if features is None:
            # Basic technical indicators
            features = ['rsi_14', 'macd', 'macd_signal', 'bb_upper', 'bb_middle', 'bb_lower', 
                       'atr_14', 'ema_9', 'ema_21', 'cci_20', 'adx_14']
        
        # Check if features exist, add them if not
        for feature in features:
            if feature not in data.columns:
                logger.warning(f"Feature '{feature}' not found in data. Adding technical indicators.")
                data = self._add_technical_indicators(data)
                break
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        
        feature_data = data[features].copy()
        scaler = StandardScaler()
        data[features] = scaler.fit_transform(feature_data)
        
        # Store feature list and scaler
        self.features = features
        self.scaler = scaler
        
        logger.info(f"Data prepared with {len(features)} features")
        
        return data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to data.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with technical indicators
        """
        # Create copy to avoid modifying original
        df = data.copy()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Calculate Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        std_dev = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (std_dev * 2)
        df['bb_lower'] = df['bb_middle'] - (std_dev * 2)
        
        # Calculate ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr_14'] = true_range.rolling(window=14).mean()
        
        # Calculate EMAs
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        
        # Calculate CCI
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        mean_dev = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        df['cci_20'] = (typical_price - typical_price.rolling(window=20).mean()) / (0.015 * mean_dev)
        
        # Calculate ADX
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr = true_range
        plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / tr.ewm(alpha=1/14).mean())
        minus_di = abs(100 * (minus_dm.ewm(alpha=1/14).mean() / tr.ewm(alpha=1/14).mean()))
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        df['adx_14'] = dx.ewm(alpha=1/14).mean()
        
        # Fill NaN values
        df.fillna(method='bfill', inplace=True)
        
        logger.info("Technical indicators added to data")
        
        return df
    
    def create_environment(self, data: pd.DataFrame, initial_balance: float = 10000,
                          commission: float = 0.001) -> TradingEnvironment:
        """
        Create trading environment.
        
        Args:
            data: DataFrame with price data and features
            initial_balance: Initial account balance
            commission: Trading commission
            
        Returns:
            Trading environment
        """
        # Create environment
        env = TradingEnvironment(
            data=data,
            features=self.features,
            window_size=self.window_size,
            initial_balance=initial_balance,
            commission=commission
        )
        
        # Store environment
        self.env = env
        
        logger.info(f"Trading environment created with {len(data)} bars")
        
        return env
    
    def create_agent(self) -> DQNAgent:
        """
        Create DQN agent.
        
        Returns:
            DQN agent
        """
        if self.env is None:
            raise ValueError("Environment not created. Call create_environment first.")
        
        # Get state and action sizes
        state_size = (self.window_size, len(self.features) + 2)  # +2 for position info
        action_size = 3  # hold, buy, sell
        
        # Create agent config
        agent_config = {
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'memory_size': self.memory_size
        }
        
        # Create agent
        agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            config=agent_config,
            output_dir=self.output_dir
        )
        
        # Store agent
        self.agent = agent
        
        logger.info(f"DQN agent created with state size {state_size} and action size {action_size}")
        
        return agent
    
    def train(self, data: pd.DataFrame = None, features: List[str] = None,
             episodes: int = None) -> Dict[str, Any]:
        """
        Train the agent.
        
        Args:
            data: DataFrame with price data (optional if already set)
            features: List of feature columns to use (optional if already set)
            episodes: Number of episodes (optional if already set)
            
        Returns:
            Training results
        """
        # Prepare data if provided
        if data is not None:
            data = self.prepare_data(data, features)
        elif self.env is None:
            raise ValueError("No data provided and environment not created")
        
        # Create environment if not already created
        if self.env is None:
            self.create_environment(data)
        
        # Create agent if not already created
        if self.agent is None:
            self.create_agent()
        
        # Set episodes
        if episodes is not None:
            self.train_episodes = episodes
        
        # Train agent
        results = self.agent.train(
            env=self.env,
            episodes=self.train_episodes
        )
        
        logger.info(f"Agent trained for {self.train_episodes} episodes")
        
        return results
    
    def test(self, data: pd.DataFrame = None, features: List[str] = None) -> Dict[str, Any]:
        """
        Test the agent.
        
        Args:
            data: DataFrame with price data (optional if already set)
            features: List of feature columns to use (optional if already set)
            
        Returns:
            Test results
        """
        # Prepare data if provided
        if data is not None:
            data = self.prepare_data(data, features)
        elif self.env is None:
            raise ValueError("No data provided and environment not created")
        
        # Create environment if not already created
        if self.env is None:
            self.create_environment(data)
        
        # Create agent if not already created
        if self.agent is None:
            self.create_agent()
        
        # Test agent
        results = self.agent.test(env=self.env)
        
        logger.info("Agent tested")
        
        return results
    
    def save(self, filepath: str = None) -> str:
        """
        Save trader state.
        
        Args:
            filepath: Path to save state (default: output_dir/trader.pkl)
            
        Returns:
            Path to saved state
        """
        if filepath is None:
            filepath = os.path.join(self.output_dir, 'trader.pkl')
        
        # Save agent model
        model_path = self.agent.save_model("model_final")
        
        # Prepare state to save
        state = {
            'config': self.config,
            'features': self.features,
            'model_path': model_path,
            'scaler': self.scaler
        }
        
        # Save state
        joblib.dump(state, filepath)
        logger.info(f"Trader state saved to {filepath}")
        
        return filepath
    
    def load(self, filepath: str = None) -> None:
        """
        Load trader state.
        
        Args:
            filepath: Path to load state from (default: output_dir/trader.pkl)
        """
        if filepath is None:
            filepath = os.path.join(self.output_dir, 'trader.pkl')
        
        if not os.path.exists(filepath):
            logger.warning(f"State file {filepath} not found")
            return
        
        # Load state
        state = joblib.load(filepath)
        
        # Restore state
        self.config = state.get('config', {})
        self.features = state.get('features', [])
        self.scaler = state.get('scaler')
        
        # Update parameters from config
        self.window_size = self.config.get('window_size', 50)
        self.batch_size = self.config.get('batch_size', 32)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.gamma = self.config.get('gamma', 0.95)
        self.epsilon = self.config.get('epsilon', 1.0)
        self.epsilon_min = self.config.get('epsilon_min', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
        self.memory_size = self.config.get('memory_size', 10000)
        self.train_episodes = self.config.get('train_episodes', 100)
        
        # Create agent
        state_size = (self.window_size, len(self.features) + 2)  # +2 for position info
        action_size = 3  # hold, buy, sell
        
        agent_config = {
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'memory_size': self.memory_size
        }
        
        self.agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            config=agent_config,
            output_dir=self.output_dir
        )
        
        # Load agent model
        model_path = state.get('model_path')
        if model_path and os.path.exists(model_path):
            self.agent.load_model(model_path)
        
        logger.info(f"Trader state loaded from {filepath}")
