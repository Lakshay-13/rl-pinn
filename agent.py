import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import logging
import pickle
from pathlib import Path
from typing import Optional, Dict

from core.device import get_device

logger = logging.getLogger(__name__)

class QNetwork(nn.Module):
    def __init__(self, state_shape, action_dims):
        """
        state_shape: (C, H, W) e.g. (7, 16, 41)
        action_dims: list of ints, e.g. [19, 15, 2, 11, ...]
        """
        super(QNetwork, self).__init__()
        self.action_dims = action_dims
        
        # Input: (Batch, 7, 16, 41)
        # Conv1: 64 filters, 1x1, stride 1, padding same (0) if 1x1
        # Original: Conv2D(64, (1, 1), padding='same', activation='relu')
        self.conv1 = nn.Conv2d(state_shape[0], 64, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        
        # Conv2: 32 filters, 1x1
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        
        # Flatten
        # Output size: 32 * 16 * 41 = 20992
        flatten_size = 32 * state_shape[1] * state_shape[2]
        
        self.fc1 = nn.Linear(flatten_size, 128)
        
        # Heads: One linear layer per action dimension
        self.heads = nn.ModuleList([nn.Linear(128, dim) for dim in action_dims])
        
    def forward(self, x):
        # x: (Batch, C, H, W)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        
        x = x.view(x.size(0), -1) # Flatten
        x = self.relu(self.fc1(x))
        
        # Returns list of tensors, one per head
        return [head(x) for head in self.heads]

class ReplayBuffer:
    def __init__(self, capacity=10000):
        # We remove maxlen to hold infinite memories as requested
        self.buffer = deque()
    
    def push(self, state, action, reward, next_state, done):
        # State/Next_state are assumed to be numpy arrays or tensors
        # Action is list/array of ints
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def save_to_disk(self, filepath):
        """Serializes the entire memory buffer to disk without discarding."""
        with open(filepath, 'wb') as f:
            pickle.dump(list(self.buffer), f)
            
    def load_from_disk(self, filepath):
        """Appends memories from a pickle file to the current buffer."""
        if not Path(filepath).exists():
            logger.warning(f"Memory file {filepath} not found.")
            return
        with open(filepath, 'rb') as f:
            memories = pickle.load(f)
            # Use extend to append loaded memories to the existing buffer
            self.buffer.extend(memories)
        logger.info(f"Loaded {len(memories)} memories from {filepath}. Total: {len(self.buffer)}")

    def clear(self):
        """Empties the memory buffer."""
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)

class PinnAgent:
    def __init__(self, state_shape, action_dims, device=None,
                 lr=0.001, gamma=0.99, epsilon=0.99, epsilon_decay=0.8, epsilon_min=0.5, n_steps=1):
        self.device = device if device else get_device()
        self.state_shape = state_shape
        self.action_dims = action_dims
        self.gamma = gamma
        self.n_steps = int(max(1, n_steps))
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Networks
        self.q_net = QNetwork(state_shape, action_dims).float().to(self.device)
        self.target_net = QNetwork(state_shape, action_dims).float().to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        
        self.memory = ReplayBuffer(capacity=50000)  # Capacity param ignored, truly unlimited
        self._best_weights: Optional[Dict[str, torch.Tensor]] = None
        self._best_loss: float = float("inf")
        self._update_target_counter: int = 0
        self._target_update_interval: int = 30

        # Action mappings (Precomputed from CINNS_RL)
        # 19 Learning Rates
        self.lr_options = np.concatenate((np.linspace(1e-4, 9e-4, 9), np.linspace(1e-3, 1e-2, 10)), axis=0).astype(np.float32)
        # 15 Beta combinations (from 6 values)
        beta_values = [0.8, 0.88, 0.888, 0.9, 0.99, 0.999]
        import itertools
        self.beta_options = list(itertools.combinations(beta_values, 2))
        # 11 Scales (Minimum 0.1 to avoid zeroing out equations, 11 options to match action_dims)
        self.scale_options = np.linspace(0.1, 1.0, 11).astype(np.float32)

    def decode_action(self, action_indices):
        """
        Converts discrete action indices to actual PINN hyperparameters.
        action_indices: [lr_idx, beta_idx, opt_idx, scale_idx_0, ..., scale_idx_6]
        """
        # 1. LR
        lr = self.lr_options[action_indices[0]]
        
        # 2. Betas
        betas = self.beta_options[action_indices[1]]
        
        # 3. Optimizer
        # CINNS_RL: 0 -> LBFGS, 1 -> Adam
        # But in map_actions_to_params: "optimizer = 1 #permanently selecting Adam for poc"
        # I will respect the 'poc' override but keep logic available.
        optimizer_name = 'adam' # Always Adam per original code override
        
        # 4. Scales (indices 3 to 9)
        scale_indices = action_indices[3:]
        raw_scales = np.array([self.scale_options[i] for i in scale_indices])
        
        # Normalize Scales Logic from CINNS_RL
        # normalized_numbers = scales**2/np.sqrt(np.sum(scales**2))
        # normalization_factor = np.sqrt(np.sum(normalized_numbers ** 2))
        # normalized_numbers /= normalization_factor
        
        eps = 1e-8
        sq_sum = np.sqrt(np.sum(raw_scales**2) + eps)
        norm1 = raw_scales**2 / (sq_sum + eps)
        norm2_sum = np.sqrt(np.sum(norm1**2) + eps)
        final_scales = norm1 / (norm2_sum + eps)
        
        return {
            'lr': float(lr),
            'betas': betas,
            'optimizer': optimizer_name,
            's_v': final_scales
        }

    def act(self, state, eval_mode=False):
        """
        state: np.array (7, 16, 41) or tensor
        Returns: list of ints (indices)
        """
        if not eval_mode and random.random() < self.epsilon:
            return [random.randint(0, dim-1) for dim in self.action_dims]
        
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0) # Batch dim
        
        with torch.no_grad():
            q_heads = self.q_net(state_t)
            # q_heads is list of (Batch, Dim)
            actions = []
            for head in q_heads:
                action_idx = head.argmax(dim=1).item()
                actions.append(action_idx)
        
        return actions

    def train_step(self, batch_size=64):
        if len(self.memory) < batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device) # (B, 10)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1) # (B, 1)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1) # (B, 1)
        
        # Compute Targets
        with torch.no_grad():
            next_q_heads = self.target_net(next_states_t)
            # next_q_heads: list of (B, Dim_i)
            # We need max Q for each head
            max_next_q_list = []
            for i, head in enumerate(next_q_heads):
                max_q, _ = head.max(dim=1, keepdim=True) # (B, 1)
                max_next_q_list.append(max_q)
            
            # Independent Bellman update for each dimension?
            # CINNS_RL calculates loss for each dimension and averages them.
            # Q_target = r + gamma * max Q_next (if not done)
            
            targets_list = []
            for max_q in max_next_q_list:
                target = rewards_t + (self.gamma ** self.n_steps) * max_q * (1 - dones_t)
                targets_list.append(target)
                
        # Compute Current Q
        curr_q_heads = self.q_net(states_t)
        loss = 0
        
        for i, head in enumerate(curr_q_heads):
            # Gather Q-values for chosen actions
            # head: (B, Dim_i)
            # actions_t[:, i]: (B,)
            
            gathered_q = head.gather(1, actions_t[:, i].unsqueeze(1)) # (B, 1)
            target = targets_list[i]
            
            loss += self.loss_fn(gathered_q, target)
            
        # Average loss across dimensions
        loss = loss / len(self.action_dims)
        loss_value = loss.item()

        # Track best weights for restoration after training
        if loss_value < self._best_loss:
            self._best_loss = loss_value
            self._best_weights = {k: v.clone() for k, v in self.q_net.state_dict().items()}

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically instead of every step
        self._update_target_counter += 1
        if self._update_target_counter >= self._target_update_interval:
            self._update_target_counter = 0
            self.update_target_net()

        return loss_value

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_net(self, tau=0.001):
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def restore_best_weights(self) -> None:
        """Restore the network weights from the best loss point during training."""
        if self._best_weights is not None:
            self.q_net.load_state_dict(self._best_weights)

    def reset_training_state(self) -> None:
        """Reset best weights tracking for a new training cycle."""
        self._best_weights = None
        self._best_loss = float("inf")
        self._update_target_counter = 0

    def to_device(self, device):
        """Moves models and optimizer state to the specified device."""
        self.device = device
        self.q_net.to(device)
        self.target_net.to(device)
        
        # Move optimizer state
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        logger.info(f"Agent moved to {device}")

    def save(self, filepath, include_memory=False):
        """Persist the online/target networks and optimizer state."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "q_net_state_dict": self.q_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "state_shape": self.state_shape,
            "action_dims": self.action_dims,
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "best_weights": self._best_weights,
            "best_loss": self._best_loss,
        }
        if include_memory:
            data["memory"] = list(self.memory.buffer)
            
        torch.save(data, filepath)

    def load(self, filepath, load_memory=False):
        """Restore the agent from a checkpoint written by save()."""
        state = torch.load(filepath, map_location=self.device)
        self.q_net.load_state_dict(state["q_net_state_dict"])
        self.target_net.load_state_dict(state["target_net_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        for opt_state in self.optimizer.state.values():
            for k, v in opt_state.items():
                if isinstance(v, torch.Tensor):
                    opt_state[k] = v.to(self.device)
        self.epsilon = float(state["epsilon"])
        self.state_shape = tuple(state.get("state_shape", self.state_shape))
        self.action_dims = list(state.get("action_dims", self.action_dims))
        self.gamma = float(state.get("gamma", self.gamma))
        self.n_steps = int(state.get("n_steps", self.n_steps))
        self.epsilon_decay = float(state.get("epsilon_decay", self.epsilon_decay))
        self.epsilon_min = float(state.get("epsilon_min", self.epsilon_min))
        self._best_weights = state.get("best_weights")
        self._best_loss = float(state.get("best_loss", float("inf")))
        
        if load_memory and "memory" in state:
            self.memory.buffer.clear()
            self.memory.buffer.extend(state["memory"])
            logger.info(f"Restored {len(self.memory)} transitions from checkpoint memory.")

        self.target_net.eval()
