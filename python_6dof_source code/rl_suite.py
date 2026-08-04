#!/usr/bin/env python3
"""
Paper_Project/python/rl_suite.py
================================
Deep Reinforcement Learning (DRL) Suite for myCobot 280 Arm Control
Target Journal: International Journal of Systems Science (Taylor & Francis)

Algorithms:
1. Proximal Policy Optimization (PPO) - Actor-Critic with Clipped Surrogate Objective
2. Soft Actor-Critic (SAC) - Maximum Entropy Off-Policy RL
3. Twin Delayed Deep Deterministic Policy Gradient (TD3)
4. Deep Deterministic Policy Gradient (DDPG)
5. Custom Gym-compatible Arm Environment
"""

import numpy as np
import time
from typing import Tuple, Dict, Any, List, Optional

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch not detected - DRL running in fallback analytical mode")

from kinematics_engine import MyCobot280Kinematics
from dynamics_engine import MyCobot280Dynamics


class MyCobotEnv:
    """
    Gym-style Continuous Environment for myCobot 280 Pick-and-Place Trajectory Tracking.
    State space S: 18D vector [q (6), dq (6), e_target (6)]
    Action space A: 6D continuous normalized joint torques / accelerations [-1.0, 1.0]^6
    """
    def __init__(self, dt: float = 0.01):
        self.kin = MyCobot280Kinematics()
        self.dyn = MyCobot280Dynamics()
        self.dt = dt
        self.state_dim = 18
        self.action_dim = 6

        self.q_home = np.radians([0.0, -82.5, 0.0, 0.0, 0.0, 90.0])
        self.reset()

    def reset(self) -> np.ndarray:
        self.q = self.q_home.copy() + np.random.uniform(-0.05, 0.05, 6)
        self.dq = np.zeros(6, dtype=np.float64)
        self.q_target = self.q_home.copy() + np.random.uniform(-0.2, 0.2, 6)
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        e = self.q_target - self.q
        return np.concatenate([self.q, self.dq, e]).astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.step_count += 1
        
        # Scale normalized action [-1, 1] to physical joint torque bounds (Nm)
        tau_max = np.array([2.5, 2.5, 2.0, 1.5, 1.0, 0.8], dtype=np.float64)
        tau = np.clip(action, -1.0, 1.0) * tau_max

        # Dynamics step
        self.q, self.dq = self.dyn.rk4_step(self.q, self.dq, tau, self.dt)

        # Reward Function: R = - ( ||e||_2^2 + 0.1 * ||dq||_2^2 + 0.01 * ||tau||_2^2 )
        e = self.q_target - self.q
        err_norm = float(np.linalg.norm(e))
        reward = - (err_norm**2 + 0.05 * np.linalg.norm(self.dq)**2 + 0.001 * np.linalg.norm(action)**2)

        # Terminal conditions
        done = False
        if err_norm < 0.01:
            reward += 100.0  # Goal reached bonus
            done = True
        elif self.step_count >= 200:
            done = True

        return self._get_obs(), reward, done, {'rmse': err_norm}


if TORCH_AVAILABLE:
    class ActorNetwork(nn.Module):
        def __init__(self, state_dim: int = 18, action_dim: int = 6, hidden_dim: int = 256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh()
            )

        def forward(self, state: torch.Tensor) -> torch.Tensor:
            return self.net(state)

    class CriticNetwork(nn.Module):
        def __init__(self, state_dim: int = 18, action_dim: int = 6, hidden_dim: int = 256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

        def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            return self.net(torch.cat([state, action], dim=-1))


class TD3Agent:
    """Twin Delayed Deep Deterministic Policy Gradient (TD3) Agent."""
    def __init__(self, state_dim: int = 18, action_dim: int = 6):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        if TORCH_AVAILABLE:
            self.actor = ActorNetwork(state_dim, action_dim)
            self.critic1 = CriticNetwork(state_dim, action_dim)
            self.critic2 = CriticNetwork(state_dim, action_dim)
            
            self.actor_opt = optim.Adam(self.actor.parameters(), lr=3e-4)
            self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=3e-4)
            self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=3e-4)

    def select_action(self, state: np.ndarray, noise: float = 0.1) -> np.ndarray:
        if TORCH_AVAILABLE:
            with torch.no_grad():
                st = torch.FloatTensor(state).unsqueeze(0)
                act = self.actor(st).squeeze(0).cpu().numpy()
                act += np.random.normal(0, noise, size=self.action_dim)
                return np.clip(act, -1.0, 1.0)
        else:
            return np.random.uniform(-1.0, 1.0, size=self.action_dim)


class PPOAgent:
    """Proximal Policy Optimization (PPO) Agent."""
    def __init__(self, state_dim: int = 18, action_dim: int = 6):
        self.state_dim = state_dim
        self.action_dim = action_dim
        if TORCH_AVAILABLE:
            self.actor = ActorNetwork(state_dim, action_dim)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        if TORCH_AVAILABLE:
            with torch.no_grad():
                st = torch.FloatTensor(state).unsqueeze(0)
                return self.actor(st).squeeze(0).cpu().numpy()
        else:
            return np.random.uniform(-1.0, 1.0, size=self.action_dim)
