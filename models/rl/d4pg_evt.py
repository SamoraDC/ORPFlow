"""
D4PG + EVT: Distributed Distributional DDPG with Extreme Value Theory
Advanced RL agent for risk-aware trading with tail risk modeling
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration"""

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class CriticNetwork(nn.Module):
    """Distributional critic network for D4PG"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
    ):
        super().__init__()

        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.register_buffer(
            "atoms",
            torch.linspace(v_min, v_max, n_atoms)
        )
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, n_atoms)

        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)

        x = F.relu(self.layer_norm1(self.fc1(x)))
        x = F.relu(self.layer_norm2(self.fc2(x)))
        x = F.relu(self.fc3(x))

        logits = self.fc_out(x)
        probs = F.softmax(logits, dim=-1)

        return probs

    def get_q_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        probs = self.forward(state, action)
        q_value = (probs * self.atoms).sum(dim=-1)
        return q_value


class ActorNetwork(nn.Module):
    """Actor network with deterministic policy"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        action_scale: float = 1.0,
    ):
        super().__init__()

        self.action_scale = action_scale

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = NoisyLinear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, action_dim)

        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer_norm1(self.fc1(state)))
        x = F.relu(self.layer_norm2(self.fc2(x)))
        x = F.relu(self.fc3(x))

        action = torch.tanh(self.fc_out(x)) * self.action_scale
        return action

    def reset_noise(self):
        self.fc3.reset_noise()


class EVTRiskModel:
    """Extreme Value Theory model for tail risk estimation"""

    def __init__(self, threshold_percentile: float = 95.0):
        self.threshold_percentile = threshold_percentile
        self.losses = []
        self.shape = None  # xi (shape parameter)
        self.scale = None  # sigma (scale parameter)
        self.threshold = None

    def update(self, returns: np.ndarray):
        """Update EVT model with new returns data"""
        losses = -returns[returns < 0]
        self.losses.extend(losses.tolist())

        if len(self.losses) < 100:
            return

        losses_array = np.array(self.losses)
        self.threshold = np.percentile(losses_array, self.threshold_percentile)

        exceedances = losses_array[losses_array > self.threshold] - self.threshold

        if len(exceedances) < 10:
            return

        try:
            # Fit Generalized Pareto Distribution
            self.shape, _, self.scale = stats.genpareto.fit(exceedances, floc=0)
        except Exception as e:
            logger.warning(f"EVT fitting failed: {e}")

    def var(self, confidence: float = 0.99) -> float:
        """Calculate Value at Risk using EVT"""
        if self.shape is None or self.scale is None:
            return 0.0

        n = len(self.losses)
        n_exceedances = sum(1 for l in self.losses if l > self.threshold)

        if n_exceedances == 0:
            return 0.0

        p = n_exceedances / n
        q = 1 - confidence

        if self.shape == 0:
            var = self.threshold + self.scale * np.log(p / q)
        else:
            var = self.threshold + (self.scale / self.shape) * ((p / q) ** self.shape - 1)

        return float(var)

    def cvar(self, confidence: float = 0.99) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = self.var(confidence)

        if self.shape is None or self.scale is None:
            return var

        if self.shape >= 1:
            return float("inf")

        cvar = var / (1 - self.shape) + (self.scale - self.shape * self.threshold) / (1 - self.shape)
        return float(cvar)


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer"""

    def __init__(
        self,
        capacity: int = 1_000_000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        experience = (state, action, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple:
        n = len(self.buffer)
        priorities = self.priorities[:n] ** self.alpha
        probabilities = priorities / priorities.sum()

        indices = np.random.choice(n, batch_size, p=probabilities, replace=False)

        weights = (n * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment)

        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices,
            weights,
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        return len(self.buffer)


class TradingEnvironment:
    """Simulated trading environment for RL training"""

    def __init__(
        self,
        data: np.ndarray,
        features: np.ndarray,
        initial_balance: float = 100_000,
        transaction_cost: float = 0.0005,
        max_position: float = 1.0,
    ):
        self.data = data  # OHLCV data
        self.features = features  # Pre-computed features
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position

        self.reset()

    def reset(self) -> np.ndarray:
        self.balance = self.initial_balance
        self.position = 0.0
        self.step_idx = 0
        self.portfolio_values = [self.initial_balance]
        self.returns = []

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        market_state = self.features[self.step_idx]
        portfolio_state = np.array([
            self.position / self.max_position,
            self.balance / self.initial_balance - 1,
            len(self.returns) > 0 and np.mean(self.returns[-20:]) or 0,
            len(self.returns) > 0 and np.std(self.returns[-20:]) or 0,
        ])

        return np.concatenate([market_state, portfolio_state])

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        target_position = float(np.clip(action[0], -self.max_position, self.max_position))
        position_change = target_position - self.position

        current_price = self.data[self.step_idx, 3]  # Close price

        transaction_cost = abs(position_change) * current_price * self.transaction_cost

        self.step_idx += 1
        done = self.step_idx >= len(self.data) - 1

        if not done:
            next_price = self.data[self.step_idx, 3]
            price_return = (next_price - current_price) / current_price

            pnl = self.position * price_return * self.balance - transaction_cost
            self.balance += pnl

            step_return = pnl / self.portfolio_values[-1]
            self.returns.append(step_return)
            self.portfolio_values.append(self.balance)

            self.position = target_position

            reward = self._calculate_reward(step_return)
        else:
            reward = 0.0

        info = {
            "balance": self.balance,
            "position": self.position,
            "return": self.returns[-1] if self.returns else 0,
        }

        next_state = self._get_state() if not done else np.zeros_like(self._get_state())

        return next_state, reward, done, info

    def _calculate_reward(self, step_return: float) -> float:
        # Risk-adjusted reward (Sharpe-like)
        if len(self.returns) < 2:
            return step_return * 100

        mean_return = np.mean(self.returns[-20:])
        std_return = np.std(self.returns[-20:]) + 1e-8

        sharpe_component = mean_return / std_return

        # Penalize drawdown
        peak = max(self.portfolio_values)
        drawdown = (peak - self.balance) / peak
        drawdown_penalty = -drawdown * 0.5

        reward = sharpe_component + drawdown_penalty + step_return * 10

        return float(reward)


class D4PGAgent:
    """D4PG agent with EVT risk management"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        hidden_dim: int = 256,
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 1e-4,
        critic_lr: float = 3e-4,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        n_step: int = 5,
        risk_aversion: float = 0.5,
        var_confidence: float = 0.99,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_step = n_step
        self.risk_aversion = risk_aversion
        self.var_confidence = var_confidence

        # Networks
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = CriticNetwork(state_dim, action_dim, hidden_dim, n_atoms, v_min, v_max).to(self.device)
        self.critic2 = CriticNetwork(state_dim, action_dim, hidden_dim, n_atoms, v_min, v_max).to(self.device)
        self.critic1_target = CriticNetwork(state_dim, action_dim, hidden_dim, n_atoms, v_min, v_max).to(self.device)
        self.critic2_target = CriticNetwork(state_dim, action_dim, hidden_dim, n_atoms, v_min, v_max).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # Replay buffer
        self.buffer = PrioritizedReplayBuffer(buffer_size)

        # N-step buffer
        self.n_step_buffer = deque(maxlen=n_step)

        # EVT risk model
        self.evt_model = EVTRiskModel()

        # Training stats
        self.training_step = 0
        self.actor_losses = []
        self.critic_losses = []

        logger.info(f"D4PG+EVT agent initialized on {self.device}")

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]

        if not evaluate:
            self.actor.reset_noise()

            # Apply EVT-based risk adjustment
            var = self.evt_model.var(self.var_confidence)
            if var > 0:
                risk_scale = max(0.1, 1.0 - self.risk_aversion * var)
                action = action * risk_scale

        return action

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.n_step_buffer.append((state, action, reward, next_state, done))

        if len(self.n_step_buffer) == self.n_step:
            n_step_reward = sum(
                self.gamma ** i * t[2] for i, t in enumerate(self.n_step_buffer)
            )
            first_state = self.n_step_buffer[0][0]
            first_action = self.n_step_buffer[0][1]
            last_next_state = self.n_step_buffer[-1][3]
            last_done = self.n_step_buffer[-1][4]

            self.buffer.add(first_state, first_action, n_step_reward, last_next_state, last_done)

        # Update EVT model
        self.evt_model.update(np.array([reward]))

    def train(self) -> Dict:
        if len(self.buffer) < self.batch_size:
            return {}

        states, actions, rewards, next_states, dones, indices, weights = self.buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_states)

            target_probs1 = self.critic1_target(next_states, next_actions)
            target_probs2 = self.critic2_target(next_states, next_actions)
            target_probs = torch.min(target_probs1, target_probs2)

            target_atoms = rewards + (1 - dones) * (self.gamma ** self.n_step) * self.critic1.atoms.unsqueeze(0)
            target_atoms = target_atoms.clamp(self.critic1.v_min, self.critic1.v_max)

            b = (target_atoms - self.critic1.v_min) / self.critic1.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            l = l.clamp(0, self.critic1.n_atoms - 1)
            u = u.clamp(0, self.critic1.n_atoms - 1)

            target_dist = torch.zeros_like(target_probs)
            offset = torch.linspace(0, (self.batch_size - 1) * self.critic1.n_atoms, self.batch_size).long().unsqueeze(1).to(self.device)

            target_dist.view(-1).index_add_(0, (l + offset).view(-1), (target_probs * (u.float() - b)).view(-1))
            target_dist.view(-1).index_add_(0, (u + offset).view(-1), (target_probs * (b - l.float())).view(-1))

        current_probs1 = self.critic1(states, actions)
        current_probs2 = self.critic2(states, actions)

        critic1_loss = -(target_dist * torch.log(current_probs1 + 1e-8)).sum(dim=1)
        critic2_loss = -(target_dist * torch.log(current_probs2 + 1e-8)).sum(dim=1)

        critic1_loss = (critic1_loss * weights.squeeze()).mean()
        critic2_loss = (critic2_loss * weights.squeeze()).mean()

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step()

        # Actor update
        actor_actions = self.actor(states)
        actor_loss = -self.critic1.get_q_value(states, actor_actions).mean()

        # Add EVT risk penalty to actor loss
        var = self.evt_model.var(self.var_confidence)
        cvar = self.evt_model.cvar(self.var_confidence)
        risk_penalty = self.risk_aversion * (var + cvar) * 0.1 if var > 0 else 0

        actor_loss = actor_loss + risk_penalty

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # Update priorities
        with torch.no_grad():
            td_errors = torch.abs(
                rewards.squeeze() +
                (1 - dones.squeeze()) * self.gamma * self.critic1.get_q_value(next_states, self.actor_target(next_states)) -
                self.critic1.get_q_value(states, actions)
            ).cpu().numpy()
            self.buffer.update_priorities(indices, td_errors)

        # Soft update targets
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.training_step += 1
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append((critic1_loss.item() + critic2_loss.item()) / 2)

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": (critic1_loss.item() + critic2_loss.item()) / 2,
            "var": var,
            "cvar": cvar,
        }

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor_state": self.actor.state_dict(),
            "critic1_state": self.critic1.state_dict(),
            "critic2_state": self.critic2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic1_optimizer": self.critic1_optimizer.state_dict(),
            "critic2_optimizer": self.critic2_optimizer.state_dict(),
            "training_step": self.training_step,
        }, path)
        logger.info(f"D4PG agent saved to {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state"])
        self.critic1.load_state_dict(checkpoint["critic1_state"])
        self.critic2.load_state_dict(checkpoint["critic2_state"])
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic1_optimizer.load_state_dict(checkpoint["critic1_optimizer"])
        self.critic2_optimizer.load_state_dict(checkpoint["critic2_optimizer"])
        self.training_step = checkpoint["training_step"]
        logger.info(f"D4PG agent loaded from {path}")

    def export_onnx(self, path: str):
        """Export actor network to ONNX for inference"""
        self.actor.eval()

        dummy_input = torch.randn(1, self.state_dim).to(self.device)

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            self.actor,
            dummy_input,
            path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["state"],
            output_names=["action"],
            dynamic_axes={
                "state": {0: "batch_size"},
                "action": {0: "batch_size"},
            },
        )

        logger.info(f"D4PG actor exported to ONNX: {path}")


def train_d4pg(
    data: np.ndarray,
    features: np.ndarray,
    episodes: int = 500,
    max_steps: int = 5000,
) -> D4PGAgent:
    """Train D4PG+EVT agent"""

    state_dim = features.shape[1] + 4  # features + portfolio state
    agent = D4PGAgent(state_dim=state_dim)

    env = TradingEnvironment(data, features)

    best_return = -float("inf")
    returns_history = []

    for episode in range(episodes):
        state = env.reset()
        episode_return = 0
        episode_steps = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)
            train_info = agent.train()

            state = next_state
            episode_return += reward
            episode_steps += 1

            if done:
                break

        total_return = (env.balance - env.initial_balance) / env.initial_balance
        returns_history.append(total_return)

        if total_return > best_return:
            best_return = total_return

        if (episode + 1) % 10 == 0:
            avg_return = np.mean(returns_history[-10:])
            logger.info(
                f"Episode {episode + 1}/{episodes} - "
                f"Return: {total_return:.2%}, Avg: {avg_return:.2%}, "
                f"VaR: {agent.evt_model.var():.4f}"
            )

    return agent


def main():
    """Train D4PG+EVT agent on historical data"""
    import sys
    sys.path.append(str(Path(__file__).parent.parent))

    from data.preprocessor import FeatureEngineer
    import pandas as pd

    data_path = Path(__file__).parent.parent / "data" / "processed" / "features.parquet"

    if not data_path.exists():
        logger.error("Processed features not found. Run preprocessor.py first.")
        return

    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} rows")

    engineer = FeatureEngineer()
    feature_cols = engineer.get_feature_columns(df)

    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    data = df[ohlcv_cols].values
    features = df[feature_cols].values

    # Normalize features
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    features = scaler.fit_transform(features)

    agent = train_d4pg(data, features, episodes=200)

    model_dir = Path(__file__).parent.parent / "trained"
    model_dir.mkdir(parents=True, exist_ok=True)

    agent.save(str(model_dir / "d4pg_evt_agent.pt"))
    agent.export_onnx(str(model_dir / "onnx" / "d4pg_actor.onnx"))

    print("\n" + "=" * 50)
    print("D4PG+EVT Agent Results")
    print("=" * 50)
    print(f"Training Steps: {agent.training_step}")
    print(f"Final VaR (99%): {agent.evt_model.var():.4f}")
    print(f"Final CVaR (99%): {agent.evt_model.cvar():.4f}")


if __name__ == "__main__":
    main()
