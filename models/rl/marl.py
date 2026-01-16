"""
Multi-Agent Reinforcement Learning (MARL) for Trading
Cooperative multi-agent system with specialized roles
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass
from enum import Enum
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Specialized agent roles in the trading system"""
    TREND_FOLLOWER = "trend_follower"
    MEAN_REVERTER = "mean_reverter"
    MOMENTUM_TRADER = "momentum_trader"
    RISK_MANAGER = "risk_manager"
    COORDINATOR = "coordinator"


@dataclass
class AgentConfig:
    """Configuration for individual agent"""
    role: AgentRole
    state_dim: int
    action_dim: int
    hidden_dim: int = 128
    learning_rate: float = 3e-4


class CommunicationChannel:
    """Message passing between agents"""

    def __init__(self, n_agents: int, message_dim: int = 32):
        self.n_agents = n_agents
        self.message_dim = message_dim
        self.messages = {i: np.zeros(message_dim) for i in range(n_agents)}

    def send(self, sender_id: int, message: np.ndarray):
        self.messages[sender_id] = message

    def receive(self, receiver_id: int) -> np.ndarray:
        """Receive aggregated messages from all other agents"""
        other_messages = [
            self.messages[i] for i in range(self.n_agents) if i != receiver_id
        ]
        if not other_messages:
            return np.zeros(self.message_dim)
        return np.mean(other_messages, axis=0)

    def get_all_messages(self) -> np.ndarray:
        return np.stack(list(self.messages.values()))


class AttentionModule(nn.Module):
    """Multi-head attention for agent communication"""

    def __init__(self, embed_dim: int, n_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attention(queries, keys, values)
        return self.layer_norm(queries + attn_output)


class AgentNetwork(nn.Module):
    """Individual agent policy network with communication"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        message_dim: int = 32,
        n_agents: int = 5,
    ):
        super().__init__()

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.message_encoder = nn.Sequential(
            nn.Linear(message_dim * (n_agents - 1), hidden_dim // 2),
            nn.ReLU(),
        )

        self.message_generator = nn.Sequential(
            nn.Linear(hidden_dim, message_dim),
            nn.Tanh(),
        )

        # Policy head
        self.policy = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

        # Value head
        self.value = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        messages: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_embedding = self.state_encoder(state)
        message_embedding = self.message_encoder(messages.flatten(start_dim=-2))

        combined = torch.cat([state_embedding, message_embedding], dim=-1)

        action = self.policy(combined)
        value = self.value(combined)

        outgoing_message = self.message_generator(state_embedding)

        return action, value, outgoing_message


class CentralizedCritic(nn.Module):
    """Centralized critic for CTDE (Centralized Training, Decentralized Execution)"""

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        total_state_dim = n_agents * state_dim
        total_action_dim = n_agents * action_dim

        self.fc1 = nn.Linear(total_state_dim + total_action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, n_agents)  # Q-value per agent

        self.attention = AttentionModule(hidden_dim, n_heads=4)

        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        states: torch.Tensor,  # (batch, n_agents, state_dim)
        actions: torch.Tensor,  # (batch, n_agents, action_dim)
    ) -> torch.Tensor:
        batch_size = states.size(0)

        states_flat = states.view(batch_size, -1)
        actions_flat = actions.view(batch_size, -1)

        x = torch.cat([states_flat, actions_flat], dim=-1)

        x = F.relu(self.layer_norm1(self.fc1(x)))
        x = F.relu(self.layer_norm2(self.fc2(x)))
        x = F.relu(self.fc3(x))

        q_values = self.fc_out(x)

        return q_values


class MARLAgent:
    """Individual MARL agent"""

    def __init__(
        self,
        agent_id: int,
        config: AgentConfig,
        n_agents: int,
        message_dim: int = 32,
        device: torch.device = torch.device("cpu"),
    ):
        self.agent_id = agent_id
        self.config = config
        self.device = device

        self.network = AgentNetwork(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            message_dim=message_dim,
            n_agents=n_agents,
        ).to(device)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
        )

        self.last_message = torch.zeros(message_dim)

    def select_action(
        self,
        state: np.ndarray,
        messages: np.ndarray,
        evaluate: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        messages_tensor = torch.FloatTensor(messages).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, _, message = self.network(state_tensor, messages_tensor)

        action = action.cpu().numpy()[0]
        message = message.cpu().numpy()[0]

        if not evaluate:
            action = action + np.random.normal(0, 0.1, size=action.shape)
            action = np.clip(action, -1, 1)

        self.last_message = message

        return action, message


class MARLSystem:
    """Multi-Agent RL System for trading"""

    def __init__(
        self,
        state_dim: int,
        n_agents: int = 5,
        action_dim: int = 1,
        hidden_dim: int = 128,
        message_dim: int = 32,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 256,
        buffer_size: int = 500_000,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.message_dim = message_dim

        # Create specialized agents
        self.agents: List[MARLAgent] = []
        roles = list(AgentRole)

        for i in range(n_agents):
            role = roles[i % len(roles)]
            config = AgentConfig(
                role=role,
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
            )
            agent = MARLAgent(i, config, n_agents, message_dim, self.device)
            self.agents.append(agent)

        # Centralized critic
        self.critic = CentralizedCritic(
            n_agents, state_dim, action_dim, hidden_dim * 2
        ).to(self.device)
        self.critic_target = CentralizedCritic(
            n_agents, state_dim, action_dim, hidden_dim * 2
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        # Communication channel
        self.comm_channel = CommunicationChannel(n_agents, message_dim)

        # Replay buffer
        self.buffer = MARLReplayBuffer(buffer_size, n_agents)

        # Reward shaping weights per role
        self.role_weights = {
            AgentRole.TREND_FOLLOWER: {"trend": 1.5, "risk": 0.5},
            AgentRole.MEAN_REVERTER: {"mean_reversion": 1.5, "risk": 0.5},
            AgentRole.MOMENTUM_TRADER: {"momentum": 1.5, "risk": 0.5},
            AgentRole.RISK_MANAGER: {"risk": 2.0, "return": 0.5},
            AgentRole.COORDINATOR: {"coordination": 1.5, "return": 1.0},
        }

        self.training_step = 0
        logger.info(f"MARL system initialized with {n_agents} agents on {self.device}")

    def select_actions(
        self,
        states: np.ndarray,  # (n_agents, state_dim)
        evaluate: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select actions for all agents"""
        actions = []
        messages = []

        for i, agent in enumerate(self.agents):
            received_messages = self.comm_channel.receive(i)
            other_messages = np.stack([
                self.comm_channel.messages[j] for j in range(self.n_agents) if j != i
            ])

            action, message = agent.select_action(
                states[i], other_messages, evaluate
            )

            actions.append(action)
            messages.append(message)
            self.comm_channel.send(i, message)

        return np.array(actions), np.array(messages)

    def get_aggregated_action(
        self,
        individual_actions: np.ndarray,
        role_confidence: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Aggregate individual actions into a single trading decision"""
        if role_confidence is None:
            role_confidence = np.ones(self.n_agents) / self.n_agents

        weighted_action = np.sum(individual_actions * role_confidence[:, None], axis=0)
        final_action = np.clip(weighted_action, -1, 1)

        return final_action

    def store_transition(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        messages: np.ndarray,
    ):
        self.buffer.add(states, actions, rewards, next_states, dones, messages)

    def train(self) -> Dict:
        if len(self.buffer) < self.batch_size:
            return {}

        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones, messages = batch

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Critic update
        with torch.no_grad():
            next_actions = []
            for i, agent in enumerate(self.agents):
                agent_next_states = next_states[:, i]
                other_messages = messages[:, [j for j in range(self.n_agents) if j != i]]
                other_messages_tensor = torch.FloatTensor(other_messages).to(self.device)

                next_action, _, _ = agent.network(agent_next_states, other_messages_tensor)
                next_actions.append(next_action)

            next_actions = torch.stack(next_actions, dim=1)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # Actor update for each agent
        actor_losses = []
        for i, agent in enumerate(self.agents):
            agent_states = states[:, i]
            other_messages = messages[:, [j for j in range(self.n_agents) if j != i]]
            other_messages_tensor = torch.FloatTensor(other_messages).to(self.device)

            agent_action, _, _ = agent.network(agent_states, other_messages_tensor)

            # Construct joint actions
            joint_actions = actions.clone()
            joint_actions[:, i] = agent_action

            actor_loss = -self.critic(states, joint_actions)[:, i].mean()

            agent.optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.network.parameters(), 1.0)
            agent.optimizer.step()

            actor_losses.append(actor_loss.item())

        # Soft update target critic
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.training_step += 1

        return {
            "critic_loss": critic_loss.item(),
            "actor_losses": actor_losses,
            "mean_actor_loss": np.mean(actor_losses),
        }

    def shape_rewards(
        self,
        base_reward: float,
        market_state: Dict,
    ) -> np.ndarray:
        """Shape rewards based on agent roles"""
        rewards = np.zeros(self.n_agents)

        for i, agent in enumerate(self.agents):
            role = agent.config.role
            weights = self.role_weights[role]

            reward = base_reward

            if "trend" in weights and "trend_strength" in market_state:
                reward += weights["trend"] * market_state["trend_strength"] * base_reward

            if "mean_reversion" in weights and "deviation" in market_state:
                reward += weights["mean_reversion"] * (1 - abs(market_state["deviation"])) * abs(base_reward)

            if "momentum" in weights and "momentum" in market_state:
                reward += weights["momentum"] * market_state["momentum"] * base_reward

            if "risk" in weights and "volatility" in market_state:
                risk_penalty = -weights["risk"] * market_state["volatility"] * abs(base_reward)
                reward += risk_penalty

            if "coordination" in weights:
                # Reward for agreement with other agents
                coordination_bonus = weights["coordination"] * 0.1
                reward += coordination_bonus

            rewards[i] = reward

        return rewards

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        state_dict = {
            "critic_state": self.critic.state_dict(),
            "critic_target_state": self.critic_target.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "training_step": self.training_step,
        }

        for i, agent in enumerate(self.agents):
            state_dict[f"agent_{i}_network"] = agent.network.state_dict()
            state_dict[f"agent_{i}_optimizer"] = agent.optimizer.state_dict()
            state_dict[f"agent_{i}_role"] = agent.config.role.value

        torch.save(state_dict, path)
        logger.info(f"MARL system saved to {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)

        self.critic.load_state_dict(checkpoint["critic_state"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.training_step = checkpoint["training_step"]

        for i, agent in enumerate(self.agents):
            agent.network.load_state_dict(checkpoint[f"agent_{i}_network"])
            agent.optimizer.load_state_dict(checkpoint[f"agent_{i}_optimizer"])

        logger.info(f"MARL system loaded from {path}")

    def export_onnx(self, path: str):
        """Export individual agent networks to ONNX"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        for i, agent in enumerate(self.agents):
            agent.network.eval()

            dummy_state = torch.randn(1, self.state_dim).to(self.device)
            dummy_messages = torch.randn(1, self.n_agents - 1, self.message_dim).to(self.device)

            agent_path = str(Path(path).parent / f"marl_agent_{i}_{agent.config.role.value}.onnx")

            # Export only the forward pass that produces action
            class ActionExtractor(nn.Module):
                def __init__(self, network):
                    super().__init__()
                    self.network = network

                def forward(self, state, messages):
                    action, _, _ = self.network(state, messages)
                    return action

            extractor = ActionExtractor(agent.network)

            torch.onnx.export(
                extractor,
                (dummy_state, dummy_messages),
                agent_path,
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=["state", "messages"],
                output_names=["action"],
                dynamic_axes={
                    "state": {0: "batch_size"},
                    "messages": {0: "batch_size"},
                    "action": {0: "batch_size"},
                },
            )

            logger.info(f"Agent {i} ({agent.config.role.value}) exported to {agent_path}")


class MARLReplayBuffer:
    """Replay buffer for MARL"""

    def __init__(self, capacity: int, n_agents: int):
        self.capacity = capacity
        self.n_agents = n_agents
        self.buffer = deque(maxlen=capacity)

    def add(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        messages: np.ndarray,
    ):
        self.buffer.append((states, actions, rewards, next_states, dones, messages))

    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, messages = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            np.array(messages),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class MARLTradingEnvironment:
    """Trading environment for MARL"""

    def __init__(
        self,
        data: np.ndarray,
        features: np.ndarray,
        n_agents: int = 5,
        initial_balance: float = 100_000,
        transaction_cost: float = 0.0005,
    ):
        self.data = data
        self.features = features
        self.n_agents = n_agents
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost

        self.reset()

    def reset(self) -> np.ndarray:
        self.balance = self.initial_balance
        self.position = 0.0
        self.step_idx = 0
        self.portfolio_values = [self.initial_balance]
        self.returns = []

        return self._get_states()

    def _get_states(self) -> np.ndarray:
        """Get state for each agent with slight variations"""
        base_state = self.features[self.step_idx]

        portfolio_state = np.array([
            self.position,
            self.balance / self.initial_balance - 1,
            np.mean(self.returns[-20:]) if self.returns else 0,
            np.std(self.returns[-20:]) if len(self.returns) > 1 else 0,
        ])

        states = []
        for i in range(self.n_agents):
            # Add agent-specific noise for exploration
            agent_state = np.concatenate([
                base_state + np.random.normal(0, 0.01, size=base_state.shape),
                portfolio_state,
            ])
            states.append(agent_state)

        return np.array(states)

    def _get_market_state(self) -> Dict:
        """Get market state for reward shaping"""
        if len(self.returns) < 20:
            return {
                "trend_strength": 0,
                "momentum": 0,
                "deviation": 0,
                "volatility": 0,
            }

        returns = np.array(self.returns[-20:])
        prices = self.data[max(0, self.step_idx - 20):self.step_idx, 3]

        trend_strength = np.mean(returns) / (np.std(returns) + 1e-8)
        momentum = returns[-1] - np.mean(returns)

        if len(prices) > 1:
            ma = np.mean(prices)
            deviation = (prices[-1] - ma) / ma
        else:
            deviation = 0

        volatility = np.std(returns) * np.sqrt(252 * 24 * 60)

        return {
            "trend_strength": trend_strength,
            "momentum": momentum,
            "deviation": deviation,
            "volatility": volatility,
        }

    def step(
        self,
        aggregated_action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        target_position = float(np.clip(aggregated_action[0], -1, 1))
        position_change = target_position - self.position

        current_price = self.data[self.step_idx, 3]
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

            base_reward = step_return * 100
        else:
            base_reward = 0.0

        market_state = self._get_market_state()
        next_states = self._get_states() if not done else np.zeros((self.n_agents, self.features.shape[1] + 4))

        info = {
            "balance": self.balance,
            "position": self.position,
            "return": self.returns[-1] if self.returns else 0,
            "market_state": market_state,
        }

        return next_states, base_reward, done, info


def train_marl(
    data: np.ndarray,
    features: np.ndarray,
    n_agents: int = 5,
    episodes: int = 300,
    max_steps: int = 5000,
) -> MARLSystem:
    """Train MARL system"""

    state_dim = features.shape[1] + 4
    marl = MARLSystem(state_dim=state_dim, n_agents=n_agents)

    env = MARLTradingEnvironment(data, features, n_agents)

    best_return = -float("inf")
    returns_history = []

    for episode in range(episodes):
        states = env.reset()
        episode_return = 0

        for step in range(max_steps):
            actions, messages = marl.select_actions(states)
            aggregated_action = marl.get_aggregated_action(actions)

            next_states, base_reward, done, info = env.step(aggregated_action)

            # Shape rewards based on roles
            rewards = marl.shape_rewards(base_reward, info["market_state"])

            dones = np.full(n_agents, done)
            marl.store_transition(states, actions, rewards, next_states, dones, messages)

            train_info = marl.train()

            states = next_states
            episode_return += base_reward

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
                f"Best: {best_return:.2%}"
            )

    return marl


def main():
    """Train MARL system on historical data"""
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

    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    features = scaler.fit_transform(features)

    marl = train_marl(data, features, n_agents=5, episodes=200)

    model_dir = Path(__file__).parent.parent / "trained"
    model_dir.mkdir(parents=True, exist_ok=True)

    marl.save(str(model_dir / "marl_system.pt"))
    marl.export_onnx(str(model_dir / "onnx" / "marl"))

    print("\n" + "=" * 50)
    print("MARL System Results")
    print("=" * 50)
    print(f"Training Steps: {marl.training_step}")
    print(f"Number of Agents: {marl.n_agents}")
    for i, agent in enumerate(marl.agents):
        print(f"  Agent {i}: {agent.config.role.value}")


if __name__ == "__main__":
    main()
