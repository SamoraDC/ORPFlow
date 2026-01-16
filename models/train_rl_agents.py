"""
Train both D4PG+EVT and MARL agents for ORPFlow trading
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_and_prepare_data():
    """Load and prepare data for RL training"""
    data_path = project_root / "data" / "processed" / "features.parquet"

    if not data_path.exists():
        logger.error(f"Data file not found at {data_path}")
        raise FileNotFoundError(f"Data file not found at {data_path}")

    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")

    # Get feature columns (exclude metadata, OHLCV, and string columns)
    exclude_cols = {
        'open_time', 'close_time', 'open', 'high', 'low', 'close', 'volume',
        'symbol', 'ignore'  # String columns
    }
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    logger.info(f"Using {len(feature_cols)} features")

    # Extract OHLCV and features
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    data = df[ohlcv_cols].values
    features = df[feature_cols].values

    # Normalize features
    logger.info("Normalizing features with RobustScaler")
    scaler = RobustScaler()
    features = scaler.fit_transform(features)

    # Remove any NaN/Inf values
    mask = ~(np.isnan(features).any(axis=1) | np.isinf(features).any(axis=1))
    data = data[mask]
    features = features[mask]

    logger.info(f"After cleaning: {len(data)} rows")

    return data, features


def train_d4pg():
    """Train D4PG+EVT agent"""
    logger.info("=" * 60)
    logger.info("Training D4PG+EVT Agent")
    logger.info("=" * 60)

    from models.rl.d4pg_evt import D4PGAgent, TradingEnvironment

    data, features = load_and_prepare_data()

    # Use subset for faster training if data is too large
    max_samples = 50000
    if len(data) > max_samples:
        logger.info(f"Using last {max_samples} samples for training")
        data = data[-max_samples:]
        features = features[-max_samples:]

    state_dim = features.shape[1] + 4  # features + portfolio state
    logger.info(f"State dimension: {state_dim}")

    agent = D4PGAgent(state_dim=state_dim, batch_size=128)
    env = TradingEnvironment(data, features)

    episodes = 100
    max_steps = 5000
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
            var = agent.evt_model.var()
            cvar = agent.evt_model.cvar()

            logger.info(
                f"Episode {episode + 1}/{episodes} - "
                f"Return: {total_return:.2%}, Avg: {avg_return:.2%}, "
                f"VaR: {var:.4f}, CVaR: {cvar:.4f}"
            )

    # Save model
    model_dir = project_root / "trained"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "d4pg_evt_agent.pt"
    agent.save(str(model_path))
    logger.info(f"Model saved to {model_path}")

    # Export ONNX
    onnx_dir = model_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    agent.export_onnx(str(onnx_dir / "d4pg_actor.onnx"))

    # Print results
    print("\n" + "=" * 60)
    print("D4PG+EVT Agent Training Results")
    print("=" * 60)
    print(f"Total Episodes: {episodes}")
    print(f"Training Steps: {agent.training_step}")
    print(f"Best Return: {best_return:.2%}")
    print(f"Final Return: {returns_history[-1]:.2%}")
    print(f"Avg Last 10: {np.mean(returns_history[-10:]):.2%}")
    print(f"Final VaR (99%): {agent.evt_model.var():.4f}")
    print(f"Final CVaR (99%): {agent.evt_model.cvar():.4f}")
    print(f"Actor Losses: {len(agent.actor_losses)} updates")
    print(f"Critic Losses: {len(agent.critic_losses)} updates")

    return agent


def train_marl():
    """Train MARL system"""
    logger.info("\n" + "=" * 60)
    logger.info("Training MARL System")
    logger.info("=" * 60)

    from models.rl.marl import MARLSystem, MARLTradingEnvironment

    data, features = load_and_prepare_data()

    # Use subset for faster training
    max_samples = 50000
    if len(data) > max_samples:
        logger.info(f"Using last {max_samples} samples for training")
        data = data[-max_samples:]
        features = features[-max_samples:]

    state_dim = features.shape[1] + 4
    n_agents = 5

    logger.info(f"State dimension: {state_dim}")
    logger.info(f"Number of agents: {n_agents}")

    marl = MARLSystem(state_dim=state_dim, n_agents=n_agents, batch_size=128)
    env = MARLTradingEnvironment(data, features, n_agents)

    episodes = 100
    max_steps = 5000
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

    # Save model
    model_dir = project_root / "trained"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "marl_system.pt"
    marl.save(str(model_path))
    logger.info(f"Model saved to {model_path}")

    # Export ONNX
    onnx_dir = model_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    marl.export_onnx(str(onnx_dir / "marl"))

    # Print results
    print("\n" + "=" * 60)
    print("MARL System Training Results")
    print("=" * 60)
    print(f"Total Episodes: {episodes}")
    print(f"Training Steps: {marl.training_step}")
    print(f"Best Return: {best_return:.2%}")
    print(f"Final Return: {returns_history[-1]:.2%}")
    print(f"Avg Last 10: {np.mean(returns_history[-10:]):.2%}")
    print(f"Number of Agents: {marl.n_agents}")
    print("\nAgent Roles:")
    for i, agent in enumerate(marl.agents):
        print(f"  Agent {i}: {agent.config.role.value}")

    return marl


def main():
    """Main training pipeline"""
    try:
        # Train D4PG+EVT
        d4pg_agent = train_d4pg()

        # Train MARL
        marl_system = train_marl()

        print("\n" + "=" * 60)
        print("All RL Training Completed Successfully")
        print("=" * 60)
        print(f"Models saved to: {project_root / 'trained'}")
        print(f"ONNX exports at: {project_root / 'trained' / 'onnx'}")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
