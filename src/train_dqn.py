"""
Training script for DQN and Double DQN on LunarLander-v3.
Run this to train and save models + logs.

Usage:
    python train_dqn.py              # trains both DQN and Double DQN
    python train_dqn.py --dqn        # train only DQN
    python train_dqn.py --ddqn       # train only Double DQN
"""

import gymnasium as gym
import numpy as np
import argparse
import os
import time

from dqn_agent import DQNAgent
from utils import (plot_training_rewards, plot_epsilon_decay,
                   plot_losses, save_results, print_training_summary)


def train(agent, env, num_episodes=800, max_steps=1000):
    """Train the DQN agent and collect metrics."""
    all_rewards = []
    all_losses = []
    all_epsilons = []

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        episode_losses = []

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            loss = agent.step(state, action, reward, next_state, done)
            if loss is not None:
                episode_losses.append(loss)
            state = next_state
            episode_reward += reward

            if done:
                break

        agent.decay_epsilon()
        all_rewards.append(episode_reward)
        all_epsilons.append(agent.epsilon)
        if episode_losses:
            all_losses.append(np.mean(episode_losses))

        # print progress every 50 episodes
        if ep % 50 == 0:
            avg_reward = np.mean(all_rewards[-100:])
            print(f"Episode {ep}/{num_episodes} | "
                  f"Avg Reward (100): {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.4f}")

        # early stopping if solved
        if len(all_rewards) >= 100:
            recent_avg = np.mean(all_rewards[-100:])
            if recent_avg >= 200:
                print(f"\nSolved at episode {ep}! Average reward: {recent_avg:.2f}")
                break

    return all_rewards, all_losses, all_epsilons


def run_training(double_dqn=False, seed=42):
    """Full training pipeline for one algorithm variant."""
    algo_name = "Double DQN" if double_dqn else "DQN"
    save_dir = os.path.join('..', 'results', 'ddqn' if double_dqn else 'dqn')

    print(f"\n{'#'*60}")
    print(f"  Training {algo_name} on LunarLander-v3")
    print(f"{'#'*60}\n")

    env = gym.make('LunarLander-v3')

    agent = DQNAgent(
        state_dim=8,
        action_dim=4,
        seed=seed,
        lr=5e-4,
        gamma=0.99,
        tau=1e-3,
        buffer_size=100000,
        batch_size=64,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        update_every=4,
        double_dqn=double_dqn
    )

    start_time = time.time()
    rewards, losses, epsilons = train(agent, env, num_episodes=800)
    elapsed = time.time() - start_time

    print(f"\nTraining took {elapsed/60:.1f} minutes")

    # save model
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f'{"ddqn" if double_dqn else "dqn"}_model.pth')
    agent.save(model_path)
    print(f"Model saved to {model_path}")

    # save config for reproducibility
    config = {
        'algorithm': algo_name,
        'lr': 5e-4,
        'gamma': 0.99,
        'tau': 1e-3,
        'buffer_size': 100000,
        'batch_size': 64,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'update_every': 4,
        'seed': seed,
        'training_time_min': round(elapsed/60, 2),
        'total_episodes': len(rewards),
    }
    save_results(rewards, losses, epsilons, config, save_dir)

    # generate plots
    plot_training_rewards(
        rewards,
        title=f'{algo_name} - Training Rewards',
        save_path=os.path.join(save_dir, 'training_rewards.png')
    )
    if epsilons:
        plot_epsilon_decay(
            epsilons,
            save_path=os.path.join(save_dir, 'epsilon_decay.png')
        )

    print_training_summary(rewards, algo_name)
    env.close()

    return rewards


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dqn', action='store_true', help='Train DQN only')
    parser.add_argument('--ddqn', action='store_true', help='Train Double DQN only')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # if no flag given, train both
    if not args.dqn and not args.ddqn:
        dqn_rewards = run_training(double_dqn=False, seed=args.seed)
        ddqn_rewards = run_training(double_dqn=True, seed=args.seed)

        # comparison plot
        from utils import plot_comparison
        plot_comparison(
            {'DQN': dqn_rewards, 'Double DQN': ddqn_rewards},
            title='DQN vs Double DQN Comparison',
            save_path=os.path.join('..', 'results', 'dqn_vs_ddqn_comparison.png')
        )
    elif args.dqn:
        run_training(double_dqn=False, seed=args.seed)
    elif args.ddqn:
        run_training(double_dqn=True, seed=args.seed)
