"""
Training script for TD3 on LunarLanderContinuous-v3.
TD3 (Twin Delayed DDPG) was covered in Week 6 of the module.

Usage:
    python train_td3.py
"""

import gymnasium as gym
import numpy as np
import os
import time

from td3_agent import TD3Agent
from utils import (plot_training_rewards, save_results, print_training_summary)


def train_td3(agent, env, num_episodes=1000, max_steps=1000):
    """Train TD3 agent and collect metrics."""
    all_rewards = []

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.step(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if done:
                break

        all_rewards.append(episode_reward)

        if ep % 50 == 0:
            avg_reward = np.mean(all_rewards[-100:])
            print(f"Episode {ep}/{num_episodes} | "
                  f"Avg Reward (100): {avg_reward:.2f} | "
                  f"Buffer: {len(agent.memory)}")

        # check if solved
        if len(all_rewards) >= 100:
            recent_avg = np.mean(all_rewards[-100:])
            if recent_avg >= 200:
                print(f"\nSolved at episode {ep}! Average reward: {recent_avg:.2f}")
                break

    return all_rewards


def main():
    save_dir = os.path.join('..', 'results', 'td3')

    print(f"\n{'#'*60}")
    print(f"  Training TD3 on LunarLanderContinuous-v3")
    print(f"{'#'*60}\n")

    env = gym.make('LunarLanderContinuous-v3')

    agent = TD3Agent(
        state_dim=8,
        action_dim=2,
        seed=42,
        actor_lr=1e-3,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        buffer_size=200000,
        batch_size=100,
        policy_noise=0.2,
        noise_clip=0.5,
        exploration_noise=0.1,
        policy_delay=2,
        hidden_dim=128,
        learning_starts=5000
    )

    start_time = time.time()
    rewards = train_td3(agent, env, num_episodes=800)
    elapsed = time.time() - start_time

    print(f"\nTraining took {elapsed/60:.1f} minutes")

    # save model and results
    os.makedirs(save_dir, exist_ok=True)
    agent.save(os.path.join(save_dir, 'td3_model.pth'))

    config = {
        'algorithm': 'TD3',
        'actor_lr': 1e-3,
        'critic_lr': 1e-3,
        'gamma': 0.99,
        'tau': 0.005,
        'buffer_size': 200000,
        'batch_size': 100,
        'policy_noise': 0.2,
        'noise_clip': 0.5,
        'exploration_noise': 0.1,
        'policy_delay': 2,
        'hidden_dim': 128,
        'learning_starts': 5000,
        'seed': 42,
        'training_time_min': round(elapsed/60, 2),
        'total_episodes': len(rewards),
    }
    save_results(rewards, [], [], config, save_dir)

    plot_training_rewards(
        rewards,
        title='TD3 - Training Rewards (LunarLanderContinuous-v3)',
        save_path=os.path.join(save_dir, 'training_rewards.png')
    )

    print_training_summary(rewards, 'TD3')
    env.close()


if __name__ == '__main__':
    main()
