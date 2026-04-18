"""
Evaluate trained models and generate analysis plots.

Usage:
    python evaluate.py              # evaluate all models
    python evaluate.py --dqn        # evaluate DQN only
    python evaluate.py --ddqn       # evaluate Double DQN only
    python evaluate.py --td3        # evaluate TD3 only
"""

import gymnasium as gym
import numpy as np
import torch
import os
import argparse

from dqn_agent import DQNAgent
from td3_agent import TD3Agent
from utils import (plot_eval_histogram, plot_comparison,
                   plot_training_rewards, print_training_summary)


def evaluate_dqn(model_path, double_dqn=False, num_episodes=100, render=False):
    """Run evaluation episodes with a trained DQN/DDQN model."""
    algo_name = "Double DQN" if double_dqn else "DQN"
    print(f"\nEvaluating {algo_name}...")

    env = gym.make('LunarLander-v3', render_mode='human' if render else None)

    agent = DQNAgent(double_dqn=double_dqn)
    agent.load(model_path)
    agent.epsilon = 0.0  # no exploration during eval

    eval_rewards = []
    for ep in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0

        for step in range(1000):
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        eval_rewards.append(total_reward)

    env.close()

    mean_r = np.mean(eval_rewards)
    std_r = np.std(eval_rewards)
    success_rate = sum(1 for r in eval_rewards if r >= 200) / len(eval_rewards) * 100

    print(f"  {algo_name} Evaluation ({num_episodes} episodes):")
    print(f"    Mean reward:  {mean_r:.2f} +/- {std_r:.2f}")
    print(f"    Best:         {max(eval_rewards):.2f}")
    print(f"    Worst:        {min(eval_rewards):.2f}")
    print(f"    Success rate: {success_rate:.1f}%")

    return eval_rewards


def evaluate_td3(model_path, num_episodes=100, render=False):
    """Run evaluation episodes with a trained TD3 model."""
    print(f"\nEvaluating TD3...")

    env = gym.make('LunarLanderContinuous-v3', render_mode='human' if render else None)

    agent = TD3Agent(hidden_dim=128)
    agent.load(model_path)

    eval_rewards = []
    for ep in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0

        for step in range(1000):
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        eval_rewards.append(total_reward)

    env.close()

    mean_r = np.mean(eval_rewards)
    std_r = np.std(eval_rewards)
    success_rate = sum(1 for r in eval_rewards if r >= 200) / len(eval_rewards) * 100

    print(f"  TD3 Evaluation ({num_episodes} episodes):")
    print(f"    Mean reward:  {mean_r:.2f} +/- {std_r:.2f}")
    print(f"    Best:         {max(eval_rewards):.2f}")
    print(f"    Worst:        {min(eval_rewards):.2f}")
    print(f"    Success rate: {success_rate:.1f}%")

    return eval_rewards


def generate_all_plots():
    """Load saved training data and generate all comparison plots."""
    results_dir = os.path.join('..', 'results')

    # load training rewards if available
    rewards_dict = {}
    for algo, folder in [('DQN', 'dqn'), ('Double DQN', 'ddqn'), ('TD3', 'td3')]:
        rewards_path = os.path.join(results_dir, folder, 'rewards.npy')
        if os.path.exists(rewards_path):
            rewards_dict[algo] = np.load(rewards_path)
            plot_training_rewards(
                rewards_dict[algo],
                title=f'{algo} - Training Rewards',
                save_path=os.path.join(results_dir, folder, 'training_rewards.png')
            )

    # DQN vs DDQN comparison (discrete)
    if 'DQN' in rewards_dict and 'Double DQN' in rewards_dict:
        plot_comparison(
            {'DQN': rewards_dict['DQN'], 'Double DQN': rewards_dict['Double DQN']},
            title='DQN vs Double DQN - Training Comparison',
            save_path=os.path.join(results_dir, 'dqn_vs_ddqn_comparison.png')
        )

    # all three
    if len(rewards_dict) == 3:
        plot_comparison(
            rewards_dict,
            title='All Algorithms - Training Comparison',
            save_path=os.path.join(results_dir, 'all_algorithms_comparison.png')
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dqn', action='store_true')
    parser.add_argument('--ddqn', action='store_true')
    parser.add_argument('--td3', action='store_true')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--plots-only', action='store_true', help='Just regenerate plots')
    args = parser.parse_args()

    results_dir = os.path.join('..', 'results')

    if args.plots_only:
        generate_all_plots()
    else:
        run_all = not (args.dqn or args.ddqn or args.td3)

        if args.dqn or run_all:
            model_path = os.path.join(results_dir, 'dqn', 'dqn_model.pth')
            if os.path.exists(model_path):
                dqn_rewards = evaluate_dqn(model_path, double_dqn=False, render=args.render)
                plot_eval_histogram(dqn_rewards, title='DQN Evaluation Rewards',
                                    save_path=os.path.join(results_dir, 'dqn', 'eval_histogram.png'))
            else:
                print(f"DQN model not found at {model_path}. Train it first.")

        if args.ddqn or run_all:
            model_path = os.path.join(results_dir, 'ddqn', 'ddqn_model.pth')
            if os.path.exists(model_path):
                ddqn_rewards = evaluate_dqn(model_path, double_dqn=True, render=args.render)
                plot_eval_histogram(ddqn_rewards, title='Double DQN Evaluation Rewards',
                                    save_path=os.path.join(results_dir, 'ddqn', 'eval_histogram.png'))
            else:
                print(f"Double DQN model not found at {model_path}. Train it first.")

        if args.td3 or run_all:
            model_path = os.path.join(results_dir, 'td3', 'td3_model.pth')
            if os.path.exists(model_path):
                td3_rewards = evaluate_td3(model_path, render=args.render)
                plot_eval_histogram(td3_rewards, title='TD3 Evaluation Rewards',
                                    save_path=os.path.join(results_dir, 'td3', 'eval_histogram.png'))
            else:
                print(f"TD3 model not found at {model_path}. Train it first.")

        generate_all_plots()
