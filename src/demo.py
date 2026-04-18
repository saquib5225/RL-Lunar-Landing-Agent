"""
Live demo — run trained agents with visual rendering.

Usage:
    python demo.py              # demo all agents
    python demo.py --dqn        # DQN only
    python demo.py --ddqn       # Double DQN only
    python demo.py --td3        # TD3 only
    python demo.py --episodes 5 # number of episodes to show
"""

import gymnasium as gym
import numpy as np
import os
import argparse
import time

from dqn_agent import DQNAgent
from td3_agent import TD3Agent


def demo_dqn(model_path, double_dqn=False, episodes=3):
    algo = "Double DQN" if double_dqn else "DQN"
    print(f"\n{'='*40}")
    print(f"  {algo} Demo — LunarLander-v3")
    print(f"{'='*40}\n")

    env = gym.make('LunarLander-v3', render_mode='human')
    agent = DQNAgent(double_dqn=double_dqn)
    agent.load(model_path)
    agent.epsilon = 0.0

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        total_reward = 0

        for step in range(1000):
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        status = "LANDED" if total_reward >= 200 else "CRASHED" if total_reward < 0 else "PARTIAL"
        print(f"  Episode {ep}: reward = {total_reward:.1f}  [{status}]")
        time.sleep(0.5)

    env.close()


def demo_td3(model_path, episodes=3):
    print(f"\n{'='*40}")
    print(f"  TD3 Demo — LunarLanderContinuous-v3")
    print(f"{'='*40}\n")

    env = gym.make('LunarLanderContinuous-v3', render_mode='human')
    agent = TD3Agent(hidden_dim=128)
    agent.load(model_path)

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        total_reward = 0

        for step in range(1000):
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        status = "LANDED" if total_reward >= 200 else "CRASHED" if total_reward < 0 else "PARTIAL"
        print(f"  Episode {ep}: reward = {total_reward:.1f}  [{status}]")
        time.sleep(0.5)

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Live demo of trained RL agents')
    parser.add_argument('--dqn', action='store_true')
    parser.add_argument('--ddqn', action='store_true')
    parser.add_argument('--td3', action='store_true')
    parser.add_argument('--episodes', type=int, default=3, help='Episodes per agent')
    args = parser.parse_args()

    results_dir = os.path.join('..', 'results')
    run_all = not (args.dqn or args.ddqn or args.td3)

    if args.td3 or run_all:
        path = os.path.join(results_dir, 'td3', 'td3_model.pth')
        if os.path.exists(path):
            demo_td3(path, args.episodes)
        else:
            print("TD3 model not found. Train it first.")

    if args.dqn or run_all:
        path = os.path.join(results_dir, 'dqn', 'dqn_model.pth')
        if os.path.exists(path):
            demo_dqn(path, double_dqn=False, episodes=args.episodes)
        else:
            print("DQN model not found. Train it first.")

    if args.ddqn or run_all:
        path = os.path.join(results_dir, 'ddqn', 'ddqn_model.pth')
        if os.path.exists(path):
            demo_dqn(path, double_dqn=True, episodes=args.episodes)
        else:
            print("Double DQN model not found. Train it first.")

    print("\nDemo complete.")
